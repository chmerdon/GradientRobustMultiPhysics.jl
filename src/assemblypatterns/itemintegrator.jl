abstract type APT_ItemIntegrator <: AssemblyPatternType end

"""
````
function ItemIntegrator(
    T::Type{<:Real},
    AT::Type{<:AbstractAssemblyType},
    operators::Array{DataType,1}, 
    action::AbstractAction; 
    regions::Array{Int,1} = [0])
````

Creates an ItemIntegrator assembly pattern with the given operators and action etc.
"""
function ItemIntegrator(T::Type{<:Real}, AT::Type{<:AbstractAssemblyType}, operators, action; regions = [0])
    return AssemblyPattern{APT_ItemIntegrator, T, AT}([],operators,action,regions)
end

"""
````
function L2ErrorIntegrator(
    T::Type{<:Real},
    compare_data::UserData{AbstractDataFunction}, # can be omitted if zero
    operator::Type{<:AbstractFunctionOperator};
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    time = 0)
````

Creates an ItemIntegrator that compares FEVectorBlock operator-evaluations against the given compare_data and returns the L2-error.
"""
function L2ErrorIntegrator(T::Type{<:Real},
    compare_data::UserData{AbstractDataFunction},
    operator::Type{<:AbstractFunctionOperator};
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    regions = [0],
    time = 0)

    ncomponents::Int = compare_data.dimensions[1]
    temp = zeros(T,ncomponents)
    function L2error_function(result,input,x)
        eval!(temp,compare_data,x,time)
        result[1] = 0
        for j=1:ncomponents
            result[1] += (temp[j] - input[j])^2
        end    
    end    
    action_kernel = ActionKernel(L2error_function, [1,compare_data.dimensions[1]]; name = "L2 error kernel", dependencies = "X", quadorder = 2 * compare_data.quadorder)
    return ItemIntegrator(T,AT, [operator], Action(T, action_kernel); regions = regions)
end
function L2NormIntegrator(T::Type{<:Real},
    ncomponents::Int,
    operator::Type{<:AbstractFunctionOperator};
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    regions = [0])
    function L2norm_function(result,input)
        result[1] = 0
        for j=1:ncomponents
            result[1] += input[j]^2
        end    
    end    
    action_kernel = ActionKernel(L2norm_function, [1,ncomponents]; name = "L2 norm kernel", dependencies = "", quadorder = 2)
    return ItemIntegrator(T,AT, [operator], Action(T, action_kernel); regions = regions)
end
function L2DifferenceIntegrator(T::Type{<:Real},
    ncomponents::Int,
    operator::Type{<:AbstractFunctionOperator};
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    regions = [0])
    function L2difference_function(result,input)
        result[1] = 0
        for j=1:ncomponents
            result[1] += (input[j]-input[ncomponents+j])^2
        end    
    end    
    action_kernel = ActionKernel(L2difference_function, [1,2*ncomponents]; name = "L2 difference kernel", dependencies = "", quadorder = 2)
    return ItemIntegrator(T,AT, [operator, operator], Action(T, action_kernel); regions = regions)
end



"""
````
function evaluate!(
    b::AbstractArray{T,2},
    AP::AssemblyPattern{APT,T,AT},
    FEB::FEVectorBlock;
    verbosity::Int = 0) where {APT <: APT_ItemIntegrator, T<: Real, AT <: AbstractAssemblyType}
````

Evaluation of an ItemIntegrator assembly pattern with given FEVectorBlocks FEB into given two-dimensional Array b.
"""
function evaluate!(
    b::AbstractArray{T,2},
    AP::AssemblyPattern{APT,T,AT},
    FEB::Array{<:FEVectorBlock,1};
    skip_preps::Bool = false,
    verbosity::Int = 0) where {APT <: APT_ItemIntegrator, T<: Real, AT <: AbstractAssemblyType}

    # prepare assembly
    nFE = length(FEB)
    if !skip_preps
        FE = Array{FESpace,1}(undef, nFE)
        for j = 1 : nFE
            FE[j] = FEB[j].FES
        end
        @assert length(FEB) == length(AP.operators)
        prepare_assembly!(AP; FES = FE, verbosity = verbosity - 1)
    end
    AM::AssemblyManager{T} = AP.AM
    xItemVolumes::Array{T,1} = FEB[1].FES.xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemRegions::Union{VectorOfConstants{Int32}, Array{Int32,1}} = FEB[1].FES.xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = length(xItemVolumes)

    # prepare action
    action = AP.action
    action_resultdim::Int = action.argsizes[1]
    maxnweights = get_maxnqweights(AM)
    action_input = Array{Array{T,1},1}(undef,maxnweights)
    for j = 1 : maxnweights
        action_input[j] = zeros(T,action.argsizes[2]) # heap for action input
    end
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output

    if verbosity > 0
        println("  Evaluating ($APT,$AT,$T)")
        println("   skip_preps = $skip_preps")
        println("    operators = $(AP.operators)")
        println("      regions = $(AP.regions)")
        println("       action = $(AP.action.name) (size = $(action.argsizes))")
        println("        qf[1] = $(AM.qf[1].name) ")
        
    end

    # loop over items
    offsets = zeros(Int,nFE+1)
    maxdofs = get_maxndofs(AM)
    basisevaler::FEBasisEvaluator = get_basisevaler(AM, 1, 1)
    for j = 1 : nFE
        basisevaler = get_basisevaler(AM, j, 1)
        offsets[j+1] = offsets[j] + size(basisevaler.cvals,1)
    end
    maxdofitems::Array{Int,1} = get_maxdofitems(AM)
    coeffs = zeros(T,sum(maxdofs[1:end]))
    weights::Array{T,1} = get_qweights(AM)
    regions::Array{Int,1} = AP.regions
    allitems::Bool = (regions == [0])
    nregions::Int = length(regions)
    for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if allitems || xItemRegions[item] == regions[r]

        # update assembly manager (also updates necessary basisevaler)
        update!(AP.AM, item)
        weights = get_qweights(AM)

        # assemble all operators into action_input
        for FEid = 1 : nFE
            for di = 1 : maxdofitems[FEid]
                if AM.dofitems[FEid][di] != 0
                    # get correct basis evaluator for dofitem (was already updated by AM)
                    basisevaler = get_basisevaler(AM, FEid, di)
    
                    # update action on dofitem
                    update!(action, basisevaler, AM.dofitems[FEid][di], item, regions[r])

                    # get coefficients of FE number FEid on current dofitem
                    get_coeffs!(coeffs, FEB[FEid], AM, FEid, di)
                    coeffs .*= AM.coeff4dofitem[FEid][di]

                    # write evaluation of operator of current FE into action_input
                    for i in eachindex(weights)
                        eval!(action_input[i], basisevaler, coeffs, i, offsets[FEid])
                    end  
                end
            end
        end

        # update action on item/dofitem (of first operator)
        basisevaler4dofitem = get_basisevaler(AM, 1, 1)
        update!(action, basisevaler4dofitem, AM.dofitems[1][1], item, regions[r])

        # apply action to FEVector and accumulate
        for i in eachindex(weights)
            apply_action!(action_result, action_input[i], action, i)
            for j = 1 : action_resultdim
                b[j,item] += action_result[j] * weights[i] * xItemVolumes[item]
            end
            fill!(action_input[i], 0)
        end  

        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
    return nothing
end

"""
````
function evaluate(
    AP::AssemblyPattern{APT,T,AT},
    FEB::Array{<:FEVectorBlock,1};
    verbosity::Int = 0) where {APT <: APT_ItemIntegrator, T<: Real, AT <: AbstractAssemblyType}

````

Evaluation of an ItemIntegrator assembly pattern with given FEVectorBlocks FEB, only returns accumulation over all items.
"""
function evaluate(
    AP::AssemblyPattern{APT,T,AT},
    FEB;
    skip_preps::Bool = false,
    verbosity::Int = 0) where {APT <: APT_ItemIntegrator, T<: Real, AT <: AbstractAssemblyType}

    # quick and dirty : we mask the resulting array as an AbstractArray{T,2} using AccumulatingVector
    # and use the itemwise evaluation above
    resultdim = AP.action.argsizes[1]
    AV = AccumulatingVector{T}(zeros(T,resultdim), 0)

    if typeof(FEB) <: Array{<:FEVectorBlock,1}
        evaluate!(AV, AP, FEB; verbosity = verbosity, skip_preps = skip_preps)
    else
        evaluate!(AV, AP, [FEB]; verbosity = verbosity, skip_preps = skip_preps)
    end

    if resultdim == 1
        return AV.entries[1]
    else
        return AV.entries
    end
end