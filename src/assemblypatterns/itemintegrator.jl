
"""
$(TYPEDEF)

itemintegrator assembly pattern type
"""
abstract type APT_ItemIntegrator <: AssemblyPatternType end

function Base.show(io::IO, ::Type{APT_ItemIntegrator})
    print(io, "ItemIntegrator")
end

"""
````
function ItemIntegrator(
    T::Type{<:Real},
    AT::Type{<:AbstractAssemblyType},
    operators::Array{DataType,1}, 
    action::AbstractAction; 
    name = "ItemIntegrator",
    regions::Array{Int,1} = [0])
````

Creates an ItemIntegrator assembly pattern with the given operators and action etc.
"""
function ItemIntegrator(T::Type{<:Real}, AT::Type{<:AbstractAssemblyType}, operators, action = NoAction(); regions = [0], name = "ItemIntegrator")
    return AssemblyPattern{APT_ItemIntegrator, T, AT}(name,[],operators,action,1:length(operators),regions)
end

"""
````
function L2ErrorIntegrator(
    T::Type{<:Real},
    compare_data::UserData{AbstractDataFunction}, # can be omitted if zero
    operator::Type{<:AbstractFunctionOperator};
    quadorder = "auto",
    name = "auto",
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    time = 0)
````

Creates an ItemIntegrator that compares FEVectorBlock operator-evaluations against the given compare_data and returns the L2-error.
"""
function L2ErrorIntegrator(
    T::Type{<:Real},
    compare_data::UserData{AbstractDataFunction},
    operator::Type{<:AbstractFunctionOperator};
    quadorder = "auto",
    name = "auto",
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
    if quadorder == "auto"
        quadorder = 2 * compare_data.quadorder
    end
    if name == "auto"
        name = "L2 error ($(compare_data.name))"
    end
    action_kernel = ActionKernel(L2error_function, [1,compare_data.dimensions[1]]; name = name, dependencies = "X", quadorder = quadorder)
    return ItemIntegrator(T,AT, [operator], Action(T, action_kernel); regions = regions, name = name)
end

"""
````
L2NormIntegrator(
    T::Type{<:Real},
    ncomponents::Int,
    operator::Type{<:AbstractFunctionOperator};
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    name = "L2 norm",
    quadorder = 2,
    regions = [0])
````

Creates an ItemIntegrator that computes the L2 norm of an operator evaluation where ncomponents is the expected length of the operator evaluation.
"""
function L2NormIntegrator(
    T::Type{<:Real},
    ncomponents::Int,
    operator::Type{<:AbstractFunctionOperator};
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    name = "L2 norm",
    quadorder = 2,
    regions = [0])
    function L2norm_function(result,input)
        result[1] = 0
        for j=1:ncomponents
            result[1] += input[j]^2
        end    
    end    
    action_kernel = ActionKernel(L2norm_function, [1,ncomponents]; name = "L2 norm kernel", dependencies = "", quadorder = quadorder)
    return ItemIntegrator(T,AT, [operator], Action(T, action_kernel); regions = regions, name = name)
end

"""
````
function L2DifferenceIntegrator(
    T::Type{<:Real},
    ncomponents::Int,
    operator::Union{Type{<:AbstractFunctionOperator},Array{DataType,1}};
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    name = "L2 difference",
    quadorder = 2,
    regions = [0])
````

Creates an ItemIntegrator that computes the L2 norm difference between two arguments evalauted with the same operator (or with different operators if operator is an array) where ncomponents is the expected length of each operator evaluation.
Note that all arguments in an evaluation call need to be defined on the same grid !
"""
function L2DifferenceIntegrator(
    T::Type{<:Real},
    ncomponents::Int,
    operator;
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    name = "L2 difference",
    quadorder = 2,
    regions = [0])
    function L2difference_function(result,input)
        result[1] = 0
        for j=1:ncomponents
            result[1] += (input[j]-input[ncomponents+j])^2
        end    
    end    
    action_kernel = ActionKernel(L2difference_function, [1,2*ncomponents]; name = "L2 difference kernel", dependencies = "", quadorder = quadorder)
    if typeof(operator) <: DataType
        return ItemIntegrator(T,AT, [operator, operator], Action(T, action_kernel); regions = regions, name = name)
    else
        return ItemIntegrator(T,AT, [operator[1], operator[2]], Action(T, action_kernel); regions = regions, name = name)
    end
end



"""
````
function evaluate!(
    b::AbstractArray{T,2},          # target vector
    AP::AssemblyPattern{APT,T,AT},  # ItemIntegrator pattern
    FEB::Array{<:FEVectorBlock,1}   # coefficients for arguments
    where {APT <: APT_ItemIntegrator, T, AT}
````

Evaluation of an ItemIntegrator assembly pattern with given FEVectorBlocks FEB into given two-dimensional Array b.
"""
function evaluate!(
    b::AbstractArray{T,2},
    AP::AssemblyPattern{APT,T,AT},
    FEB::Array{<:FEVectorBlock,1};
    skip_preps::Bool = false) where {APT <: APT_ItemIntegrator, T<: Real, AT <: AbstractAssemblyType}

    # prepare assembly
    nFE = length(FEB)
    if !skip_preps
        FE = Array{FESpace,1}(undef, nFE)
        for j = 1 : nFE
            FE[j] = FEB[j].FES
        end
        @assert length(FEB) == length(AP.operators)
        prepare_assembly!(AP; FES = FE)
    end
    AM::AssemblyManager{T} = AP.AM
    xItemVolumes::Array{T,1} = FEB[1].FES.xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemRegions::Union{VectorOfConstants{Int32}, Array{Int32,1}} = FEB[1].FES.xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = length(xItemVolumes)

    # prepare action
    action = AP.action
    maxnweights = get_maxnqweights(AM)
    if typeof(action) <: NoAction
        action_resultdim = size(get_basisevaler(AM, 1, 1).cvals,1)
        action_input = Array{Array{T,1},1}(undef,maxnweights)
        if length(FEB) > 1
            @warn "NoAction() used in ItemIntegrator with more than one argument!"
        end
        for j = 1 : maxnweights
            action_input[j] = zeros(T,action_resultdim) # heap for action input
        end
    else
        action_resultdim::Int = action.argsizes[1]
        action_input = Array{Array{T,1},1}(undef,maxnweights)
        for j = 1 : maxnweights
            action_input[j] = zeros(T,action.argsizes[2]) # heap for action input
        end
    end
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output

    if AP.regions != [0]
        @logmsg MoreInfo "Evaluating $(AP.name) for $((p->p.name).(FEB)) ($AT in regions = $(AP.regions))"
    else
        @logmsg MoreInfo "Evaluating $(AP.name) for $((p->p.name).(FEB)) ($AT)"
    end
    @debug AP

    # loop over items
    offsets = zeros(Int,nFE+1)
    maxdofs = get_maxndofs(AM)
    basisevaler::Array{FEBasisEvaluator,1} = Array{FEBasisEvaluator,1}(undef,nFE)
    for j = 1 : nFE
        basisevaler[j] = get_basisevaler(AM, j, 1)
        offsets[j+1] = offsets[j] + size(basisevaler[j].cvals,1)
    end
    basisxref::Array{Array{T,1},1} = basisevaler[1].xref
    maxdofitems::Array{Int,1} = get_maxdofitems(AM)
    coeffs::Array{T,1} = zeros(T,sum(maxdofs[1:end]))
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
                    basisevaler[FEid] = get_basisevaler(AM, FEid, di)
    
                    # update action on dofitem
                    update!(action, basisevaler[FEid], AM.dofitems[FEid][di], item, regions[r])

                    # get coefficients of FE number FEid on current dofitem
                    get_coeffs!(coeffs, FEB[FEid], AM, FEid, di)
                    coeffs .*= AM.coeff4dofitem[FEid][di]

                    # write evaluation of operator of current FE into action_input
                    for i in eachindex(weights)
                        eval!(action_input[i], basisevaler[FEid], coeffs, i, offsets[FEid])
                    end  
                end
            end
        end

        if typeof(action) <: NoAction
            for i in eachindex(weights)
                for j = 1 : action_resultdim
                    b[j,item] += action_input[i][j] * weights[i] * xItemVolumes[item]
                end
                fill!(action_input[i], 0)
            end  
        else
            # update action on item/dofitem (of first operator)
            basisxref = basisevaler[1].xref
            update!(action, basisevaler[1], AM.dofitems[1][1], item, regions[r])
            # apply action to FEVector and accumulate
            for i in eachindex(weights)
                apply_action!(action_result, action_input[i], action, i, basisxref[i])
                for j = 1 : action_resultdim
                    b[j,item] += action_result[j] * weights[i] * xItemVolumes[item]
                end
                fill!(action_input[i], 0)
            end  
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
    AP::AssemblyPattern{APT,T,AT},  # ItemIntegrator pattern
    FEB::Array{<:FEVectorBlock,1})  # coefficients for arguments
    where {APT <: APT_ItemIntegrator, T, AT}

````

Evaluation of an ItemIntegrator assembly pattern with given FEVectorBlocks FEB, only returns accumulation over all items.
"""
function evaluate(
    AP::AssemblyPattern{APT,T,AT},
    FEB;
    skip_preps::Bool = false) where {APT <: APT_ItemIntegrator, T<: Real, AT <: AbstractAssemblyType}

    if typeof(FEB) <: FEVectorBlock
        FEB = [FEB]
    end

    # prepare assembly
    nFE = length(FEB)
    if !skip_preps
        FE = Array{FESpace,1}(undef, nFE)
        for j = 1 : nFE
            FE[j] = FEB[j].FES
        end
        @assert length(FEB) == length(AP.operators)
        prepare_assembly!(AP; FES = FE)
    end
    AM::AssemblyManager{T} = AP.AM

    # quick and dirty : we mask the resulting array as an AbstractArray{T,2} using AccumulatingVector
    # and use the itemwise evaluation above
    if typeof(AP.action) <: NoAction
        resultdim = size(get_basisevaler(AM, 1, 1).cvals,1)
    else
        resultdim = AP.action.argsizes[1]
    end
    AV = AccumulatingVector{T}(zeros(T,resultdim), 0)

    evaluate!(AV, AP, FEB; skip_preps = false)
    
    if resultdim == 1
        return AV.entries[1]
    else
        return AV.entries
    end
end