abstract type APT_ItemIntegrator <: AssemblyPatternType end

"""
$(TYPEDEF)

creates an item integrator that can e.g. be used for error/norm evaluations
"""
function ItemIntegrator(T::Type{<:Real}, AT::Type{<:AbstractAssemblyType}, operators, action; regions = [0])
    return AssemblyPattern{APT_ItemIntegrator, T, AT}([],operators,action,regions,AssemblyPatternPreparations(nothing,nothing,nothing,nothing,nothing))
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

Creates an item integrator that compares FEVectorBlock operator-evaluations against the given compare_data and returns the L2-error.
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
    action_kernel = ActionKernel(L2error_function, [1,compare_data.dimensions[2]]; name = "L2 error kernel", dependencies = "X", quadorder = 2 * compare_data.quadorder)
    return ItemIntegrator(T,AT, [operator], Action(T, action_kernel); regions = regions)
end
function L2ErrorIntegrator(T::Type{<:Real},
    ncomponents::Int,
    operator::Type{<:AbstractFunctionOperator};
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    regions = [0],
    time = 0)
    call = 0
    function L2error_function(result,input)
        result[1] = 0
        call += 1
        for j=1:ncomponents
            result[1] += input[j]^2
        end    
    end    
    action_kernel = ActionKernel(L2error_function, [1,ncomponents]; name = "L2 error kernel", dependencies = "", quadorder = 2)
    return ItemIntegrator(T,AT, [operator], Action(T, action_kernel); regions = regions)
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

    # get adjacencies
    operators = AP.operators
    @assert length(FEB) == length(operators)
    nFE = length(FEB)
    FE = Array{FESpace,1}(undef, nFE)
    xItemDofs = Array{Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}},1}(undef, nFE)
    for j = 1 : nFE
        FE[j] = FEB[j].FES
        xItemDofs[j] = Dofmap4AssemblyType(FE[j], DofitemAT4Operator(AT, operators[j]))
    end
    xItemVolumes::Array{T,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemRegions::Union{VectorOfConstants{Int32}, Array{Int32,1}} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = length(xItemVolumes)

    # prepare assembly
    action = AP.action
    if !skip_preps
        prepare_assembly!(AP; FES = FE, verbosity = verbosity - 1)
    end
    EG = AP.APP.EG
    ndofs4EG::Array{Array{Int,1},1} = AP.APP.ndofs4EG
    qf::Array{QuadratureRule,1} = AP.APP.qf
    basisevaler::Array{FEBasisEvaluator,4} = AP.APP.basisevaler
    dii4op::Array{Function,1} = AP.APP.dii4op

    # get size informations
    ncomponents = zeros(Int,nFE)
    offsets = zeros(Int,nFE+1)
    maxdofs = 0
    for j = 1 : nFE
        ncomponents[j] = get_ncomponents(eltype(FE[j]))
        maxdofs = max(maxdofs, max_num_targets_per_source(xItemDofs[j]))
        offsets[j+1] = offsets[j] + size(basisevaler[end,j,1,1].cvals,1)
    end
    action_resultdim::Int = action.argsizes[1]
    maxdofs2 = max_num_targets_per_source(xItemDofs[end])

    if verbosity > 0
        println("  Evaluating ($APT,$AT,$T)")
        println("   skip_preps = $skip_preps")
        println("    operators = $(AP.operators)")
        println("      regions = $(AP.regions)")
        println("       action = $(AP.action.name) (size = $(action.argsizes))")
        println("        qf[1] = $(qf[1].name) ")
        println("           EG = $EG")
    end

    maxnweights = 0
    for j = 1 : length(qf)
        maxnweights = max(maxnweights, length(qf[j].w))
    end
    action_input = Array{Array{T,1},1}(undef,maxnweights)
    for j = 1 : maxnweights
        action_input[j] = zeros(T,offsets[end]) # heap for action input
    end
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output

    # loop over items
    EG4item::Int = 0
    EG4dofitem::Array{Int,1} = [1,1] # type of the current item
    ndofs4dofitem::Int = 0 # number of dofs for item
    dofitems::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found
    itempos4dofitem::Array{Int,1} = [1,1] # local item position in dofitem
    orientation4dofitem::Array{Int,1} = [1,2] # local orientation
    coefficient4dofitem::Array{T,1} = [0.0,0.0]
    dofitem::Int = 0
    coeffs = zeros(T,maxdofs)
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations
    basisevaler4dofitem::FEBasisEvaluator = basisevaler[1,1,1,1]
    regions::Array{Int,1} = AP.regions
    allitems::Bool = (regions == [0])
    nregions::Int = length(regions)
    for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if allitems || xItemRegions[item] == regions[r]

        # get dofitem informations
        EG4item = dii4op[1](dofitems, EG4dofitem, itempos4dofitem, coefficient4dofitem, orientation4dofitem, item)

        if dofitems[1] == 0
            break;
        end

        # get information on dofitems
        weights = qf[EG4item].w
        for di = 1 : length(dofitems)
            dofitem = dofitems[di]
            if dofitem != 0
                for FEid = 1 : nFE
                    # update FEbasisevaler on dofitem
                    basisevaler4dofitem = basisevaler[EG4dofitem[di],FEid,itempos4dofitem[di],orientation4dofitem[di]]
                    update!(basisevaler4dofitem, dofitem)

                    # update coeffs on dofitem
                    ndofs4dofitem = ndofs4EG[FEid][EG4dofitem[di]]
                    for j=1:ndofs4dofitem
                        fdof = xItemDofs[FEid][j,dofitem]
                        coeffs[j] = FEB[FEid][fdof]
                    end

                    for i in eachindex(weights)
                        if FEid == 1 && di == 1
                            fill!(action_input[i], 0)
                        end
                        eval!(action_input[i], basisevaler4dofitem, coeffs, i; offset = offsets[FEid], factor = coefficient4dofitem[di])
                    end  
                end
            end
        end

        # update action on item/dofitem
       # basisevaler4dofitem = basisevaler[EG4item[1],1,1,1]
        update!(action, basisevaler4dofitem, dofitems[1], item, regions[r])

        for i in eachindex(weights)
            # apply action to FEVector
            apply_action!(action_result, action_input[i], action, i)
            for j = 1 : action_resultdim
                b[j,item] += action_result[j] * weights[i] * xItemVolumes[item]
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



