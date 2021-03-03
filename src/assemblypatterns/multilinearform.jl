abstract type APT_MultilinearForm <: AssemblyPatternType end


"""
````
function MultilinearForm(
    T::Type{<:Real},
    AT::Type{<:AbstractAssemblyType},
    FE::Array{FESpace,1},
    operators::Array{DataType,1}, 
    action::AbstractAction; 
    regions::Array{Int,1} = [0])
````

Creates a MultilinearForm assembly pattern with the given FESpaces, operators and action etc.
"""
function MultilinearForm(
    T::Type{<:Real},
    AT::Type{<:AbstractAssemblyType},
    FE::Array{FESpace,1},
    operators::Array{DataType,1}, 
    action::AbstractAction; 
    regions::Array{Int,1} = [0])

    @assert length(FE) == length(operators)
    return AssemblyPattern{APT_MultilinearForm, T, AT}(FE,operators,action,regions)
end

"""
````
assemble!(
    assemble!(
    b::AbstractVector,
    FE::Array{<:FEVectorBlock,1},
    AP::AssemblyPattern{APT,T,AT};
    verbosity::Int = 0,
    factor = 1)
````

Assembly of a MultilinearForm AP into given one-dimensional AbstractArray (e.g. a FEVectorBlock).
Here, the all but the last arguments are fixed by the given coefficients in the components of FE.
"""
function assemble!(
    b::AbstractVector,
    FEB::Array{AbstractVector,1},
    AP::AssemblyPattern{APT,T,AT};
    verbosity::Int = 0,
    skip_preps::Bool = false,
    factor = 1,
    offset = 0,
    offsets2 = [0]) where {APT <: APT_MultilinearForm, T <: Real, AT <: AbstractAssemblyType}

    # prepare assembly
    FE = AP.FES
    action = AP.action
    if !skip_preps
        prepare_assembly!(AP; verbosity = verbosity - 1)
    end
    AM::AssemblyManager{T} = AP.AM
    xItemVolumes::Array{T,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemRegions::Union{VectorOfConstants{Int32}, Array{Int32,1}} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
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
        println("  Assembling ($APT,$AT,$T) into vector with fixed_arguments = [1:$(nFE-1)]")
        println("   skip_preps = $skip_preps")
        println("    operators = $(AP.operators)")
        println("      regions = $(AP.regions)")
        println("       factor = $factor")
        println("       action = $(AP.action.name) (apply_to = [1:$(nFE-1)], size = $(action.argsizes))")
        println("        qf[1] = $(AM.qf[1].name) ")
        
    end

    # loop over items
    offsets = zeros(Int,nFE+1)
    maxdofs = get_maxndofs(AM)
    basisevaler::FEBasisEvaluator = get_basisevaler(AM, 1, 1)
    basisvals::Array{T,3} = basisevaler[1].cvals
    for j = 1 : nFE
        basisevaler = get_basisevaler(AM, j, 1)
        offsets[j+1] = offsets[j] + size(basisevaler.cvals,1)
    end
    maxdofitems::Array{Int,1} = get_maxdofitems(AM)
    coeffs = zeros(T,sum(maxdofs[1:end]))
    weights::Array{T,1} = get_qweights(AM)
    itemfactor::T = 0
    bdof::Int = 0
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

        # assemble all but the last operators into action_input
        for FEid = 1 : nFE - 1
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
        
        # multiply last operator of testfunction
        for di = 1 : maxdofitems[end]
            if AM.dofitems[end][di] != 0
                basisevaler = get_basisevaler(AM, nFE, di[nFE])
                basisvals = basisevaler.cvals
                update!(action, basisevaler4dofitem, AM.dofitems[nFE][di], item, regions[r])
                for i in eachindex(weights)
                    # apply action
                    apply_action!(action_result, action_input[i], action, i)
        
                    # multiply last component
                    for dof_j = 1 : get_ndofs(AM, nFE, di)
                        temp = 0
                        for k = 1 : action_resultdim
                            temp += action_result[k] * basisvals[k,dof_j,i]
                        end
                        localb[dof_j] += temp * weights[i]
                    end 
                end  
        
                itemfactor = xItemVolumes[item] * factor * AM.coeff4dofitem[nFE][di]
                for dof_i = 1 : get_ndofs(AM, nFE, di)
                    bdof = get_dof(AM, nFE, di, dof_i) + offset
                    b[bdof] += localb[dof_i] * itemfactor
                end
            end
        end
        
        fill!(localb, 0.0)

        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
    return nothing
end


function assemble!(
    b::FEVectorBlock,
    FEB::Array{<:FEVectorBlock,1},
    AP::AssemblyPattern{APT,T,AT};
    verbosity::Int = 0,
    skip_preps::Bool = false,
    factor = 1) where {APT <: APT_MultilinearForm, T <: Real, AT <: AbstractAssemblyType}

    FEBarrays = Array{AbstractVector,1}(undef, length(FEB))
    offsets = zeros(Int,length(FEB))
    for j = 1 : length(FEB)
        FEBarrays[j] = FEB[j].entries
        offsets[j] = FEB[j].offset
    end

    assemble!(b.entries, FEBarrays, AP; verbosity = verbosity, factor = factor, offset = b.offset, offsets2 = offsets, skip_preps = skip_preps)
end