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

Creates a MultilinearForm that can be only assembled into a vector (with all but one fixed argument which currently is always the last one).
"""
function MultilinearForm(
    T::Type{<:Real},
    AT::Type{<:AbstractAssemblyType},
    FE::Array{FESpace,1},
    operators::Array{DataType,1}, 
    action::AbstractAction; 
    regions::Array{Int,1} = [0])

    @assert length(FE) == length(operators)
    return AssemblyPattern{APT_MultilinearForm, T, AT}(FE,operators,action,regions,AssemblyPatternPreparations(nothing,nothing,nothing,nothing,nothing))
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

    # get adjacencies
    FE = AP.FES
    xItemVolumes::Array{T,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemDofs = Array{Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}},1}(undef,length(FE))
    for j = 1 : length(FE)
        xItemDofs[j] = Dofmap4AssemblyType(FE[j], DofitemAT4Operator(AT, AP.operators[j]))
    end
    xItemRegions::Union{VectorOfConstants{Int32}, Array{Int32,1}} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = length(xItemVolumes)

    # prepare assembly
    action = AP.action
    if !skip_preps
        prepare_assembly!(AP; verbosity = verbosity - 1)
    elseif verbosity > 0
        println("warning: skipping assembly preparations for $APT")
    end
    EG = AP.APP.EG
    ndofs4EG = AP.APP.ndofs4EG
    qf = AP.APP.qf
    basisevaler = AP.APP.basisevaler
    dii4op = AP.APP.dii4op

    # get size informations
    ncomponents = zeros(Int,length(FE))
    offsets = zeros(Int,length(FE)+1)
    maxdofs = 0
    for j = 1 : length(FE)
        ncomponents[j] = get_ncomponents(eltype(FE[j]))
        maxdofs = max(maxdofs, max_num_targets_per_source(xItemDofs[j]))
        offsets[j+1] = offsets[j] + size(basisevaler[1,j,1,1].cvals,1)
    end
    action_resultdim::Int = action.argsizes[1]

    maxnweights = 0
    for j = 1 : length(qf)
        maxnweights = max(maxnweights, length(qf[j].w))
    end
    action_input = Array{Array{T,1},1}(undef,maxnweights)
    for j = 1 : maxnweights
        action_input[j] = zeros(T,offsets[end]) # heap for action input
    end

    # loop over items
    EG4item::Int = 0
    EG4dofitem::Array{Int,1} = [1,1] # EG id of the current item with respect to operator
    dofitems::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found
    itempos4dofitem::Array{Int,1} = [1,1] # local item position in dofitem
    orientation4dofitem::Array{Int,1} = [1,1] # local item position in dofitem
    coefficient4dofitem::Array{T,1} = [0,0] # coefficients for operator
    ndofs4dofitem::Int = 0 # number of dofs for item
    dofitem::Int = 0
    coeffs::Array{T,1} = zeros(T,maxdofs)
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations
    basisevaler4dofitem = basisevaler[1,1,1,1]
    basisvals::Array{T,3} = basisevaler4dofitem.cvals
    temp::T = 0 # some temporary variable
    localb::Array{T,1} = zeros(T,maxdofs)
    nFE::Int = length(FE)
    bdof::Int = 0
    fdof::Int = 0

    regions::Array{Int,1} = AP.regions
    allitems::Bool = (regions == [0])
    nregions::Int = length(regions)
    for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if allitems || xItemRegions[item] == regions[r]

        for FEid = 1 : nFE - 1
            # get dofitem informations
            EG4item = dii4op[FEid](dofitems, EG4dofitem, itempos4dofitem, coefficient4dofitem, orientation4dofitem, item)

            # get information on dofitems
            weights = qf[EG4item].w
            for di = 1 : length(dofitems)
                dofitem = dofitems[di]
                if dofitem != 0
                    # update FEbasisevaler on dofitem
                    basisevaler4dofitem = basisevaler[EG4dofitem[di],FEid,itempos4dofitem[di],orientation4dofitem[di]]
                    update!(basisevaler4dofitem, dofitem)

                    # update coeffs on dofitem
                    ndofs4dofitem = ndofs4EG[FEid][EG4dofitem[di]]
                    for j=1:ndofs4dofitem
                        fdof = xItemDofs[FEid][j,dofitem] + offsets2[FEid]
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
        EG4item = dii4op[nFE](dofitems, EG4dofitem, itempos4dofitem, coefficient4dofitem, orientation4dofitem, item)
        ndofs4dofitem = ndofs4EG[nFE][EG4item]
        
        for di = 1 : length(dofitems)
            dofitem = dofitems[di]
            if dofitem != 0
                basisevaler4dofitem = basisevaler[EG4dofitem[di],nFE,itempos4dofitem[di],orientation4dofitem[di]]
                basisvals = basisevaler4dofitem.cvals
                ndofs4dofitem = ndofs4EG[nFE][EG4dofitem[di]]
                update!(action, basisevaler4dofitem, dofitems[di], item, regions[r])

                for i in eachindex(weights)
        
                    # apply action
                    apply_action!(action_result, action_input[i], action, i)
        
                    # multiply third component
                    for dof_j = 1 : ndofs4dofitem
                        temp = 0
                        for k = 1 : action_resultdim
                            temp += action_result[k] * basisvals[k,dof_j,i]
                        end
                        localb[dof_j] += temp * weights[i] * coefficient4dofitem[di]
                    end 
                end  
        
                for dof_i = 1 : ndofs4dofitem
                    bdof = xItemDofs[nFE][dof_i,dofitem] + offset
                    b[bdof] += localb[dof_i] * xItemVolumes[item] * factor
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


