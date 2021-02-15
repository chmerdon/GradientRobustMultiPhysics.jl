abstract type APT_LinearForm <: AssemblyPatternType end

"""
$(TYPEDEF)

creates a linearform assembly pattern
"""
function LinearForm(T::Type{<:Real}, AT::Type{<:AbstractAssemblyType}, FES, operators, action; regions = [0])
    return AssemblyPattern{APT_LinearForm, T, AT}(FES,operators,action,regions,AssemblyPatternPreparations(nothing,nothing,nothing,nothing,nothing))
end


"""
````
assemble!(
    b::Union{AbstractArray{T,1},AbstractArray{T,2}},
    AP::AssemblyPattern{APT,T,AT},
    verbosity::Int = 0,
    skip_preps::Bool = false,
    factor = 1,
    offset = 0) where {APT <: APT_LinearForm <: Real, AT <: AbstractAssemblyType}

````

Assembly of a LinearForm assembly pattern into a given one- or two-dimensional Array b.
"""
function assemble!(
    b::Union{AbstractArray{T,1},AbstractArray{T,2}},
    AP::AssemblyPattern{APT,T,AT};
    verbosity::Int = 0,
    skip_preps::Bool = false,
    factor = 1,
    offset = 0) where {APT <: APT_LinearForm, T <: Real, AT <: AbstractAssemblyType}

    # get adjacencies
    FE = AP.FES[1]
    xItemVolumes::Array{T,1} = FE.xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemDofs::Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}} = Dofmap4AssemblyType(FE, DofitemAT4Operator(AT, AP.operators[1]))
    xItemRegions::Union{VectorOfConstants{Int32}, Array{Int32,1}} = FE.xgrid[GridComponentRegions4AssemblyType(AT)]
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
    ncomponents::Int = get_ncomponents(eltype(FE))
    cvals_resultdim::Int = size(basisevaler[end,1,1,1].cvals,1)
    action_resultdim::Int = action.argsizes[1]

    if typeof(b) <: AbstractArray{T,1}
        @assert action_resultdim == 1
        onedimensional = true
    else
        onedimensional = false
    end

    # loop over items
    EG4dofitem::Array{Int,1}  = [1,1] # type of the current item
    ndofs4dofitem::Int = 0 # number of dofs for item
    dofitems::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found
    itempos4dofitem::Array{Int,1} = [1,1] # local item position in dofitem
    orientation4dofitem::Array{Int,1} = [1,2] # local orientation
    coefficient4dofitem::Array{T,1} = [0,0]
    dofitem::Int = 0
    maxndofs::Int = max_num_targets_per_source(xItemDofs)
    dofs = zeros(Int,maxndofs)
    action_input::Array{T,1} = zeros(T,cvals_resultdim) # heap for action input
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations
    basisevaler4dofitem = basisevaler[1]
    localb::Array{T,2} = zeros(T,maxndofs,action_resultdim)
    bdof::Int = 0
    itemfactor::T = 0

    regions::Array{Int,1} = AP.regions
    allitems::Bool = (regions == [0])
    nregions::Int = length(regions)
    for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if allitems || xItemRegions[item] == regions[r]

        # get dofitem informations
        dii4op[1](dofitems, EG4dofitem, itempos4dofitem, coefficient4dofitem, orientation4dofitem, item)
        itemfactor = factor * xItemVolumes[item]

        # loop over associated dofitems
        for di = 1 : length(dofitems)
            dofitem = dofitems[di]
            if dofitem != 0

                # get number of dofs on this dofitem
                ndofs4dofitem = ndofs4EG[1][EG4dofitem[di]]

                # update FEbasisevaler on dofitem
                basisevaler4dofitem = basisevaler[EG4dofitem[di],1,itempos4dofitem[di],orientation4dofitem[di]]
                update!(basisevaler4dofitem,dofitem)

                # update action on dofitem
                update!(action, basisevaler4dofitem, dofitem, item, regions[r])

                # update dofs
                for j=1:ndofs4dofitem
                    dofs[j] = xItemDofs[j,dofitem]
                end

                weights = qf[EG4dofitem[di]].w
                for i in eachindex(weights)
                    for dof_i = 1 : ndofs4dofitem
                        # apply action
                        eval!(action_input, basisevaler4dofitem , dof_i, i)
                        apply_action!(action_result, action_input, action, i)
                        for j = 1 : action_resultdim
                            localb[dof_i,j] += action_result[j] * weights[i] * coefficient4dofitem[di]
                        end
                    end 
                end  

                if onedimensional
                    for dof_i = 1 : ndofs4dofitem
                        bdof = dofs[dof_i] + offset
                        b[bdof] += localb[dof_i,1] * itemfactor
                    end
                else
                    for dof_i = 1 : ndofs4dofitem, j = 1 : action_resultdim
                        bdof = dofs[dof_i] + offset
                        b[bdof,j] += localb[dof_i,j] * itemfactor
                    end
                end
                fill!(localb, 0.0)
            end
        end
        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
    return nothing
end

function assemble!(
    b::FEVectorBlock{T},
    AP::AssemblyPattern{APT,T,AT};
    verbosity::Int = 0,
    skip_preps::Bool = false,
    factor = 1) where {APT <: APT_LinearForm, T <: Real, AT <: AbstractAssemblyType}

    @assert b.FES == AP.FES[1]

    assemble!(b.entries, AP; verbosity = verbosity, factor = factor, offset = b.offset, skip_preps = skip_preps)
end



