abstract type APT_LinearForm <: AssemblyPatternType end

"""
````
function LinearForm(
    T::Type{<:Real},
    AT::Type{<:AbstractAssemblyType},
    FE::Array{FESpace,1},
    operators::Array{DataType,1}, 
    action::AbstractAction; 
    regions::Array{Int,1} = [0])
````

Creates a LinearForm assembly pattern with the given FESpaces, operators and action etc.
"""
function LinearForm(T::Type{<:Real}, AT::Type{<:AbstractAssemblyType}, FES, operators, action; regions = [0])
    return AssemblyPattern{APT_LinearForm, T, AT}(FES,operators,action,regions,AssemblyPatternPreparations(nothing,nothing,nothing,nothing,nothing,nothing))
end


function assemble!(
    b::Union{AbstractArray{T,1},AbstractArray{T,2}},
    AP::AssemblyPattern{APT,T,AT};
    verbosity::Int = 0,
    skip_preps::Bool = false,
    factor = 1,
    offset = 0) where {APT <: APT_LinearForm, T <: Real, AT <: AbstractAssemblyType}

    # prepare assembly
    FE = AP.FES[1]
    action = AP.action
    if !skip_preps
        prepare_assembly!(AP; verbosity = verbosity - 1)
    end
    EG = AP.APP.EG
    ndofs4EG::Array{Array{Int,1},1} = AP.APP.ndofs4EG
    qf::Array{QuadratureRule,1} = AP.APP.qf
    basisevaler::Array{FEBasisEvaluator,4} = AP.APP.basisevaler
    dii4op::Array{Function,1} = AP.APP.dii4op

    xItemVolumes::Array{T,1} = FE.xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemDofs::Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}} = Dofmap4AssemblyType(FE, AP.APP.basisAT[1])
    xItemRegions::Union{VectorOfConstants{Int32}, Array{Int32,1}} = FE.xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = length(xItemVolumes)


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

    if verbosity > 0
        println("  Assembling ($APT,$AT,$T) into vector")
        println("   skip_preps = $skip_preps")
        println("    operators = $(AP.operators)")
        println("      regions = $(AP.regions)")
        println("       factor = $factor")
        println("       action = $(AP.action.name) (size = $(action.argsizes))")
        println("        qf[1] = $(qf[1].name) ")
        println("           EG = $EG")
    end

    # loop over items
    EG4dofitem::Array{Int,1}  = [1,1] # type of the current item
    ndofs4dofitem::Int = 0 # number of dofs for item
    dofitems::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found
    itempos4dofitem::Array{Int,1} = [1,1] # local item position in dofitem
    orientation4dofitem::Array{Int,1} = [1,1] # local orientation
    dofoffset4dofitem::Array{Int,1} = [0,0] # local dof offset (for broken dofmaps)
    coefficient4dofitem::Array{T,1} = [0,0]
    dofitem::Int = 0
    maxndofs::Int = max_num_targets_per_source(xItemDofs)
    action_input::Array{T,1} = zeros(T,cvals_resultdim) # heap for action input
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations
    basisevaler4dofitem::FEBasisEvaluator = basisevaler[1]
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
                        bdof = xItemDofs[dof_i + dofoffset4dofitem[di],dofitem] + offset
                        b[bdof] += localb[dof_i,1] * itemfactor
                    end
                else
                    for dof_i = 1 : ndofs4dofitem, j = 1 : action_resultdim
                        bdof = xItemDofs[dof_i + dofoffset4dofitem[di],dofitem] + offset
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



