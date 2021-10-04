
"""
$(TYPEDEF)

linearform assembly pattern type
"""
abstract type APT_LinearForm <: AssemblyPatternType end

function Base.show(io::IO, ::Type{APT_LinearForm})
    print(io, "LinearForm")
end

"""
````
function LinearForm(
    T::Type{<:Real},
    AT::Type{<:AssemblyType},
    FE::Array{FESpace,1},
    operators::Array{DataType,1}, 
    action::AbstractAction; 
    regions::Array{Int,1} = [0])
````

Creates a LinearForm assembly pattern with the given FESpaces, operators and action etc.
"""
function LinearForm(T::Type{<:Real}, AT::Type{<:AssemblyType}, FES::Array{<:FESpace{Tv,Ti},1}, operators, action = NoAction(); regions = [0], name = "LF") where {Tv,Ti}
    @assert length(operators) == 1
    @assert length(FES) == 1
    return AssemblyPattern{APT_LinearForm, T, AT}(name,FES,operators,action,[1],regions)
end


"""
````
assemble!(
    b::Union{AbstractArray{T,1},AbstractArray{T,2}},    # target vector/matrix
    AP::AssemblyPattern{APT,T,AT};                      # LinearForm pattern
    factor = 1)                                         # factor that is multiplied
    where {APT <: APT_LinearForm, T, AT}
````

Assembly of a LinearForm pattern AP into a vector or matrix (if action is vetor-valued).
"""
function assemble!(
    b::Union{AbstractArray{T,1},AbstractArray{T,2}},
    AP::AssemblyPattern{APT,T,AT};
    skip_preps::Bool = false,
    factor = 1,
    offset = 0) where {APT <: APT_LinearForm, T <: Real, AT <: AssemblyType}

    # prepare assembly
    FE = AP.FES[1]
    if !skip_preps
        prepare_assembly!(AP)
    end
    AM::AssemblyManager{T} = AP.AM
    xItemVolumes::Array{T,1} = FE.xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemRegions::GridRegionTypes{Int32} = FE.xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = length(xItemVolumes)

    # prepare action
    action = AP.action
    if typeof(action) <: NoAction
        action_resultdim = size(get_basisevaler(AM, 1, 1).cvals,1)
        action_result = zeros(T,action_resultdim) # heap for action output
        action_input = action_result
    else
        action_input::Array{T,1} = zeros(T,action.argsizes[2]) # heap for action input
        action_resultdim::Int = action.argsizes[1]
        action_result = zeros(T,action_resultdim) # heap for action output
    end
    if typeof(b) <: AbstractArray{T,1}
        @assert action_resultdim == 1
        onedimensional = true
    else
        onedimensional = false
    end

    if AP.regions != [0]
        @logmsg MoreInfo "Assembling $(AP.name) ($AT in regions = $(AP.regions))"
    else
        @logmsg MoreInfo "Assembling $(AP.name) ($AT)"
    end
    @debug AP

    # loop over items
    basisevaler::FEBasisEvaluator = get_basisevaler(AM, 1, 1)
    basisxref::Array{Array{T,1},1} = basisevaler.xref
    weights::Array{T,1} = get_qweights(AM)
    localb::Array{T,2} = zeros(T,get_maxndofs(AM)[1],action_resultdim)
    ndofitems::Int = get_maxdofitems(AM)[1]
    bdof::Int = 0
    ndofs4dofitem::Int = 0
    itemfactor::T = 0
    regions::Array{Int,1} = AP.regions
    allitems::Bool = (regions == [0])
    nregions::Int = length(regions)
    for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if allitems || xItemRegions[item] == regions[r]

        # update assembly manager (also updates necessary basisevaler)
        update_assembly!(AM, item)
        weights = get_qweights(AM)

        # loop over associated dofitems
        for di = 1: ndofitems
            if AM.dofitems[1][di] != 0

                # get information on dofitem
                ndofs4dofitem = get_ndofs(AM, 1, di)

                # get correct basis evaluator for dofitem (was already updated by AM)
                basisevaler = get_basisevaler(AM, 1, di)

                # update action on dofitem
                update_action!(action, basisevaler, AM.dofitems[1][di], item, regions[r])

                for i in eachindex(weights)
                    for dof_i = 1 : ndofs4dofitem
                        # apply action
                        eval_febe!(action_input, basisevaler, dof_i, i)
                        apply_action!(action_result, action_input, action, i, basisevaler.xref[i])
                        for j = 1 : action_resultdim
                            localb[dof_i,j] += action_result[j] * weights[i]
                        end
                    end 
                end  

                ## copy into global vector
                itemfactor = factor * xItemVolumes[item] * AM.coeff4dofitem[1][di]
                if onedimensional
                    for dof_i = 1 : ndofs4dofitem
                        bdof = get_dof(AM, 1, di, dof_i) + offset
                        b[bdof] += localb[dof_i,1] * itemfactor
                    end
                else
                    for dof_i = 1 : ndofs4dofitem, j = 1 : action_resultdim
                        bdof = get_dof(AM, 1, di, dof_i) + offset
                        b[bdof,j] += localb[dof_i,j] * itemfactor
                    end
                end
                fill!(localb, 0)
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
    skip_preps::Bool = false,
    factor = 1) where {APT <: APT_LinearForm, T <: Real, AT <: AssemblyType}

    @assert b.FES == AP.FES[1]

    assemble!(b.entries, AP; factor = factor, offset = b.offset, skip_preps = skip_preps)
end