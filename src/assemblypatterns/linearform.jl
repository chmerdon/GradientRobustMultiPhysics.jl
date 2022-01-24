
"""
$(TYPEDEF)

linearform assembly pattern type
"""
abstract type APT_LinearForm <: AssemblyPatternType end

function Base.show(io::IO, ::Type{APT_LinearForm})
    print(io, "LinearForm")
end

"""
$(TYPEDSIGNATURES)

Creates a (discrete) LinearForm assembly pattern based on:

- operators : operators that should be evaluated for the coressponding FESpace (last one refers to test function)
- FES       : FESpaces for each operator (last one refers to test function)
- action    : an Action with kernel of interface (result, input, kwargs) that takes input (= all but last operator evaluations) and computes result to be dot-producted with test function evaluation
              (if no action is specified, the full input vector is dot-producted with the test function operator evaluation)

Optional arguments:
- regions   : specifies in which regions the operator should assemble, default [0] means all regions
- name      : name for this LinearForm that is used in print messages
- AT        : specifies on which entities of the grid the LinearForm is assembled (default: ON_CELLS)

"""
function DiscreteLinearForm(operators, FES::Array{<:FESpace{Tv,Ti},1}, action = NoAction(); T = Float64, AT = ON_CELLS, regions = [0], name = "LF") where {Tv,Ti}
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
    AP::AssemblyPattern{APT,T,AT},
    FEB = [];
    skip_preps::Bool = false,
    factor = 1,
    fixed_arguments = nothing, # ignored
    offset = 0) where {APT <: APT_LinearForm, T <: Real, Tv <: Real, Ti <: Int, AT <: AssemblyType}

    # prepare assembly
    FE = AP.FES
    nFE = length(FE)
    @assert length(FEB) == nFE-1
    if !skip_preps
        prepare_assembly!(AP)
    end
    AM::AssemblyManager{T} = AP.AM
    xItemVolumes::Array{T,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemRegions::GridRegionTypes{Int32} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = length(xItemVolumes)

    # prepare action
    action = AP.action
    if typeof(action) <: NoAction
        action_resultdim = size(get_basisevaler(AM, nFE, 1).cvals,1)
        action_result = ones(T,action_resultdim) # heap for action output
        action_input = action_result
    else
        action_resultdim::Int = action.argsizes[1]
        action_result = action.val
    end
    if nFE > 1
        maxnweights = get_maxnqweights(AM)
        action_input_FEB = Array{Array{T,1},1}(undef,maxnweights)
        if typeof(action) <: NoAction
            for j = 1 : maxnweights
                action_input_FEB[j] = zeros(T,action_resultdim) # heap for action input
            end
        else
            for j = 1 : maxnweights
                action_input_FEB[j] = zeros(T,action.argsizes[2]) # heap for action input
            end
        end
        action_input = action_input_FEB[1]
    else
        if typeof(action) <: NoAction
            action_input = zeros(T,action_resultdim)
        else
            action_input = zeros(T,action.argsizes[2])
        end
    end
    if typeof(b) <: AbstractArray{T,1}
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
    offsets::Array{Int,1} = zeros(Int,nFE+1)
    basisevaler = get_basisevaler(AM, 1, 1)
    for j = 1 : nFE
        basisevaler = get_basisevaler(AM, j, 1)
        offsets[j+1] = offsets[j] + get_basisdim(AM, j)
    end
    basisvals::Union{SharedCValView{T},Array{T,3}} = basisevaler.cvals
    weights::Array{T,1} = get_qweights(AM)
    localb::Array{T,1} = zeros(T,get_maxndofs(AM)[1])
    maxdofs::Array{Int,1} = get_maxndofs(AM)
    maxdofitems::Array{Int,1} = get_maxdofitems(AM)
    coeffs::Array{T,1} = zeros(T,sum(maxdofs[1:end]))
    ndofitems::Int = get_maxdofitems(AM)[1]
    bdof::Int = 0
    temp::T = 0 # some temporary variable
    ndofs4dofitem::Int = 0
    itemfactor::T = 0
    regions::Array{Int,1} = AP.regions
    allitems::Bool = (regions == [0])
    nregions::Int = length(regions)
    loop_allocations = @allocated for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if allitems || xItemRegions[item] == regions[r]

        # update assembly manager (also updates necessary basisevaler)
        update_assembly!(AM, item)
        weights = get_qweights(AM)

        # assemble all operators (of current solution in FEB) but the last into action_input_FEB
        if nFE > 1
            for i in eachindex(weights)
                fill!(action_input_FEB[i], 0)
            end
        end
        for FEid = 1 : nFE - 1
            for di = 1 : maxdofitems[FEid]
                if AM.dofitems[FEid][di] != 0
                    # get correct basis evaluator for dofitem (was already updated by AM)
                    basisevaler= get_basisevaler(AM, FEid, di)

                    # get coefficients of FE number FEid on current dofitem
                    get_coeffs!(coeffs, FEB[FEid], AM, FEid, di)
                    coeffs .*= AM.coeff4dofitem[FEid][di]

                    # write evaluation of operator of current FE into action_input_FEB
                    for i in eachindex(weights)
                        eval_febe!(action_input_FEB[i], basisevaler, coeffs, i, offsets[FEid])
                    end  
                end
            end
        end


        # loop over associated dofitems of testfunction operator evaluation
        for di = 1: ndofitems
            if AM.dofitems[nFE][di] != 0

                # get information on dofitem
                ndofs4dofitem = get_ndofs(AM, nFE, di)
                basisevaler = get_basisevaler(AM, nFE, di)
                basisvals = basisevaler.cvals

                # update action on dofitem
                if is_itemdependent(action)
                    action.item[1] = item
                    action.item[2] = AM.dofitems[nFE][di]
                    action.item[3] = xItemRegions[item]
                    action.item[4] = di
                end

                for i in eachindex(weights)
                    if nFE > 1
                        action_input = action_input_FEB[i]
                    end
                    if is_xdependent(action)
                        update_trafo!(basisevaler.L2G, AM.dofitems[nFE][di])
                        eval_trafo!(action.x, basisevaler.L2G, basisevaler.xref[i])
                    end
                    # apply action
                    eval_action!(action, action_input)

                    # multiply with test function
                    for dof_i = 1 : ndofs4dofitem
                        temp = 0
                        for k = 1 : action_resultdim
                            temp += action_result[k] * basisvals[k,dof_i,i]
                        end
                        localb[dof_i] += temp * weights[i]
                    end 
                end 

                ## copy into global vector
                itemfactor = factor * xItemVolumes[item] * AM.coeff4dofitem[nFE][di]
                if onedimensional
                    for dof_i = 1 : ndofs4dofitem
                        bdof = get_dof(AM, nFE, di, dof_i) + offset
                        b[bdof] += localb[dof_i,1] * itemfactor
                    end
                else
                    for dof_i = 1 : ndofs4dofitem, j = 1 : action_resultdim
                        bdof = get_dof(AM, nFE, di, dof_i) + offset
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
    AP.last_allocations = loop_allocations

    return nothing
end

function assemble!(
    b::FEVectorBlock{T},
    AP::AssemblyPattern{APT,T,AT},
    FEB::Array{<:FEVectorBlock{T,Tv,Ti},1};
    skip_preps::Bool = false,
    factor = 1,
    fixed_arguments = nothing # ignored
    ) where {APT <: APT_LinearForm, Tv <: Real, T <: Real, Ti <: Int, AT <: AssemblyType}

    @assert b.FES == AP.FES[1]

    assemble!(b.entries, AP, FEB; factor = factor, offset = b.offset, skip_preps = skip_preps)
end