
"""
$(TYPEDEF)

bilinearform assembly pattern type
"""
abstract type APT_BilinearForm <: AssemblyPatternType end

"""
$(TYPEDEF)

symmetric bilinearform assembly pattern type
"""
abstract type APT_SymmetricBilinearForm <: APT_BilinearForm end

"""
$(TYPEDEF)

lumped bilinearform assembly pattern type where only the diagonal elements on each item are assembled
"""
abstract type APT_LumpedBilinearForm <: APT_BilinearForm end

function Base.show(io::IO, ::Type{APT_BilinearForm})
    print(io, "BilinearForm")
end
function Base.show(io::IO, ::Type{APT_SymmetricBilinearForm})
    print(io, "SymmetricBilinearForm")
end
function Base.show(io::IO, ::Type{APT_LumpedBilinearForm})
    print(io, "LumpedBilinearForm")
end

"""
$(TYPEDSIGNATURES)

Creates a (discrete) SymmetricBilinearForm assembly pattern. For more details see BilinearForm constructor.
"""
function DiscreteSymmetricBilinearForm(operators, FES, action = NoAction(); T = Float64, AT = ON_CELLS, name = "symBLF", regions = [0], apply_action_to = [1])
    @assert length(operators) == length(FES) == 2 "each FESpace needs an operator and vice versa"
    @assert apply_action_to in [[1],[2]] "action can only be applied to one argument [1] or [2]"
    return AssemblyPattern{APT_SymmetricBilinearForm, T, AT}(name, FES,operators,action,apply_action_to,regions)
end

"""
$(TYPEDSIGNATURES)

Creates a (discrete) BilinearForm assembly pattern based on:

- operators : operators that should be evaluated for the coressponding FESpace (last two refer to ansatz and test function)
- FES       : FESpaces for each operator (last two refer to ansatz and test function)
- action    : an Action with kernel of interface (result, input, kwargs) that takes input (= all but last operator evaluations) and computes result to be dot-producted with test function evaluation
              (if no action is specified, the full input vector is dot-producted with the test function operator evaluation)

Optional arguments:
- apply_action_to : specifies which of the two linear arguments is part of the action input ([1] = ansatz, [2] = test)
- regions   : specifies in which regions the operator should assemble, default [0] means all regions
- name      : name for this LinearForm that is used in print messages
- AT        : specifies on which entities of the grid the LinearForm is assembled (default: ON_CELLS)
"""
function DiscreteBilinearForm(operators, FES, action = NoAction(); T = Float64, AT = ON_CELLS, name = "BLF", regions = [0], apply_action_to = [1])
    @assert length(operators) == length(FES) == 2 "each FESpace needs an operator and vice versa"
    @assert apply_action_to in [[1],[2]] "action can only be applied to one argument [1] or [2]"
    return AssemblyPattern{APT_BilinearForm, T, AT}(name,FES,operators,action,apply_action_to,regions)
end

"""
$(TYPEDSIGNATURES)

Creates a (discrete) LumpedBilinearForm assembly pattern. For more details see BilinearForm constructor.
"""
function DiscreteLumpedBilinearForm(operators, FES, action = NoAction(); T = Float64, AT = ON_CELLS, name = "lumpedBLF", regions = [0], apply_action_to = [1])
    @assert length(operators) == length(FES) == 2 "each FESpace needs an operator and vice versa"
    @assert apply_action_to in [[1],[2]] "action can only be applied to one argument [1] or [2]"
    return AssemblyPattern{APT_LumpedBilinearForm, T, AT}(name, FES,operators,action,apply_action_to,regions)
end


"""
````
assemble!(
    A::AbstractArray{T,2},                  # target matrix
    AP::AssemblyPattern{APT,T,AT};          # BilinearForm Pattern
    apply_action_to::Int = 1,               # action is applied to which argument?
    factor = 1,                             # factor that is multiplied
    transposed_assembly::Bool = false,      # transpose result?
    transpose_copy = Nothing)               # copy a transposed block to this matrix
    where {APT <: APT_BilinearForm, T, AT}
````

Assembly of a BilinearForm BLF into given two-dimensional AbstractArray (e.g. FEMatrixBlock or a ExtendableSparseMatrix).
"""
function assemble!(
    A::AbstractArray{T,2},
    AP::AssemblyPattern{APT,T,AT,Tv,Ti},
    FEB = [];
    factor = 1,
    transposed_assembly::Bool = false,
    transpose_copy = nothing,
    skip_preps::Bool = false,
    fixed_arguments = nothing, # ignored
    offsetX = 0,
    offsetY = 0) where {APT <: APT_BilinearForm, T <: Real, AT <: AssemblyType,Tv,Ti}

    # prepare assembly
    FE = AP.FES
    nFE = length(FE)
    if !skip_preps
        prepare_assembly!(AP)
    end
    AM::AssemblyManager{T} = AP.AM
    xItemVolumes::Array{T,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemRegions::GridRegionTypes{Ti} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = length(xItemVolumes)

    # prepare action
    action = AP.action
    apply_action_to = AP.apply_action_to[1]
    if typeof(action) <: NoAction
        action_resultdim = size(get_basisevaler(AM, apply_action_to, 1).cvals,1)
        action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output
    else
        maxnweights = get_maxnqweights(AM)
        action_resultdim::Int = action.argsizes[1]
        action_input::Array{T,1} = zeros(T,action.argsizes[2]) # heap for action input
        action_result = action.val
        if nFE > 2
            action_input_FEB = Array{Array{T,1},1}(undef,maxnweights)
            for j = 1 : maxnweights
                action_input_FEB[j] = zeros(T,action.argsizes[2]) # heap for action input
            end
            action_input = action_input_FEB[1]
        else
            action_input = zeros(T,action.argsizes[2])
        end
    end

    if AP.regions != [0]
        @logmsg MoreInfo "Assembling $(AP.name) ($AT in regions = $(AP.regions)) into matrix"
    else
        @logmsg MoreInfo "Assembling $(AP.name) ($AT) into matrix"
    end
    @debug AP
    @debug "offsets = [$offsetX,$offsetY], factor = $factor"
 
    # loop over items
    offsets::Array{Int,1} = zeros(Int,nFE+1)
    for j = 1 : nFE-2
        offsets[j+1] = offsets[j] + get_basisdim(AM, j)
    end
    if apply_action_to == 1
        offsets[nFE] = offsets[nFE-1] + get_basisdim(AM, nFE-1)
        offsets[nFE+1] = offsets[nFE] + get_basisdim(AM, nFE)
    else
        offsets[nFE] = offsets[nFE-1] + get_basisdim(AM, nFE) 
        offsets[nFE+1] = offsets[nFE] + get_basisdim(AM, nFE-1)
    end

    weights::Array{T,1} = get_qweights(AM) # somehow this saves A LOT allocations
    basisevaler::Array{FEBasisEvaluator{T,Tv,Ti},1} = [get_basisevaler(AM, 1, 1), get_basisevaler(AM, 2, 1)]
    basisvals::Union{SharedCValView{T},Array{T,3}} = basisevaler[1].cvals
    basisxref::Array{Array{T,1},1} = basisevaler[1].xref
    localmatrix::Array{T,2} = zeros(T,get_maxndofs(AM,1),get_maxndofs(AM,2))
    temp::T = 0 # some temporary variable
    ndofs4dofitem::Array{Int,1} = [0,0]
    acol::Int = 0
    arow::Int = 0
    maxdofs::Array{Int,1} = get_maxndofs(AM)
    coeffs::Array{T,1} = zeros(T,sum(maxdofs[1:end]))
    dofitems::Array{Int,1} = [0,0]
    is_symmetric::Bool = APT <: APT_SymmetricBilinearForm
    maxdofitems::Array{Int,1} = get_maxdofitems(AM)
    indexmap = CartesianIndices(zeros(Int, maxdofitems[1],maxdofitems[2]))
    other_id::Int = apply_action_to == 2 ? 1 : 2
    other_FEid::Int = nFE - 2 + other_id
    action_linFEid::Int = nFE - 2 + apply_action_to

    function basismul!(localmatrix,dof_i,i,::Type{<:APT_BilinearForm},is_locally_symmetric)
        _temp::T = temp # re-annotate to avoid allocations in closure
        for dof_j = 1 : ndofs4dofitem[other_id]
            _temp = 0
            for k = 1 : action_resultdim
                _temp += action_result[k] * basisvals[k,dof_j,i]
            end
            if apply_action_to == 2
                localmatrix[dof_j,dof_i] += weights[i] * _temp
            else
                localmatrix[dof_i,dof_j] += weights[i] * _temp
            end
        end
        return nothing
    end

    function basismul!(localmatrix,dof_i,i,::Type{APT_SymmetricBilinearForm},is_locally_symmetric)
        if is_locally_symmetric
            _temp::T = temp # re-annotate to avoid allocations in closure
            for dof_j = dof_i : ndofs4dofitem[other_id]
                _temp = 0
                for k = 1 : action_resultdim
                    _temp += action_result[k] * basisvals[k,dof_j,i]
                end
                localmatrix[dof_i,dof_j] += weights[i] * _temp
            end
        else
            basismul!(localmatrix,dof_i,i,APT_BilinearForm,is_locally_symmetric)
        end
        return nothing
    end

    function basismul!(localmatrix,dof_i,i,::Type{APT_LumpedBilinearForm},is_locally_symmetric)
        _temp::T = temp # re-annotate to avoid allocations in closure
        _temp = 0
        for k = 1 : action_resultdim
            _temp += action_result[k] * basisvals[k,dof_i,i]
        end
        localmatrix[dof_i,dof_i] += weights[i] * _temp
        return nothing
    end

    itemfactor::T = 0
    regions::Array{Ti,1} = AP.regions
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
        if nFE > 2
            for i in eachindex(weights)
                fill!(action_input_FEB[i], 0)
            end
        end
        for FEid = 1 : nFE - 2
            for di = 1 : maxdofitems[FEid]
                if AM.dofitems[FEid][di] != 0
                    # get correct basis evaluator for dofitem (was already updated by AM)
                    basisevaler[1] = get_basisevaler(AM, FEid, di)

                    # get coefficients of FE number FEid on current dofitem
                    get_coeffs!(coeffs, FEB[FEid], AM, FEid, di)
                    coeffs .*= AM.coeff4dofitem[FEid][di]

                    # write evaluation of operator of current FE into action_input_FEB
                    for i in eachindex(weights)
                        eval_febe!(action_input_FEB[i], basisevaler[1], coeffs, i, offsets[FEid])
                    end  
                end
            end
        end

        # loop over associated dofitems
        # di, dj == 2 is only performed if one of the operators jumps
        for di in indexmap
            dofitems[1] = AM.dofitems[nFE-1][di[1]]
            dofitems[2] = AM.dofitems[nFE][di[2]]
            if dofitems[1] > 0 && dofitems[2] > 0

                # even if global matrix is symmeric, local matrix might be not in case of JumpOperators
                is_locally_symmetric = is_symmetric * (dofitems[1] == dofitems[2])

                # get number of dofs on this dofitem
                ndofs4dofitem[1] = get_ndofs(AM, nFE-1, di[1])
                ndofs4dofitem[2] = get_ndofs(AM, nFE, di[2])

                # update FEbasisevaler
                basisevaler[1] = get_basisevaler(AM, nFE-1, di[1])::FEBasisEvaluator
                basisevaler[2] = get_basisevaler(AM, nFE, di[2])::FEBasisEvaluator
                basisvals = basisevaler[other_id].cvals
                basisxref = basisevaler[other_id].xref

                # update action on dofitem
                if apply_action_to > 0
                    if is_itemdependent(action)
                        action.item[1] = item
                        action.item[2] = dofitems[apply_action_to]
                        action.item[3] = xItemRegions[item]
                    end
                    ndofs4dofitem_action = ndofs4dofitem[apply_action_to]
                else
                    ndofs4dofitem_action = ndofs4dofitem[1]
                end

                for i in eachindex(weights)
                    if nFE > 2
                        action_input = action_input_FEB[i]
                    end
                    if is_xdependent(action)
                        update_trafo!(basisevaler[apply_action_to].L2G, dofitems[apply_action_to])
                        eval_trafo!(action.x, basisevaler[apply_action_to].L2G, basisevaler[apply_action_to].xref[i])
                    end
                    if is_xrefdependent(action)
                        action.xref = basisevaler[apply_action_to].xref[i]
                    end
                    for dof_i = 1 : ndofs4dofitem_action
                        if typeof(action) <: NoAction
                            eval_febe!(action_result, basisevaler[1], dof_i, i) # in this case no fixed components are used
                            action_result .*= AM.coeff4dofitem[action_linFEid][di[apply_action_to]]
                        else
                            eval_febe!(action_input, basisevaler[apply_action_to], dof_i, i, offsets[nFE-1])
                            view(action_input,offsets[nFE-1]+1:offsets[nFE]) .*= AM.coeff4dofitem[action_linFEid][di[apply_action_to]]
                            eval_action!(action, action_input)
                            action_result = action.val
                        end
                        basismul!(localmatrix,dof_i,i,APT,is_locally_symmetric)
                    end 
                end

                # copy localmatrix into global matrix
                itemfactor = xItemVolumes[item] * factor * AM.coeff4dofitem[other_FEid][di[other_id]]


                if APT <: Type{APT_LumpedBilinearForm}    
                    for dof_i = 1 : ndofs4dofitem[1]
                        arow = get_dof(AM, nFE, di[1], dof_i) + offsetX
                        acol = get_dof(AM, nFE-1, di[2], dof_i) + offsetY
                        _addnz(A,arow,acol,localmatrix[dof_i,dof_i] * itemfactor,1)
                    end    
                elseif is_locally_symmetric
                    for dof_i = 1 : ndofs4dofitem[1]
                        for dof_j = dof_i+1 : ndofs4dofitem[2]
                        #  if localmatrix[dof_i,dof_j] != 0 
                                arow = get_dof(AM, nFE-1, di[1], dof_i) + offsetX
                                acol = get_dof(AM, nFE, di[2], dof_j) + offsetY
                                _addnz(A,arow,acol,localmatrix[dof_i,dof_j] * itemfactor,1)
                                arow = get_dof(AM, nFE-1, di[1], dof_j) + offsetX
                                acol = get_dof(AM, nFE, di[2], dof_i) + offsetY
                                _addnz(A,arow,acol,localmatrix[dof_i,dof_j] * itemfactor,1)
                        # end
                        end
                    end    
                    for dof_i = 1 : ndofs4dofitem[1]
                        arow = get_dof(AM, nFE, di[1], dof_i) + offsetX
                        acol = get_dof(AM, nFE-1, di[2], dof_i) + offsetY
                        _addnz(A,arow,acol,localmatrix[dof_i,dof_i] * itemfactor,1)
                    end    
                else
                    for dof_i = 1 : ndofs4dofitem[1]
                        arow = get_dof(AM, nFE-1, di[1], dof_i) + offsetX
                        for dof_j = 1 : ndofs4dofitem[2]
                        #  if localmatrix[dof_i,dof_j] != 0
                                acol = get_dof(AM, nFE, di[2], dof_j) + offsetY
                                if transposed_assembly == true
                                _addnz(A,acol,arow,localmatrix[dof_i,dof_j] * itemfactor,1)
                                else 
                                _addnz(A,arow,acol,localmatrix[dof_i,dof_j] * itemfactor,1)  
                                end
                                if transpose_copy !== nothing # sign is changed in case nonzero rhs data is applied to LagrangeMultiplier (good idea?)
                                    if transposed_assembly == true
                                    _addnz(transpose_copy,arow,acol,localmatrix[dof_i,dof_j] * itemfactor,-1)
                                    else
                                    _addnz(transpose_copy,acol,arow,localmatrix[dof_i,dof_j] * itemfactor,-1)
                                    end
                                end
                        #  end
                        end
                    end
                end
                fill!(localmatrix, 0)

                #add_localmatrix!(A,localmatrix,di,APT,transposed_assembly,is_locally_symmetric, itemfactor)
            end
        end 
        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
    AP.last_allocations = loop_allocations
    return nothing
end


## wrapper for FEMatrixBlock to avoid use of setindex! functions of FEMAtrixBlock
function assemble!(
    A::FEMatrixBlock{TvM,TiM,TvG,TiG},
    AP::AssemblyPattern{APT,T,AT},
    FEB::Array{<:FEVectorBlock{T,TvG,TiG},1};
    factor = 1,
    skip_preps::Bool = false,
    fixed_arguments = nothing, # ignored
    transposed_assembly::Bool = false,
    transpose_copy = nothing) where {APT <: APT_BilinearForm, T <: Real, AT <: AssemblyType, TvM, TiM, TvG, TiG}

    if typeof(transpose_copy) <: FEMatrixBlock
        assemble!(A.entries, AP, FEB; factor = factor, transposed_assembly = transposed_assembly, transpose_copy = transpose_copy.entries, offsetX = A.offsetX, offsetY = A.offsetY, skip_preps = skip_preps)
    else
        assemble!(A.entries, AP, FEB; factor = factor, transposed_assembly = transposed_assembly, transpose_copy = transpose_copy, offsetX = A.offsetX, offsetY = A.offsetY, skip_preps = skip_preps)
    end
end



### OLD STUFF BELOW

# """
# ````
# assemble!(
#     b::AbstractArray{T,1},          # target vector
#     AP::AssemblyPattern{APT,T,AT},  # BilinearForm Pattern
#     fixedFE::AbstractArray;         # coefficients for fixed argument
#     apply_action_to::Int = 1,       # action is applied to 1st or 2nd argument?
#     fixed_arguments = [1],        # which argument is fixed?
#     factor = 1)                     # factor that is multiplied
#     where {APT <: APT_BilinearForm, T, AT}
# ````

# Assembly of a BilinearForm AP into given one-dimensional AbstractArray (e.g. a FEVectorBlock).
# Here, the second argument is fixed (default) by the given coefficients in fixedFE.
# With apply_action_to=2 the action can be also applied to the second argument instead of the first one (default).
# """
# function assemble!(
#     b::AbstractArray{T,1},
#     AP::AssemblyPattern{APT,T,AT},
#     fixedFE::AbstractArray{T,1};    # coefficients for fixed argument;
#     fixed_arguments = [1],
#     factor = 1,
#     skip_preps::Bool = false,
#     offsets::Array{Int,1} = [0,0]) where {APT <: APT_BilinearForm, T <: Real, AT <: AssemblyType}
    
#     # prepare assembly
#     FE = AP.FES
#     if !skip_preps
#         prepare_assembly!(AP)
#     end
#     AM::AssemblyManager{T} = AP.AM
#     xItemVolumes::Array{T,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
#     xItemRegions::GridRegionTypes{Int32} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
#     nitems = length(xItemVolumes)

#     # prepare action
#     action = AP.action
#     apply_action_to::Int = AP.apply_action_to[1]
#     if typeof(action) <: NoAction
#         apply_action_to = 0
#         action_resultdim = size(get_basisevaler(AM, 1, 1).cvals,1)
#     else
#         action_resultdim::Int = action.argsizes[1]
#         action_input::Array{T,1} = zeros(T,action.argsizes[2]) # heap for action input
#         action_result::Array{T,1} = action.val
#     end

#     @assert length(fixed_arguments) == 1 "Vector assembly of bilinear form requires exactly one fixed component"
#     fixed_argument = fixed_arguments[1]
#     if AP.regions != [0]
#         @logmsg MoreInfo "Assembling $(AP.name) with fixed argument $fixed_argument ($AT in regions = $(AP.regions))"
#     else
#         @logmsg MoreInfo "Assembling $(AP.name) with fixed argument $fixed_argument ($AT)"
#     end
#     @debug AP

#     # loop over items
#     weights::Array{T,1} = get_qweights(AM) # somehow this saves A LOT allocations
#     fixed_coeffs::Array{T,1} = zeros(T,get_maxndofs(AM,fixed_argument))
#     basisevaler::Array{FEBasisEvaluator,1} = [get_basisevaler(AM, 1, 1), get_basisevaler(AM, 2, 1)]
#     basisvals::Union{SharedCValView{T},Array{T,3}} = basisevaler[1].cvals
#     basisxref::Array{Array{T,1},1} = basisevaler[1].xref
#     temp::T = 0 # some temporary variable
#     ndofs4dofitem::Array{Int,1} = [0,0]
#     bdof::Int = 0
#     dofitems::Array{Int,1} = [0,0]
#     itemfactor::T = 0
#     maxdofitems::Array{Int,1} = get_maxdofitems(AM)
#     indexmap = CartesianIndices(zeros(Int, maxdofitems[1],maxdofitems[2]))
#     if apply_action_to == fixed_argument
#         fixedval = zeros(T, action.argsizes[2]) # some temporary variable
#     else
#         fixedval = zeros(T, action_resultdim) # some temporary variable
#     end
#     free_argument::Int = fixed_argument == 1 ? 2 : 1
#     localb::Array{T,2} = zeros(T,get_maxndofs(AM)[free_argument],action_resultdim)

#     regions::Array{Int,1} = AP.regions
#     allitems::Bool = (regions == [0])
#     nregions::Int = length(regions)
#     loop_allocations = @allocated for item = 1 : nitems
#     for r = 1 : nregions
#     # check if item region is in regions
#     if allitems || xItemRegions[item] == regions[r]

#         # update assembly manager (also updates necessary basisevaler)
#         update_assembly!(AM, item)
#         weights = get_qweights(AM)

#         # loop over associated dofitems
#         # di, dj == 2 is only performed if one of the operators jumps
#         for di in indexmap
#             dofitems[1] = AM.dofitems[1][di[1]]
#             dofitems[2] = AM.dofitems[2][di[2]]
#             if dofitems[1] > 0 && dofitems[2] > 0

#                 # get number of dofs on this dofitem
#                 ndofs4dofitem[1] = get_ndofs(AM, 1, di[1])
#                 ndofs4dofitem[2] = get_ndofs(AM, 2, di[2])

#                 # update FEbasisevaler
#                 basisevaler[1] = get_basisevaler(AM, 1, di[1])::FEBasisEvaluator
#                 basisevaler[2] = get_basisevaler(AM, 2, di[2])::FEBasisEvaluator
#                 basisvals = basisevaler[free_argument].cvals
#                 basisxref = basisevaler[free_argument].xref

#                 # update action on dofitem
#                 if apply_action_to > 0
#                     if is_itemdependent(action)
#                         action.item[1] = item
#                         action.item[2] = dofitems[apply_action_to]
#                         action.item[3] = xItemRegions[item]
#                     end
#                 end

#                 # update dofs
#                 get_coeffs!(fixed_coeffs, fixedFE, AM, fixed_argument, di[fixed_argument], offsets[2])

#                 for i in eachindex(weights)
#                     # evaluate fixed argument
#                     fill!(fixedval, 0.0)
#                     eval_febe!(fixedval, basisevaler[fixed_argument], fixed_coeffs, i)
#                     fixedval .*= AM.coeff4dofitem[fixed_argument][di[fixed_argument]]
#                     if is_xdependent(action)
#                         update_trafo!(basisevaler[apply_action_to].L2G, dofitems[apply_action_to])
#                         eval_trafo!(action.x, basisevaler[apply_action_to].L2G, basisevaler[apply_action_to].xref[i])
#                     end
#                     if is_xrefdependent(action)
#                         action.xref = basisevaler[apply_action_to].xref[i]
#                     end

#                     if apply_action_to == 0
#                         # multiply free argument
#                         fixedval .*= AM.coeff4dofitem[free_argument][di[free_argument]]
#                         for dof_i = 1 : ndofs4dofitem[free_argument]
#                             temp = 0
#                             for k = 1 : action_resultdim
#                                 temp += fixedval[k] * basisvals[k,dof_i,i]
#                             end
#                             localb[dof_i] += temp * weights[i]
#                         end 
#                     elseif apply_action_to == fixed_argument
#                         # apply action to fixed argument
#                         eval_action!(action, fixedval)

#                         # multiply free argument
#                         action_result .*= AM.coeff4dofitem[free_argument][di[free_argument]]
#                         for dof_i = 1 : ndofs4dofitem[free_argument]
#                             temp = 0
#                             for k = 1 : action_resultdim
#                                 temp += action_result[k] * basisvals[k,dof_i,i]
#                             end
#                             localb[dof_i] += temp * weights[i]
#                         end 
#                     else
#                         for dof_i = 1 : ndofs4dofitem[free_argument]
#                             # apply action to free argument
#                             eval_febe!(action_input, basisevaler[free_argument], dof_i, i)
#                             action_input .*= AM.coeff4dofitem[free_argument][di[free_argument]]
#                             eval_action!(action, action_input)
            
#                             # multiply fixed argument
#                             temp = 0
#                             for k = 1 : action_resultdim
#                                 temp += action_result[k] * fixedval[k]
#                             end
#                             localb[dof_i] += temp * weights[i]
#                         end 
#                     end
#                 end

#                 ## copy to global vector
#                 itemfactor = xItemVolumes[item] * factor
#                 for dof_i = 1 : ndofs4dofitem[free_argument]
#                     bdof = get_dof(AM, free_argument, di[free_argument], dof_i) + offsets[1]
#                     b[bdof] += localb[dof_i] * itemfactor
#                 end
#                 fill!(localb, 0.0)
#             end
#         end
#         break; # region for loop
#     end # if in region    
#     end # region for loop
#     end # item for loop
#     AP.last_allocations = loop_allocations
#     return nothing
# end


# # wrapper for FEVectorBlock to avoid setindex! functions of FEVectorBlock
# function assemble!(
#     b::FEVectorBlock,
#     AP::AssemblyPattern{APT,T,AT},
#     FEB::Array{<:FEVectorBlock,1}; # coefficient for fixed argument
#     fixed_arguments = [1],
#     skip_preps::Bool = false,
#     factor = 1) where {APT <: APT_BilinearForm, T <: Real, AT <: AssemblyType}

#    # @assert fixedFE.FES == AP.FES[fixed_argument]

#     assemble!(b.entries, AP, FEB[1].entries; factor = factor, fixed_arguments = fixed_arguments, offsets = [b.offset, FEB[1].offset], skip_preps = skip_preps)
# end