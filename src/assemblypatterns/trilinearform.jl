abstract type APT_TrilinearForm <: AssemblyPatternType end

function Base.show(io::IO, ::Type{APT_TrilinearForm})
    print(io, "TrilinearForm")
end

"""
````
function TrilinearForm(
    T::Type{<:Real},
    FES::Array{FESpace,1},          
    operators::Array{DataType,1},
    action::AbstractAction;
    regions::Array{Int,1} = [0])
````

Creates a TrilinearForm assembly pattern with the given FESpaces, operators and action etc.
"""
function TrilinearForm(
    T::Type{<:Real},
    AT::Type{<:AbstractAssemblyType},
    FES::Array{FESpace,1},
    operators::Array{DataType,1},
    action::AbstractAction;
    name = "TLF",
    regions::Array{Int,1} = [0])
    return  AssemblyPattern{APT_TrilinearForm, T, AT}(name,FES,operators,action,regions)
end


"""
````
assemble!(
    assemble!(
    A::AbstractArray{T,2},
    FE1::FEVectorBlock,
    AP::TrilinearForm{T, AT};
    fixed_argument::Int = 1,
    transposed_assembly::Bool = false,
    factor = 1)
````

Assembly of a TrilinearForm AP into given two-dimensional AbstractArray (e.g. a FEMatrixBlock).
Here, one argument (specified by fixed_argument) is fixed by the given coefficients in FE1. Note, that the action is
(currently) always applied to the first and second argument.
"""
function assemble!(
    A::AbstractArray{T,2},
    FE1::FEVectorBlock,
    AP::AssemblyPattern{APT,T,AT};
    fixed_argument::Int = 1,
    transposed_assembly::Bool = false,
    skip_preps::Bool = false,
    factor = 1) where {APT <: APT_TrilinearForm, T <: Real, AT <: AbstractAssemblyType}

    # prepare assembly
    FE = AP.FES
    if !skip_preps
        prepare_assembly!(AP)
    end
    AM::AssemblyManager{T} = AP.AM
    xItemVolumes::Array{T,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemRegions::Union{VectorOfConstants{Int32}, Array{Int32,1}} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = length(xItemVolumes)

    # prepare action
    action = AP.action
    action_resultdim::Int = action.argsizes[1]
    action_input::Array{T,1} = zeros(T,action.argsizes[2]) # heap for action input
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output

    if AP.regions != [0]
        @logmsg MoreInfo "Assembling $(AP.name) for given $(FE1.name) into vector ($AT in regions = $(AP.regions))"
    else
        @logmsg MoreInfo "Assembling $(AP.name) for given $(FE1.name) into vector ($AT)"
    end
    @debug AP

    # loop over items
    maxdofitems::Array{Int,1} = get_maxdofitems(AM)
    indexmap = CartesianIndices(zeros(Int, (maxdofitems[1],maxdofitems[2],maxdofitems[3])))
    basisevaler::Array{FEBasisEvaluator,1} = [get_basisevaler(AM, 1, 1),get_basisevaler(AM, 2, 1),get_basisevaler(AM, 3, 1)]
    basisvals_testfunction::Array{T,3} = basisevaler[3].cvals
    nonfixed_ids::Array{Int,1} = setdiff([1,2,3], fixed_argument)
    coeffs::Array{T,1} = zeros(T,get_maxndofs(AM,fixed_argument))
    dofs2::Array{Int,1} = zeros(Int,get_maxndofs(AM,nonfixed_ids[1]))
    dofs3::Array{Int,1} = zeros(Int,get_maxndofs(AM,nonfixed_ids[2]))
    localmatrix::Array{T,2} = zeros(T,length(dofs2),length(dofs3))
    evalfixedFE::Array{T,1} = zeros(T,size(basisevaler[fixed_argument].cvals,1)) # evaluation of argument 1
    ndofs4item::Array{Int, 1} = [0,0,0]
    dofitem::Array{Int,1} = [0,0,0]
    offsets::Array{Int,1} = [0, size(basisevaler[1].cvals,1), size(basisevaler[1].cvals,1) + size(basisevaler[2].cvals,1)]
    weights::Array{T,1} = get_qweights(AM) # somehow this saves A LOT allocations
    temp::T = 0 # some temporary variable

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

        # loop over associated dofitems
        # di, dj == 2 is only performed if one of the operators jumps
        for di in indexmap
            dofitem[1] = AM.dofitems[1][di[1]]
            dofitem[2] = AM.dofitems[2][di[2]]
            dofitem[3] = AM.dofitems[3][di[3]]

            if dofitem[1] > 0 && dofitem[2] > 0 && dofitem[3] > 0
                # get number of dofs on this dofitem
                ndofs4item[1] = get_ndofs(AM, 1, di[1])
                ndofs4item[2] = get_ndofs(AM, 2, di[2])
                ndofs4item[3] = get_ndofs(AM, 3, di[3])

                # update FEbasisevaler
                basisevaler[1] = get_basisevaler(AM, 1, di[1])
                basisevaler[2] = get_basisevaler(AM, 2, di[2])
                basisevaler[3] = get_basisevaler(AM, 3, di[3])

                # update action on dofitem
                update!(action, basisevaler[2], dofitem[2], item, regions[r])

                # update dofs of free arguments
                get_coeffs!(coeffs, FE1, AM, fixed_argument, di[fixed_argument])
                get_dofs!(dofs2, AM, nonfixed_ids[1], di[nonfixed_ids[1]])
                get_dofs!(dofs3, AM, nonfixed_ids[2], di[nonfixed_ids[2]])

                if fixed_argument in [1,2]
                    basisvals_testfunction = basisevaler[nonfixed_ids[2]].cvals
                    for i in eachindex(weights)
    
                        # evaluate fixed argument into action
                        fill!(action_input, 0)
                        eval!(action_input, basisevaler[fixed_argument], coeffs, i, offsets[fixed_argument])
                        
                        for dof_i = 1 : ndofs4item[nonfixed_ids[1]]
                            # apply action to fixed argument and first non-fixed argument
                            eval!(action_input, basisevaler[nonfixed_ids[1]], dof_i, i, offsets[nonfixed_ids[1]])
                            apply_action!(action_result, action_input, action, i)
            
                            for dof_j = 1 : ndofs4item[nonfixed_ids[2]]
                                temp = 0
                                for k = 1 : action_resultdim
                                    temp += action_result[k] * basisvals_testfunction[k,dof_j,i]
                                end
                                localmatrix[dof_i,dof_j] += temp * weights[i] 
                            end
                        end 
                    end
                else # fixed argument is the last one
                    for i in eachindex(weights)
    
                        # evaluate fixed argument into separate vector
                        fill!(evalfixedFE, 0)
                        eval!(evalfixedFE, basisevaler[fixed_argument], coeffs, i, 0)
                        
                        for dof_i = 1 : ndofs4item[nonfixed_ids[1]]
                            # apply action to fixed argument and first non-fixed argument
                            eval!(action_input, basisevaler[nonfixed_ids[1]], dof_i, i, 0)
                            
                            for dof_j = 1 : ndofs4item[nonfixed_ids[2]]
                                eval!(action_input, basisevaler[nonfixed_ids[2]], dof_j, i, offsets[2])
                                apply_action!(action_result, action_input, action, i)

                                temp = 0
                                for k = 1 : action_resultdim
                                    temp += action_result[k] * evalfixedFE[k]
                                end
                                localmatrix[dof_i,dof_j] += temp * weights[i]
                            end
                        end 
                    end
                end 
        
                # copy localmatrix into global matrix
                temp = AM.coeff4dofitem[fixed_argument][di[fixed_argument]] * AM.coeff4dofitem[nonfixed_ids[2]][di[nonfixed_ids[2]]] * AM.coeff4dofitem[nonfixed_ids[1]][di[nonfixed_ids[1]]] * xItemVolumes[item] * factor
                for dof_i = 1 : ndofs4item[nonfixed_ids[1]], dof_j = 1 : ndofs4item[nonfixed_ids[2]]
                    if localmatrix[dof_i,dof_j] != 0
                        if transposed_assembly == true
                            _addnz(A,dofs3[dof_j],dofs2[dof_i],localmatrix[dof_i,dof_j], temp)
                        else
                            _addnz(A,dofs2[dof_i],dofs3[dof_j],localmatrix[dof_i,dof_j], temp)
                        end
                    end
                end
                fill!(localmatrix,0.0)
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
assemble!(
    assemble!(
    b::AbstractVector,
    FE1::FEVectorBlock,
    FE2::FEVectorBlock.
    AP::AssemblyPattern{APT,T,AT};
    factor = 1)
````

Assembly of a TrilinearForm AP into given one-dimensional AbstractArray (e.g. a FEVectorBlock).
Here, the first two arguments are fixed by the given coefficients in FE1 and FE2.
"""
function assemble!(
    b::AbstractVector,
    FE1::FEVectorBlock,
    FE2::FEVectorBlock,
    AP::AssemblyPattern{APT,T,AT};
    skip_preps::Bool = false,
    factor::Real = 1,
    offset::Int = 0) where {APT <: APT_TrilinearForm, T <: Real, AT <: AbstractAssemblyType}

    # prepare assembly
    FE = AP.FES
    if !skip_preps
        prepare_assembly!(AP)
    end
    AM::AssemblyManager{T} = AP.AM
    xItemVolumes::Array{T,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemRegions::Union{VectorOfConstants{Int32}, Array{Int32,1}} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = length(xItemVolumes)

    # prepare action
    action = AP.action
    action_resultdim::Int = action.argsizes[1]
    action_input::Array{T,1} = zeros(T,action.argsizes[2]) # heap for action input
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output

    if AP.regions != [0]
        @logmsg MoreInfo "Assembling $(AP.name) for given $((p->p.name).([FE1,FE2])) into vector ($AT in regions = $(AP.regions)) into vector"
    else
        @logmsg MoreInfo "Assembling $(AP.name) for given $((p->p.name).([FE1,FE2])) into vector ($AT) into vector"
    end
    @debug AP

    # loop over items
    maxdofitems::Array{Int,1} = get_maxdofitems(AM)
    indexmap = CartesianIndices(zeros(Int, (maxdofitems[1],maxdofitems[2],maxdofitems[3])))
    basisevaler::Array{FEBasisEvaluator,1} = [get_basisevaler(AM, 1, 1),get_basisevaler(AM, 2, 1),get_basisevaler(AM, 3, 1)]
    basisvals_testfunction::Array{T,3} = basisevaler[3].cvals
    fixed_arguments::Array{Int,1} = [1,2]
    nonfixed_id::Int = 3
    coeffs1::Array{T,1} = zeros(T,get_maxndofs(AM,fixed_arguments[1]))
    coeffs2::Array{T,1} = zeros(T,get_maxndofs(AM,fixed_arguments[2]))
    dofs3::Array{Int,1} = zeros(Int,get_maxndofs(AM,nonfixed_id))
    dofitem::Array{Int,1} = [0,0,0]
    offsets::Array{Int,1} = [0, size(basisevaler[1].cvals,1), size(basisevaler[1].cvals,1) + size(basisevaler[2].cvals,1)]
    weights::Array{T,1} = get_qweights(AM) # somehow this saves A LOT allocations
    itemfactor::T = 0 # some temporary variable
    localb::Array{T,1} = zeros(T,length(dofs3))

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

        # loop over associated dofitems
        # di, dj == 2 is only performed if one of the operators jumps
        for di in indexmap
            dofitem[1] = AM.dofitems[1][di[1]]
            dofitem[2] = AM.dofitems[2][di[2]]
            dofitem[3] = AM.dofitems[3][di[3]]

            if dofitem[1] > 0 && dofitem[2] > 0 && dofitem[3] > 0

                # update FEbasisevaler
                basisevaler[1] = get_basisevaler(AM, 1, di[1])
                basisevaler[2] = get_basisevaler(AM, 2, di[2])
                basisevaler[3] = get_basisevaler(AM, 3, di[3])
                basisvals_testfunction = basisevaler[3].cvals

                # update action on dofitem
                update!(action, basisevaler[2], dofitem[2], item, regions[r])

                # update dofs of free arguments
                get_coeffs!(coeffs1, FE1, AM, fixed_arguments[1], di[fixed_arguments[1]])
                coeffs1 .*= AM.coeff4dofitem[fixed_arguments[1]][di[fixed_arguments[1]]]
                get_coeffs!(coeffs2, FE2, AM, fixed_arguments[2], di[fixed_arguments[2]])
                coeffs2 .*= AM.coeff4dofitem[fixed_arguments[2]][di[fixed_arguments[2]]]
                get_dofs!(dofs3, AM, nonfixed_id, di[nonfixed_id])

                for i in eachindex(weights)
                    # evaluate first and second component
                    fill!(action_input, 0)
                    eval!(action_input, basisevaler[fixed_arguments[1]], coeffs1, i)
                    eval!(action_input, basisevaler[fixed_arguments[2]], coeffs2, i, offsets[2])
        
                    # apply action to FE1 and FE2
                    apply_action!(action_result, action_input, action, i)
                   
                    # multiply third component
                    for dof_j = 1 : get_ndofs(AM, 3, di[3])
                        temp = 0
                        for k = 1 : action_resultdim
                            temp += action_result[k] * basisvals_testfunction[k,dof_j,i]
                        end
                        localb[dof_j] += temp * weights[i]
                    end 
                end 
        
                itemfactor = factor * xItemVolumes[item] * AM.coeff4dofitem[nonfixed_id][di[nonfixed_id]]
                for dof_i = 1 : get_ndofs(AM, 3, di[3])
                    b[get_dof(AM, 3, di[3], dof_i) + offset] += localb[dof_i] * itemfactor
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


## wrapper for FEVectorBlock to avoid use of setindex! functions of FEMAtrixBlock
function assemble!(
    b::FEVectorBlock,
    FE1::FEVectorBlock,
    FE2::FEVectorBlock,
    AP::AssemblyPattern{APT,T,AT};
    skip_preps::Bool = false,
    factor::Real = 1) where {APT <: APT_TrilinearForm, T <: Real, AT <: AbstractAssemblyType}

    assemble!(b.entries, FE1, FE2, AP; factor = factor, offset = b.offset, skip_preps = skip_preps)
end