
abstract type APT_BilinearForm <: AssemblyPatternType end
abstract type APT_SymmetricBilinearForm <: APT_BilinearForm end

"""
````
function SymmetricBilinearForm(
    T::Type{<:Real},
    AT::Type{<:AbstractAssemblyType},
    FE::Array{FESpace,1},
    operators::Array{DataType,1}, 
    action::AbstractAction; 
    regions::Array{Int,1} = [0])
````

Creates a symmetric BilinearForm assembly pattern with the given FESpaces, operators and action etc. Symmetry is not checked automatically, but is assumed during assembly!
"""
function SymmetricBilinearForm(T::Type{<:Real}, AT::Type{<:AbstractAssemblyType}, FES, operators, action; regions = [0])
    return AssemblyPattern{APT_SymmetricBilinearForm, T, AT}(FES,operators,action,regions)
end

"""
````
function BilinearForm(
    T::Type{<:Real},
    AT::Type{<:AbstractAssemblyType},
    FE::Array{FESpace,1},
    operators::Array{DataType,1}, 
    action::AbstractAction; 
    regions::Array{Int,1} = [0])
````

Creates a (unsymmetric) BilinearForm assembly pattern with the given FESpaces, operators and action etc.
"""
function BilinearForm(T::Type{<:Real}, AT::Type{<:AbstractAssemblyType}, FES, operators, action; regions = [0])
    return AssemblyPattern{APT_BilinearForm, T, AT}(FES,operators,action,regions)
end


"""
````
assemble!(
    A::AbstractArray{T,2},
    AP::AssemblyPattern{APT,T,AT};
    apply_action_to::Int = 1,
    verbosity::Int = 0,
    factor = 1,
    transposed_assembly::Bool = false,
    transpose_copy = Nothing)  where {APT <: APT_BilinearForm, T <: Real, AT <: AbstractAssemblyType}
````

Assembly of a BilinearForm BLF into given two-dimensional AbstractArray (e.g. FEMatrixBlock).
"""
function assemble!(
    A::AbstractArray{T,2},
    AP::AssemblyPattern{APT,T,AT};
    apply_action_to::Int = 1,
    verbosity::Int = 0,
    factor = 1,
    transposed_assembly::Bool = false,
    transpose_copy = nothing,
    skip_preps::Bool = false,
    offsetX = 0,
    offsetY = 0) where {APT <: APT_BilinearForm, T <: Real, AT <: AbstractAssemblyType}

    # prepare assembly
    FE = AP.FES
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
    action_input::Array{T,1} = zeros(T,action.argsizes[2]) # heap for action input
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output

    if verbosity > 0
        println("  Assembling ($APT,$AT,$T) into matrix (transposed_assembly = $transposed_assembly)")
        println("   skip_preps = $skip_preps")
        println("    operators = $(AP.operators)")
        println("      regions = $(AP.regions)")
        println("       factor = $factor")
        println("       action = $(AP.action.name) (apply_to = $apply_action_to, size = $(action.argsizes))")
        println("        qf[1] = $(AM.qf[1].name) ")
    end
 
    # loop over items
    weights::Array{T,1} = get_qweights(AM) # somehow this saves A LOT allocations
    basisevaler::Array{FEBasisEvaluator,1} = [get_basisevaler(AM, 1, 1), get_basisevaler(AM, 2, 1)]
    basisvals::Array{T,3} = basisevaler[1].cvals
    localmatrix::Array{T,2} = zeros(T,get_maxndofs(AM,1),get_maxndofs(AM,2))
    temp::T = 0 # some temporary variable
    ndofs4dofitem::Array{Int,1} = [0,0]
    acol::Int = 0
    arow::Int = 0
    dofitems::Array{Int,1} = [0,0]
    is_symmetric::Bool = APT <: APT_SymmetricBilinearForm
    itemfactor::T = 0
    maxdofitems::Array{Int,1} = get_maxdofitems(AM)
    indexmap = CartesianIndices(zeros(Int, maxdofitems[1],maxdofitems[2]))

    other_id::Int = apply_action_to == 1 ? 2 : 1
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
            dofitems[1] = AM.dofitems[1][di[1]]
            dofitems[2] = AM.dofitems[2][di[2]]
            if dofitems[1] > 0 && dofitems[2] > 0

                # even if global matrix is symmeric, local matrix might be not in case of JumpOperators
                is_locally_symmetric = is_symmetric * (dofitems[1] == dofitems[2])

                # get number of dofs on this dofitem
                ndofs4dofitem[1] = get_ndofs(AM, 1, di[1])
                ndofs4dofitem[2] = get_ndofs(AM, 2, di[2])

                # update FEbasisevaler
                basisevaler[1] = get_basisevaler(AM, 1, di[1])
                basisevaler[2] = get_basisevaler(AM, 2, di[2])
                basisvals = basisevaler[other_id].cvals

                # update action on dofitem
                update!(action, basisevaler[apply_action_to], dofitems[apply_action_to], item, regions[r])

                for i in eachindex(weights)
                    for dof_i = 1 : ndofs4dofitem[apply_action_to]
                        eval!(action_input, basisevaler[apply_action_to], dof_i, i)
                        action_input .*= AM.coeff4dofitem[apply_action_to][di[apply_action_to]]
                        apply_action!(action_result, action_input, action, i)
                        if is_locally_symmetric == false
                            for dof_j = 1 : ndofs4dofitem[other_id]
                                temp = 0
                                for k = 1 : action_resultdim
                                    temp += action_result[k] * basisvals[k,dof_j,i]
                                end
                                localmatrix[dof_i,dof_j] += weights[i] * temp
                            end
                        else # symmetric case
                            for dof_j = dof_i : ndofs4dofitem[other_id]
                                temp = 0
                                for k = 1 : action_resultdim
                                    temp += action_result[k] * basisvals[k,dof_j,i]
                                end
                                localmatrix[dof_i,dof_j] += weights[i] * temp
                            end
                        end
                    end 
                end


                # copy localmatrix into global matrix
                itemfactor = xItemVolumes[item] * factor * AM.coeff4dofitem[other_id][di[other_id]]
                if is_locally_symmetric == false
                    for dof_i = 1 : ndofs4dofitem[1]
                        arow = get_dof(AM, 1, di[1], dof_i) + offsetX
                        for dof_j = 1 : ndofs4dofitem[2]
                            if localmatrix[dof_i,dof_j] != 0
                                acol = get_dof(AM, 2, di[2], dof_j) + offsetY
                                if transposed_assembly == true
                                    _addnz(A,acol,arow,localmatrix[dof_i,dof_j] * itemfactor,1)
                                else 
                                    _addnz(A,arow,acol,localmatrix[dof_i,dof_j] * itemfactor,1)  
                                end
                                if transpose_copy != nothing # sign is changed in case nonzero rhs data is applied to LagrangeMultiplier (good idea?)
                                    if transposed_assembly == true
                                        _addnz(transpose_copy,arow,acol,localmatrix[dof_i,dof_j] * itemfactor,-1)
                                    else
                                        _addnz(transpose_copy,acol,arow,localmatrix[dof_i,dof_j] * itemfactor,-1)
                                    end
                                end
                            end
                        end
                    end
                else # symmetric case
                    for dof_i = 1 : ndofs4dofitem[1]
                        for dof_j = dof_i+1 : ndofs4dofitem[2]
                            if localmatrix[dof_i,dof_j] != 0 
                                arow = get_dof(AM, 1, di[1], dof_i) + offsetX
                                acol = get_dof(AM, 2, di[2], dof_j) + offsetY
                                _addnz(A,arow,acol,localmatrix[dof_i,dof_j] * itemfactor,1)
                                arow = get_dof(AM, 1, di[1], dof_j) + offsetX
                                acol = get_dof(AM, 2, di[2], dof_i) + offsetY
                                _addnz(A,arow,acol,localmatrix[dof_i,dof_j] * itemfactor,1)
                            end
                        end
                    end    
                    for dof_i = 1 : ndofs4dofitem[1]
                        arow = get_dof(AM, 2, di[1], dof_i) + offsetX
                        acol = get_dof(AM, 1, di[2], dof_i) + offsetY
                       _addnz(A,arow,acol,localmatrix[dof_i,dof_i] * itemfactor,1)
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


## wrapper for FEMatrixBlock to avoid use of setindex! functions of FEMAtrixBlock
function assemble!(
    A::FEMatrixBlock,
    AP::AssemblyPattern{APT,T,AT};
    apply_action_to::Int = 1,
    verbosity::Int = 0,
    factor = 1,
    skip_preps::Bool = false,
    transposed_assembly::Bool = false,
    transpose_copy = nothing) where {APT <: APT_BilinearForm, T <: Real, AT <: AbstractAssemblyType}

    if typeof(transpose_copy) <: FEMatrixBlock
        assemble!(A.entries, AP; apply_action_to = apply_action_to, verbosity = verbosity, factor = factor, transposed_assembly = transposed_assembly, transpose_copy = transpose_copy.entries, offsetX = A.offsetX, offsetY = A.offsetY, skip_preps = skip_preps)
    else
        assemble!(A.entries, AP; apply_action_to = apply_action_to, verbosity = verbosity, factor = factor, transposed_assembly = transposed_assembly, transpose_copy = transpose_copy, offsetX = A.offsetX, offsetY = A.offsetY, skip_preps = skip_preps)
    end
end



"""
````
assemble!(
    b::AbstractArray{T,1},
    fixedFE::FEVectorBlock,    # coefficient for fixed 2nd component
    AP::AssemblyPattern{APT,T,AT};
    apply_action_to::Int = 1,
    fixed_argument::Int = 2,
    factor = 1,
    verbosity::Int = 0) where where {APT <: APT_BilinearForm, T <: Real, AT <: AbstractAssemblyType}
````

Assembly of a BilinearForm BLF into given one-dimensional AbstractArray (e.g. a FEVectorBlock).
Here, the second argument is fixed (default) by the given coefficients in fixedFE.
With apply_action_to=2 the action can be also applied to the second argument instead of the first one (default).
"""
function assemble!(
    b::AbstractArray{T,1},
    fixedFE::AbstractArray{T,1},    # coefficient for fixed 2nd component
    AP::AssemblyPattern{APT,T,AT};
    apply_action_to::Int = 1,
    fixed_argument::Int = 2,
    factor = 1,
    skip_preps::Bool = false,
    verbosity::Int = 0,
    offsets::Array{Int,1} = [0,0]) where {APT <: APT_BilinearForm, T <: Real, AT <: AbstractAssemblyType}

    # prepare assembly
    FE = AP.FES
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
    action_input::Array{T,1} = zeros(T,action.argsizes[2]) # heap for action input
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output

    if verbosity > 0
        println("  Assembling ($APT,$AT,$T) into vector with fixed argument $fixed_argument")
        println("   skip_preps = $skip_preps")
        println("    operators = $(AP.operators)")
        println("      regions = $(AP.regions)")
        println("       factor = $factor")
        println("       action = $(AP.action.name) (apply_to = $apply_action_to, size = $(action.argsizes))")
        println("        qf[1] = $(AM.qf[1].name) ")
        
    end

    # loop over items
    weights::Array{T,1} = get_qweights(AM) # somehow this saves A LOT allocations
    fixed_coeffs::Array{T,1} = zeros(T,get_maxndofs(AM,fixed_argument))
    basisevaler::Array{FEBasisEvaluator,1} = [get_basisevaler(AM, 1, 1), get_basisevaler(AM, 2, 1)]
    basisvals::Array{T,3} = basisevaler[1].cvals
    temp::T = 0 # some temporary variable
    ndofs4dofitem::Array{Int,1} = [0,0]
    bdof::Int = 0
    dofitems::Array{Int,1} = [0,0]
    is_symmetric::Bool = APT <: APT_SymmetricBilinearForm
    itemfactor::T = 0
    maxdofitems::Array{Int,1} = get_maxdofitems(AM)
    indexmap = CartesianIndices(zeros(Int, maxdofitems[1],maxdofitems[2]))
    if apply_action_to == fixed_argument
        fixedval = zeros(T, action.argsizes[2]) # some temporary variable
    else
        fixedval = zeros(T, action.argsizes[1]) # some temporary variable
    end
    free_argument::Int = fixed_argument == 1 ? 2 : 1
    localb::Array{T,2} = zeros(T,get_maxndofs(AM)[free_argument],action_resultdim)

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
            dofitems[1] = AM.dofitems[1][di[1]]
            dofitems[2] = AM.dofitems[2][di[2]]
            if dofitems[1] > 0 && dofitems[2] > 0

                # get number of dofs on this dofitem
                ndofs4dofitem[1] = get_ndofs(AM, 1, di[1])
                ndofs4dofitem[2] = get_ndofs(AM, 2, di[2])

                # update FEbasisevaler
                basisevaler[1] = get_basisevaler(AM, 1, di[1])
                basisevaler[2] = get_basisevaler(AM, 2, di[2])
                basisvals = basisevaler[free_argument].cvals

                # update action on dofitem
                update!(action, basisevaler[apply_action_to], dofitems[apply_action_to], item, regions[r])

                # update dofs
                get_coeffs!(fixed_coeffs, fixedFE, AM, fixed_argument, di[fixed_argument], offsets[2])

                for i in eachindex(weights)
                    # evaluate fixed argument
                    fill!(fixedval, 0.0)
                    eval!(fixedval, basisevaler[fixed_argument], fixed_coeffs, i)
                    fixedval .*= AM.coeff4dofitem[fixed_argument][di[fixed_argument]]

                    if apply_action_to == fixed_argument
                        # apply action to fixed argument
                        apply_action!(action_result, fixedval, action, i)

                        # multiply free argument
                        action_result .*= AM.coeff4dofitem[free_argument][di[free_argument]]
                        for dof_i = 1 : ndofs4dofitem[free_argument]
                            temp = 0
                            for k = 1 : action_resultdim
                                temp += action_result[k] * basisvals[k,dof_i,i]
                            end
                            localb[dof_i] += temp * weights[i]
                        end 
                    else
                        for dof_i = 1 : ndofs4dofitem[free_argument]
                            # apply action to free argument
                            eval!(action_input, basisevaler[free_argument], dof_i, i)
                            action_input .*= AM.coeff4dofitem[free_argument][di[free_argument]]
                            apply_action!(action_result, action_input, action, i)
            
                            # multiply fixed argument
                            temp = 0
                            for k = 1 : action_resultdim
                                temp += action_result[k] * fixedval[k]
                            end
                            localb[dof_i] += temp * weights[i]
                        end 
                    end
                end

                ## copy to global vector
                itemfactor = xItemVolumes[item] * factor
                for dof_i = 1 : ndofs4dofitem[free_argument]
                    bdof = get_dof(AM, free_argument, di[free_argument], dof_i) + offsets[1]
                    b[bdof] += localb[dof_i] * itemfactor
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


# wrapper for FEVectorBlock to avoid setindex! functions of FEVectorBlock
function assemble!(
    b::FEVectorBlock,
    fixedFE::FEVectorBlock,    # coefficient for fixed argument
    AP::AssemblyPattern{APT,T,AT};
    apply_action_to::Int = 1,
    fixed_argument::Int = 2,
    skip_preps::Bool = false,
    factor = 1,
    verbosity::Int = 0) where {APT <: APT_BilinearForm, T <: Real, AT <: AbstractAssemblyType}

    @assert fixedFE.FES == AP.FES[fixed_argument]

    assemble!(b.entries, fixedFE.entries, AP; apply_action_to = apply_action_to, factor = factor, fixed_argument = fixed_argument, verbosity = verbosity, offsets = [b.offset, fixedFE.offset], skip_preps = skip_preps)
end