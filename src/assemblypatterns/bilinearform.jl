
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
    return AssemblyPattern{APT_SymmetricBilinearForm, T, AT}(FES,operators,action,regions,AssemblyPatternPreparations(nothing,nothing,nothing,nothing,nothing,nothing))
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
    return AssemblyPattern{APT_BilinearForm, T, AT}(FES,operators,action,regions,AssemblyPatternPreparations(nothing,nothing,nothing,nothing,nothing,nothing))
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
    action = AP.action
    if !skip_preps
        prepare_assembly!(AP; verbosity = verbosity - 1)
    end
    EG = AP.APP.EG
    ndofs4EG::Array{Array{Int,1},1} = AP.APP.ndofs4EG
    qf::Array{QuadratureRule,1} = AP.APP.qf
    basisevaler::Array{FEBasisEvaluator,4} = AP.APP.basisevaler
    dii4op::Array{Function,1} = AP.APP.dii4op
    xItemVolumes::Array{T,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemDofs1::Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}} = Dofmap4AssemblyType(FE[1], AP.APP.basisAT[1])
    xItemDofs2::Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}} = Dofmap4AssemblyType(FE[2], AP.APP.basisAT[2])
    xItemRegions::Union{VectorOfConstants{Int32}, Array{Int32,1}} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = length(xItemVolumes)


    # get size informations
    ncomponents::Int = get_ncomponents(eltype(FE[1]))
    cvals_resultdim::Int = size(basisevaler[end,apply_action_to,1,1].cvals,1)
    action_resultdim::Int = action.argsizes[1]

    if verbosity > 0
        println("  Assembling ($APT,$AT,$T) into matrix (transposed_assembly = $transposed_assembly)")
        println("   skip_preps = $skip_preps")
        println("    operators = $(AP.operators)")
        println("      regions = $(AP.regions)")
        println("       factor = $factor")
        println("       action = $(AP.action.name) (apply_to = $apply_action_to, size = $(action.argsizes))")
        println("        qf[1] = $(qf[1].name) ")
        println("           EG = $EG")
    end
 
    # loop over items
    EG4item::Int = 1
    EG4dofitem1::Array{Int,1} = [1,1] # EG id of the current item with respect to operator 1
    EG4dofitem2::Array{Int,1} = [1,1] # EG id of the current item with respect to operator 2
    dofitems1::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found (operator 1)
    dofitems2::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found (operator 2)
    itempos4dofitem1::Array{Int,1} = [1,1] # local item position in dofitem1
    itempos4dofitem2::Array{Int,1} = [1,1] # local item position in dofitem2
    orientation4dofitem1::Array{Int,1} = [1,2] # local orientation
    orientation4dofitem2::Array{Int,1} = [1,2] # local orientation
    coefficient4dofitem1::Array{T,1} = [0.0,0.0] # coefficients for operator 1
    coefficient4dofitem2::Array{T,1} = [0.0,0.0] # coefficients for operator 2
    ndofs4item1::Int = 0 # number of dofs for item
    ndofs4item2::Int = 0 # number of dofs for item
    dofitem1 = 0
    dofitem2 = 0
    maxdofs1::Int = max_num_targets_per_source(xItemDofs1)
    maxdofs2::Int = max_num_targets_per_source(xItemDofs2)
    dofs = zeros(Int,maxdofs1)
    dofs2 = zeros(Int,maxdofs2)
    action_input::Array{T,1} = zeros(T,cvals_resultdim) # heap for action input
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations
    basisevaler4dofitem1::FEBasisEvaluator = basisevaler[1,1,1,1]
    basisevaler4dofitem2::FEBasisEvaluator = basisevaler[1,2,1,1]
    basisvals1::Array{T,3} = basisevaler4dofitem1.cvals
    basisvals2::Array{T,3} = basisevaler4dofitem2.cvals
    localmatrix::Array{T,2} = zeros(T,maxdofs1,maxdofs2)
    temp::T = 0 # some temporary variable
    acol::Int = 0
    arow::Int = 0
    is_symmetric::Bool = APT <: APT_SymmetricBilinearForm
    itemfactor::T = 0
    
    regions::Array{Int,1} = AP.regions
    allitems::Bool = (regions == [0])
    nregions::Int = length(regions)
    for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if allitems || xItemRegions[item] == regions[r]

        # get dofitem informations
        EG4item = dii4op[1](dofitems1, EG4dofitem1, itempos4dofitem1, coefficient4dofitem1, orientation4dofitem1, item)
        dii4op[2](dofitems2, EG4dofitem2, itempos4dofitem2, coefficient4dofitem2, orientation4dofitem2, item)

        # get quadrature weights for integration domain
        weights = qf[EG4item].w
        itemfactor = xItemVolumes[item] * factor

        # loop over associated dofitems (maximal 2 for jump calculations)
        # di, dj == 2 is only performed if one of the operators jumps
        for di = 1 : 2, dj = 1 : 2
            dofitem1 = dofitems1[di]
            dofitem2 = dofitems2[dj]
            if dofitem1 > 0 && dofitem2 > 0

                # even if global matrix is symmeric, local matrix might be not in case of JumpOperators
                is_locally_symmetric = is_symmetric * (dofitem1 == dofitem2)

                # get number of dofs on this dofitem
                ndofs4item1 = ndofs4EG[1][EG4dofitem1[di]]
                ndofs4item2 = ndofs4EG[2][EG4dofitem2[dj]]

                # update FEbasisevaler
                basisevaler4dofitem1 = basisevaler[EG4dofitem1[di],1,itempos4dofitem1[di],orientation4dofitem1[di]]
                basisevaler4dofitem2 = basisevaler[EG4dofitem2[dj],2,itempos4dofitem2[dj],orientation4dofitem2[dj]]
                basisvals1 = basisevaler4dofitem1.cvals
                basisvals2 = basisevaler4dofitem2.cvals
                update!(basisevaler4dofitem1,dofitem1)
                update!(basisevaler4dofitem2,dofitem2)

                # update action on dofitem
                if apply_action_to == 1
                    update!(action, basisevaler4dofitem1, dofitem1, item, regions[r])
                else
                    update!(action, basisevaler4dofitem2, dofitem2, item, regions[r])
                end

                # update dofs
                for j=1:ndofs4item1
                    dofs[j] = xItemDofs1[j,dofitem1]
                end
                for j=1:ndofs4item2
                    dofs2[j] = xItemDofs2[j,dofitem2]
                end

                for i in eachindex(weights)
                    if apply_action_to == 1
                        for dof_i = 1 : ndofs4item1
                            eval!(action_input, basisevaler4dofitem1, dof_i, i)
                            apply_action!(action_result, action_input, action, i)
                            action_result .*= coefficient4dofitem1[di]

                            if is_locally_symmetric == false
                                for dof_j = 1 : ndofs4item2
                                    temp = 0
                                    for k = 1 : action_resultdim
                                        temp += action_result[k] * basisvals2[k,dof_j,i]
                                    end
                                    temp *= coefficient4dofitem2[dj]
                                    localmatrix[dof_i,dof_j] += weights[i] * temp
                                end
                            else # symmetric case
                                for dof_j = dof_i : ndofs4item2
                                    temp = 0
                                    for k = 1 : action_resultdim
                                        temp += action_result[k] * basisvals2[k,dof_j,i]
                                    end
                                    temp *= coefficient4dofitem2[dj]
                                    localmatrix[dof_i,dof_j] += weights[i] * temp
                                end
                            end
                        end 
                    else
                        for dof_j = 1 : ndofs4item2
                            eval!(action_input, basisevaler4dofitem2, dof_j, i)
                            apply_action!(action_result, action_input, action, i)
                            action_result .*= coefficient4dofitem2[dj]

                            if is_locally_symmetric == false
                                for dof_i = 1 : ndofs4item1
                                    temp = 0
                                    for k = 1 : action_resultdim
                                        temp += action_result[k] * basisvals1[k,dof_j,i]
                                    end
                                    temp *= coefficient4dofitem1[di]
                                    localmatrix[dof_i,dof_j] += weights[i] * temp
                                end
                            else # symmetric case
                                for dof_i = dof_j : ndofs4item1
                                    temp = 0
                                    for k = 1 : action_resultdim
                                        temp += action_result[k] * basisvals1[k,dof_j,i]
                                    end
                                    temp *= coefficient4dofitem1[di]
                                    localmatrix[dof_i,dof_j] += weights[i] * temp
                                end
                            end
                        end 
                    end
                end

                # copy localmatrix into global matrix
                if is_locally_symmetric == false
                    for dof_i = 1 : ndofs4item1
                        arow = dofs[dof_i] + offsetX
                        for dof_j = 1 : ndofs4item2
                            if localmatrix[dof_i,dof_j] != 0
                                acol = dofs2[dof_j] + offsetY
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
                    for dof_i = 1 : ndofs4item1
                        for dof_j = dof_i+1 : ndofs4item2
                            if localmatrix[dof_i,dof_j] != 0 
                                arow = dofs[dof_i] + offsetX
                                acol = dofs2[dof_j] + offsetY
                                _addnz(A,arow,acol,localmatrix[dof_i,dof_j] * itemfactor,1)
                                arow = dofs[dof_j] + offsetX
                                acol = dofs2[dof_i] + offsetY
                                _addnz(A,arow,acol,localmatrix[dof_i,dof_j] * itemfactor,1)
                            end
                        end
                    end    
                    for dof_i = 1 : ndofs4item1
                        arow = dofs2[dof_i] + offsetX
                        acol = dofs[dof_i] + offsetY
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
    action = AP.action
    if !skip_preps
        prepare_assembly!(AP; verbosity = verbosity - 1)
    end
    EG = AP.APP.EG
    ndofs4EG::Array{Array{Int,1},1} = AP.APP.ndofs4EG
    qf::Array{QuadratureRule,1} = AP.APP.qf
    basisevaler::Array{FEBasisEvaluator,4} = AP.APP.basisevaler
    dii4op::Array{Function,1} = AP.APP.dii4op
    xItemVolumes::Array{T,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemDofs::Array{Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}},1} = [
        Dofmap4AssemblyType(FE[1], AP.APP.basisAT[1]),
        Dofmap4AssemblyType(FE[2], AP.APP.basisAT[2])]
    xItemRegions::Union{VectorOfConstants{Int32}, Array{Int32,1}} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = length(xItemVolumes)


    # get size informations
    free_argument = 1
    if fixed_argument == 1
        free_argument = 2
    end

    ncomponents::Int = get_ncomponents(eltype(FE[1]))
    ncomponents2::Int = get_ncomponents(eltype(FE[2]))
    cvals_resultdim::Int = size(basisevaler[1,apply_action_to,1,1].cvals,1)
    cvals_resultdim2::Int = size(basisevaler[1,fixed_argument,1,1].cvals,1)
    action_resultdim::Int = action.argsizes[1]

    if verbosity > 0
        println("  Assembling ($APT,$AT,$T) into vector with fixed argument $fixed_argument")
        println("   skip_preps = $skip_preps")
        println("    operators = $(AP.operators)")
        println("      regions = $(AP.regions)")
        println("       factor = $factor")
        println("       action = $(AP.action.name) (apply_to = $apply_action_to, size = $(action.argsizes))")
        println("        qf[1] = $(qf[1].name) ")
        println("           EG = $EG")
    end

    # loop over items
    EG4dofitem1::Array{Int,1} = [1,1] # EG id of the current item with respect to operator 1
    EG4dofitem2::Array{Int,1} = [1,1] # EG id of the current item with respect to operator 2
    dofitems1::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found (operator 1)
    dofitems2::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found (operator 2)
    itempos4dofitem1::Array{Int,1} = [1,1] # local item position in dofitem1
    itempos4dofitem2::Array{Int,1} = [1,1] # local item position in dofitem2
    orientation4dofitem1::Array{Int,1} = [1,2] # local orientation
    orientation4dofitem2::Array{Int,1} = [1,2] # local orientation
    coefficient4dofitem1::Array{T,1} = [0,0] # coefficients for operator 1
    coefficient4dofitem2::Array{T,1} = [0,0] # coefficients for operator 2
    ndofs4item::Array{Int,1} = [0,0] # number of dofs for item
    dofitems::Array{Int,1} = [0,0]
    maxdofs::Array{Int,1} = [max_num_targets_per_source(xItemDofs[1]), max_num_targets_per_source(xItemDofs[2])]
    fixed_coeffs = zeros(T,maxdofs[fixed_argument])
    action_input::Array{T,1} = zeros(T,cvals_resultdim) # heap for action input
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations
    basisevaler4dofitem::Array{FEBasisEvaluator,1} = [basisevaler[1,1,1,1], basisevaler[1,2,1,1]]
    basisvals::Array{T,3} = basisevaler4dofitem[free_argument].cvals
    fixedval = zeros(T, cvals_resultdim2) # some temporary variable
    temp::T = 0 # some temporary variable
    localb::Array{T,1} = zeros(T,maxdofs[free_argument])
    bdof::Int = 0
    fdof::Int = 0
    itemfactor::T = 0     

    regions::Array{Int,1} = AP.regions
    allitems::Bool = (regions == [0])
    nregions::Int = length(regions)
    for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if allitems || xItemRegions[item] == regions[r]

        # get dofitem informations
        dii4op[1](dofitems1, EG4dofitem1, itempos4dofitem1, coefficient4dofitem1, orientation4dofitem1, item)
        dii4op[2](dofitems2, EG4dofitem2, itempos4dofitem2, coefficient4dofitem2, orientation4dofitem2, item)

        # get quadrature weights for integration domain
        weights = qf[EG4dofitem1[1]].w
        itemfactor = factor * xItemVolumes[item]

        # loop over associated dofitems (maximal 2 for jump calculations)
        # di, dj == 2 is only performed if one of the operators jumps
        for di = 1 : 2, dj = 1 : 2
            dofitems[1] = dofitems1[di]
            dofitems[2] = dofitems2[dj]
            if dofitems[1] > 0 && dofitems[2] > 0

                # get number of dofs on this dofitem
                ndofs4item[1] = ndofs4EG[1][EG4dofitem1[di]]
                ndofs4item[2] = ndofs4EG[2][EG4dofitem2[dj]]

                # update FEbasisevaler
                basisevaler4dofitem[1] = basisevaler[EG4dofitem1[di],1,itempos4dofitem1[di],orientation4dofitem1[di]]
                basisevaler4dofitem[2] = basisevaler[EG4dofitem2[dj],2,itempos4dofitem2[dj],orientation4dofitem2[dj]]
                basisvals = basisevaler4dofitem[free_argument].cvals
                update!(basisevaler4dofitem[1],dofitems[1])
                update!(basisevaler4dofitem[2],dofitems[2])

                # update action on dofitem
                update!(action, basisevaler4dofitem[apply_action_to], dofitems[apply_action_to], item, regions[r])

                # update dofs
                for j=1:ndofs4item[fixed_argument]
                    fdof = xItemDofs[fixed_argument][j,dofitems[fixed_argument]] + offsets[fixed_argument]
                    if fixed_argument == 1
                        fixed_coeffs[j] = fixedFE[fdof] * coefficient4dofitem1[di]
                    else
                        fixed_coeffs[j] = fixedFE[fdof] * coefficient4dofitem2[dj]
                    end
                end


                for i in eachindex(weights)
                
                    # evaluate fixed argument
                    fill!(fixedval, 0.0)
                    eval!(fixedval, basisevaler4dofitem[fixed_argument], fixed_coeffs, i)

                    if apply_action_to == fixed_argument
                        # apply action to fixed argument
                        apply_action!(action_result, fixedval, action, i)

                        # multiply free argument
                        for dof_i = 1 : ndofs4item[free_argument]
                            temp = 0
                            for k = 1 : action_resultdim
                                temp += action_result[k] * basisvals[k,dof_i,i]
                            end
                            if free_argument == 1
                                temp *= coefficient4dofitem1[di]
                            else
                                temp *= coefficient4dofitem2[dj]
                            end
                            localb[dof_i] += temp * weights[i]
                        end 
                    else
                        for dof_i = 1 : ndofs4item[free_argument]
                            # apply action to free argument
                            eval!(action_input, basisevaler4dofitem[free_argument], dof_i, i)
                            if free_argument == 1
                                action_input .*= coefficient4dofitem1[di]
                            else
                                action_input .*= coefficient4dofitem2[dj]
                            end
                            apply_action!(action_result, action_input, action, i)
            
                            # multiply second argument
                            temp = 0
                            for k = 1 : action_resultdim
                                temp += action_result[k] * fixedval[k]
                            end
                            localb[dof_i] += temp * weights[i]
                        end 
                    end
                end

                for dof_i = 1 : ndofs4item[free_argument]
                    bdof = xItemDofs[free_argument][dof_i,dofitems[free_argument]] + offsets[free_argument]
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



