abstract type APT_TrilinearForm <: AssemblyPatternType end

"""
````
function TrilinearForm(
    T::Type{<:Real},
    AT::Type{<:AbstractAssemblyType},
    FE1::FESpace,
    FE2::FESpace,
    FE3::FESpace,
    operator1::Type{<:AbstractFunctionOperator},
    operator2::Type{<:AbstractFunctionOperator},
    operator3::Type{<:AbstractFunctionOperator},
    action::AbstractAction; # is only applied to FE1/operator1 + FE2/operator2
    regions::Array{Int,1} = [0])
````

Creates a TrilinearForm that can be assembeld into a matrix (with one argument fixed) or into a vector (with two fixed arguments).
"""
function TrilinearForm(
    T::Type{<:Real},
    AT::Type{<:AbstractAssemblyType},
    FES::Array{FESpace,1},
    operators::Array{DataType,1},
    action::AbstractAction;
    regions::Array{Int,1} = [0])
    return  AssemblyPattern{APT_TrilinearForm, T, AT}(FES,operators,action,regions,AssemblyPatternPreparations(nothing,nothing,nothing,nothing,nothing))
end


"""
````
assemble!(
    assemble!(
    A::AbstractArray{T,2},
    FE1::FEVectorBlock,
    AP::TrilinearForm{T, AT};
    verbosity::Int = 0,
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
    verbosity::Int = 0,
    fixed_argument::Int = 1,
    transposed_assembly::Bool = false,
    skip_preps::Bool = false,
    factor = 1) where {APT <: APT_TrilinearForm, T <: Real, AT <: AbstractAssemblyType}

    # get adjacencies
    FE = AP.FES
    xItemVolumes::Array{T,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemDofs::Array{Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}},1} = [Dofmap4AssemblyType(FE[1], DofitemAT4Operator(AT, AP.operators[1])),
                 Dofmap4AssemblyType(FE[2], DofitemAT4Operator(AT, AP.operators[2])),
                 Dofmap4AssemblyType(FE[3], DofitemAT4Operator(AT, AP.operators[3]))]
    xItemRegions::Union{VectorOfConstants{Int32}, Array{Int32,1}} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = length(xItemVolumes)

    # prepare assembly
    action = AP.action
    if !skip_preps
        prepare_assembly!(AP; verbosity = verbosity - 1)
    end
    EG = AP.APP.EG
    ndofs4EG::Array{Array{Int,1},1} = AP.APP.ndofs4EG
    qf::Array{QuadratureRule,1} = AP.APP.qf
    basisevaler::Array{FEBasisEvaluator,4} = AP.APP.basisevaler
    dii4op::Array{Function,1} = AP.APP.dii4op

    # get size informations
    ncomponents = zeros(Int,length(FE))
    maxdofs = 0
    for j = 1 : length(FE)
        ncomponents[j] = get_ncomponents(eltype(FE[j]))
        maxdofs = max(maxdofs, max_num_targets_per_source(xItemDofs[j]))
    end
    action_resultdim::Int = action.argsizes[1]

    if verbosity > 0
        println("  Assembling ($APT,$AT,$T) into matrix (transposed_assembly = $transposed_assembly) with fixed_argument = $fixed_argument")
        println("   skip_preps = $skip_preps")
        println("    operators = $(AP.operators)")
        println("      regions = $(AP.regions)")
        println("       factor = $factor")
        println("       action = $(AP.action.name) (apply_to = [1,2], size = $(action.argsizes))")
        println("        qf[1] = $(qf[1].name) ")
        println("           EG = $EG")
    end

    # loop over items
    EG4item::Int = 1
    EG4dofitem1::Array{Int,1} = [1,1] # EG id of the current item with respect to operator 1
    EG4dofitem2::Array{Int,1} = [1,1] # EG id of the current item with respect to operator 2
    EG4dofitem3::Array{Int,1} = [1,1] # EG id of the current item with respect to operator 3
    dofitems1::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found (operator 1)
    dofitems2::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found (operator 2)
    dofitems3::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found (operator 3)
    itempos4dofitem1::Array{Int,1} = [1,1] # local item position in dofitem1
    itempos4dofitem2::Array{Int,1} = [1,1] # local item position in dofitem2
    itempos4dofitem3::Array{Int,1} = [1,1] # local item position in dofitem3
    orientation4dofitem1::Array{Int,1} = [1,2] # local orientation
    orientation4dofitem2::Array{Int,1} = [1,2] # local orientation
    orientation4dofitem3::Array{Int,1} = [1,2] # local orientation
    coefficient4dofitem::Array{Array{T,1},1} = [[0.0,0.0],[0.0,0.0],[0.0,0.0]] # coefficients for operators
    ndofs4item::Array{Int, 1} = [0,0,0]
    dofitem::Array{Int,1} = [0,0,0]
    offsets = [0, size(basisevaler[1,1,1,1].cvals,1), size(basisevaler[1,1,1,1].cvals,1) + size(basisevaler[1,2,1,1].cvals,1)]
    action_input::Array{T,1} = zeros(T,action.argsizes[2]) # heap for action input
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations
    basisevaler4dofitem::Array{FEBasisEvaluator,1} = [basisevaler[1,1,1,1], basisevaler[1,2,1,1], basisevaler[1,3,1,1]]
    basisvals_testfunction::Array{T,3} = basisevaler4dofitem[3].cvals
    evalfixedFE::Array{T,1} = zeros(T,size(basisevaler[1,fixed_argument,1,1].cvals,1)) # evaluation of argument 1
    temp::T = 0 # some temporary variable

    nonfixed_ids = setdiff([1,2,3], fixed_argument)
    coeffs::Array{T,1} = zeros(T,max_num_targets_per_source(xItemDofs[fixed_argument]))
    dofs2::Array{Int,1} = zeros(Int,max_num_targets_per_source(xItemDofs[nonfixed_ids[1]]))
    dofs3::Array{Int,1} = zeros(Int,max_num_targets_per_source(xItemDofs[nonfixed_ids[2]]))
    localmatrix::Array{T,2} = zeros(T,length(dofs2),length(dofs3))

    regions::Array{Int,1} = AP.regions
    allitems::Bool = (regions == [0])
    nregions::Int = length(regions)
    for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if allitems || xItemRegions[item] == regions[r]

        # get dofitem informations
        EG4item = dii4op[1](dofitems1, EG4dofitem1, itempos4dofitem1, coefficient4dofitem[1], orientation4dofitem1, item)
        dii4op[2](dofitems2, EG4dofitem2, itempos4dofitem2, coefficient4dofitem[2], orientation4dofitem2, item)
        dii4op[3](dofitems3, EG4dofitem3, itempos4dofitem3, coefficient4dofitem[3], orientation4dofitem3, item)

        # get quadrature weights for integration domain
        weights = qf[EG4item].w

        # loop over associated dofitems (maximal 2 for jump calculations)
        for di = 1 : 2, dj = 1 : 2, dk = 1 : 2
            dofitem[1] = dofitems1[di]
            dofitem[2] = dofitems2[dj]
            dofitem[3] = dofitems3[dk]

            if dofitem[1] > 0 && dofitem[2] > 0 && dofitem[3] > 0

                # get number of dofs on this dofitem
                ndofs4item[1] = ndofs4EG[1][EG4dofitem1[di]]
                ndofs4item[2] = ndofs4EG[2][EG4dofitem2[dj]]
                ndofs4item[3] = ndofs4EG[3][EG4dofitem3[dk]]

                # update FEbasisevaler
                basisevaler4dofitem[1] = basisevaler[EG4dofitem1[di],1,itempos4dofitem1[di],orientation4dofitem1[di]]
                basisevaler4dofitem[2] = basisevaler[EG4dofitem2[dj],2,itempos4dofitem2[dj],orientation4dofitem2[dj]]
                basisevaler4dofitem[3] = basisevaler[EG4dofitem3[dk],3,itempos4dofitem3[dk],orientation4dofitem3[dk]]
                update!(basisevaler4dofitem[1],dofitem[1])
                update!(basisevaler4dofitem[2],dofitem[2])
                update!(basisevaler4dofitem[3],dofitem[3])

                # update action on dofitem
                update!(action, basisevaler4dofitem[2], dofitem[2], item, regions[r])

                # update coeffs of fixed argument
                for j=1:ndofs4item[fixed_argument]
                    coeffs[j] = FE1[xItemDofs[fixed_argument][j,dofitem[fixed_argument]]]
                end
                # update dofs of free arguments
                for j=1:ndofs4item[nonfixed_ids[1]]
                    dofs2[j] = xItemDofs[nonfixed_ids[1]][j,dofitem[nonfixed_ids[1]]]
                end
                for j=1:ndofs4item[nonfixed_ids[2]]
                    dofs3[j] = xItemDofs[nonfixed_ids[2]][j,dofitem[nonfixed_ids[2]]]
                end

                if fixed_argument in [1,2]
    
                    basisvals_testfunction = basisevaler4dofitem[nonfixed_ids[2]].cvals
                    for i in eachindex(weights)
    
                        # evaluate fixed argument into action
                        fill!(action_input, 0.0)
                        eval!(action_input, basisevaler4dofitem[fixed_argument], coeffs, i; offset = offsets[fixed_argument], factor = coefficient4dofitem[fixed_argument][di])
                        
                        for dof_i = 1 : ndofs4item[nonfixed_ids[1]]
                            # apply action to fixed argument and first non-fixed argument
                            eval!(action_input, basisevaler4dofitem[nonfixed_ids[1]], dof_i, i, offset = offsets[nonfixed_ids[1]], factor = coefficient4dofitem[nonfixed_ids[1]][dj])
                            
                            apply_action!(action_result, action_input, action, i)
            
                            for dof_j = 1 : ndofs4item[nonfixed_ids[2]]
                                temp = 0
                                for k = 1 : action_resultdim
                                    temp += action_result[k] * basisvals_testfunction[k,dof_j,i]
                                end
                                localmatrix[dof_i,dof_j] += temp * weights[i] * coefficient4dofitem[nonfixed_ids[2]][dk]
                            end
                        end 
                    end
                else # fixed argument is the last one
                    for i in eachindex(weights)
    
                        # evaluate fixed argument into separate vector
                        fill!(evalfixedFE, 0.0)
                        eval!(evalfixedFE, basisevaler4dofitem[fixed_argument], coeffs, i; factor = coefficient4dofitem[fixed_argument][di])
                        
                        for dof_i = 1 : ndofs4item[nonfixed_ids[1]]
                            # apply action to fixed argument and first non-fixed argument
                            eval!(action_input, basisevaler4dofitem[nonfixed_ids[1]], dof_i, i; factor = coefficient4dofitem[nonfixed_ids[1]][di])
                            
                            for dof_j = 1 : ndofs4item[nonfixed_ids[2]]
                                eval!(action_input, basisevaler4dofitem[nonfixed_ids[2]], dof_j, i; offset = offsets[2], factor = coefficient4dofitem[nonfixed_ids[2]][dj])
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
                for dof_i = 1 : ndofs4item[nonfixed_ids[1]], dof_j = 1 : ndofs4item[nonfixed_ids[2]]
                    if localmatrix[dof_i,dof_j] != 0
                        if transposed_assembly == true
                            _addnz(A,dofs3[dof_j],dofs2[dof_i],localmatrix[dof_i,dof_j] * xItemVolumes[item], factor)
                        else
                            _addnz(A,dofs2[dof_i],dofs3[dof_j],localmatrix[dof_i,dof_j] * xItemVolumes[item], factor)
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
    verbosity::Int = 0,
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
    verbosity::Int = 0,
    skip_preps::Bool = false,
    factor::Real = 1,
    offset::Int = 0) where {APT <: APT_TrilinearForm, T <: Real, AT <: AbstractAssemblyType}

    # get adjacencies
    FE = AP.FES
    @assert FE[1] == FE1.FES
    @assert FE[2] == FE2.FES
    xItemVolumes::Array{T,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemDofs1::Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}} = Dofmap4AssemblyType(FE[1], DofitemAT4Operator(AT, AP.operators[1]))
    xItemDofs2::Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}} = Dofmap4AssemblyType(FE[2], DofitemAT4Operator(AT, AP.operators[2]))
    xItemDofs3::Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}} = Dofmap4AssemblyType(FE[3], DofitemAT4Operator(AT, AP.operators[3]))
    xItemRegions::Array{Int,1} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = length(xItemVolumes)

    # prepare assembly
    action = AP.action
    if !skip_preps
        prepare_assembly!(AP; verbosity = verbosity - 1)
    end
    EG = AP.APP.EG
    ndofs4EG::Array{Array{Int,1},1} = AP.APP.ndofs4EG
    qf::Array{QuadratureRule,1} = AP.APP.qf
    basisevaler::Array{FEBasisEvaluator,4} = AP.APP.basisevaler
    dii4op::Array{Function,1} = AP.APP.dii4op

    # get size informations
    ncomponents::Int = get_ncomponents(eltype(FE[1]))
    ncomponents2::Int = get_ncomponents(eltype(FE[2]))
    cvals_resultdim::Int = size(basisevaler[1,1,1,1].cvals,1)
    cvals_resultdim2::Int = size(basisevaler[1,2,1,1].cvals,1)
    action_resultdim::Int = action.argsizes[1]

    if verbosity > 0
        println("  Assembling ($APT,$AT,$T) into vector with fixed_arguments = [1,2]")
        println("   skip_preps = $skip_preps")
        println("    operators = $(AP.operators)")
        println("      regions = $(AP.regions)")
        println("       factor = $factor")
        println("       action = $(AP.action.name) (apply_to = [1,2], size = $(action.argsizes))")
        println("        qf[1] = $(qf[1].name) ")
        println("           EG = $EG")
    end

    # loop over items
    EG4item::Int = 1
    EG4dofitem1::Array{Int,1} = [1,1] # EG id of the current item with respect to operator 1
    EG4dofitem2::Array{Int,1} = [1,1] # EG id of the current item with respect to operator 2
    EG4dofitem3::Array{Int,1} = [1,1] # EG id of the current item with respect to operator 3
    dofitems1::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found (operator 1)
    dofitems2::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found (operator 2)
    dofitems3::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found (operator 3)
    itempos4dofitem1::Array{Int,1} = [1,1] # local item position in dofitem1
    itempos4dofitem2::Array{Int,1} = [1,1] # local item position in dofitem2
    itempos4dofitem3::Array{Int,1} = [1,1] # local item position in dofitem3
    orientation4dofitem1::Array{Int,1} = [1,2] # local orientation
    orientation4dofitem2::Array{Int,1} = [1,2] # local orientation
    orientation4dofitem3::Array{Int,1} = [1,2] # local orientation
    coefficient4dofitem1::Array{T,1} = [0.0,0.0] # coefficients for operator 1
    coefficient4dofitem2::Array{T,1} = [0.0,0.0] # coefficients for operator 2
    coefficient4dofitem3::Array{T,1} = [0.0,0.0] # coefficients for operator 3
    ndofs4item1::Int = 0 # number of dofs for item
    ndofs4item2::Int = 0 # number of dofs for item
    ndofs4item3::Int = 0 # number of dofs for item
    dofitem1::Int = 0
    dofitem2::Int = 0
    dofitem3::Int = 0
    maxdofs1::Int = max_num_targets_per_source(xItemDofs1)
    maxdofs2::Int = max_num_targets_per_source(xItemDofs2)
    maxdofs3::Int = max_num_targets_per_source(xItemDofs3)
    coeffs1::Array{T,1} = zeros(T,maxdofs1)
    coeffs2::Array{T,1} = zeros(T,maxdofs2)
    dofs3::Array{Int,1} = zeros(Int,maxdofs3)
    action_input::Array{T,1} = zeros(T,action.argsizes[2]) # heap for action input
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations
    basisevaler4dofitem1::FEBasisEvaluator = basisevaler[1,1,1,1]
    basisevaler4dofitem2::FEBasisEvaluator = basisevaler[1,2,1,1]
    basisevaler4dofitem3::FEBasisEvaluator = basisevaler[1,3,1,1]
    basisvals3::Array{T,3} = basisevaler4dofitem3.cvals
    temp::T = 0 # some temporary variable
    localb::Array{T,1} = zeros(T,maxdofs3)
    bdof::Int = 0

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
        dii4op[3](dofitems3, EG4dofitem3, itempos4dofitem3, coefficient4dofitem3, orientation4dofitem3, item)

        # get quadrature weights for integration domain
        weights = qf[EG4item].w

        # loop over associated dofitems (maximal 2 for jump calculations)
        # di, dj == 2 is only performed if one of the operators jumps
        for di = 1 : 2, dj = 1 : 2, dk = 1 : 2
            dofitem1 = dofitems1[di]
            dofitem2 = dofitems2[dj]
            dofitem3 = dofitems3[dk]

            if dofitem1 > 0 && dofitem2 > 0 && dofitem3 > 0

               # println("di/dj/dk = $di/$dj/$dk")

                # get number of dofs on this dofitem
                ndofs4item1 = ndofs4EG[1][EG4dofitem1[di]]
                ndofs4item2 = ndofs4EG[2][EG4dofitem2[dj]]
                ndofs4item3 = ndofs4EG[3][EG4dofitem3[dk]]

                # update FEbasisevaler
                basisevaler4dofitem1 = basisevaler[EG4dofitem1[di],1,itempos4dofitem1[di],orientation4dofitem1[di]]
                basisevaler4dofitem2 = basisevaler[EG4dofitem2[dj],2,itempos4dofitem2[dj],orientation4dofitem2[dj]]
                basisevaler4dofitem3 = basisevaler[EG4dofitem3[dk],3,itempos4dofitem3[dk],orientation4dofitem3[dk]]
                basisvals3 = basisevaler4dofitem3.cvals
                update!(basisevaler4dofitem1,dofitem1)
                update!(basisevaler4dofitem2,dofitem2)
                update!(basisevaler4dofitem3,dofitem3)

                # update action on dofitem
                update!(action, basisevaler4dofitem2, dofitem2, item, regions[r])

                # update coeffs, dofs
                for j=1:ndofs4item1
                    bdof = xItemDofs1[j,dofitem1]
                    coeffs1[j] = FE1[bdof]
                end
                for j=1:ndofs4item2
                    bdof = xItemDofs2[j,dofitem2]
                    coeffs2[j] = FE2[bdof]
                end
                for j=1:ndofs4item3
                    dofs3[j] = xItemDofs3[j,dofitem3]
                end

                for i in eachindex(weights)

                    # evaluate first and second component
                    fill!(action_input, 0.0)
                    eval!(action_input, basisevaler4dofitem1, coeffs1, i; factor = coefficient4dofitem1[di])
                    eval!(action_input, basisevaler4dofitem2, coeffs2, i; offset = cvals_resultdim, factor = coefficient4dofitem2[dj])
        
                    # apply action to FE1 and FE2
                    apply_action!(action_result, action_input, action, i)
                   
                    # multiply third component
                    for dof_j = 1 : ndofs4item3
                        temp = 0
                        for k = 1 : action_resultdim
                            temp += action_result[k] * basisvals3[k,dof_j,i]
                        end
                        localb[dof_j] += temp * weights[i]
                    end 
                end 
        
                for dof_i = 1 : ndofs4item3
                    bdof = dofs3[dof_i] + offset
                    b[bdof] += localb[dof_i] * xItemVolumes[item] * factor * coefficient4dofitem3[dk]
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
    verbosity::Int = 0,
    skip_preps::Bool = false,
    factor::Real = 1) where {APT <: APT_TrilinearForm, T <: Real, AT <: AbstractAssemblyType}

    assemble!(b.entries, FE1, FE2, AP; verbosity = verbosity, factor = factor, offset = b.offset, skip_preps = skip_preps)
end
