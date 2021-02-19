abstract type APT_NonlinearForm <: AssemblyPatternType end # nonlinear form whose action also gets a current solution as input to evaluate some linearised form

"""
````
function NonlinearForm(
    T::Type{<:Real},
    FES::Array{FESpace,1},          # finite element spaces for each operator of the ansatz function and the last one refers to the test function
    operators::Array{DataType,1},   # operators that should be evaluated for the ansatz function and the last one refers to the test function
    action::AbstractAction;         # action that shoul have an AbstractNLActionKernel
    regions::Array{Int,1} = [0])
````

Creates a NonlinearForm assembly pattern.
"""
function NonlinearForm(
    T::Type{<:Real},
    AT::Type{<:AbstractAssemblyType},
    FES::Array{FESpace,1}, 
    operators::Array{DataType,1},  
    action::AbstractAction;
    regions::Array{Int,1} = [0])

    return AssemblyPattern{APT_NonlinearForm, T, AT}(FES,operators,action,regions,AssemblyPatternPreparations(nothing,nothing,nothing,nothing,nothing))
end



"""
````
assemble!(
    A::AbstractArray{T,2},
    AP::AssemblyPattern{APT,T,AT};
    FEB::Array{<:FEVectorBlock,1};         # coefficients for each operator
    verbosity::Int = 0,
    factor = 1,
    transposed_assembly::Bool = false
````

Assembly of a NonlinearForm assembly pattern into given two-dimensional AbstractArray (e.g. FEMatrixBlock).
"""
function assemble!(
    A::AbstractArray{T,2},
    AP::AssemblyPattern{APT,T,AT},
    FEB::Array{<:FEVectorBlock,1};
    verbosity::Int = 0,
    factor = 1,
    transposed_assembly::Bool = false,
    skip_preps::Bool = false,
    offsetX = 0,
    offsetY = 0) where {APT <: APT_NonlinearForm, T <: Real, AT <: AbstractAssemblyType}

    # extract finite element spaces and operators
    FE = AP.FES
    nFE::Int = length(FE)
    
    # get adjacencies
    xItemVolumes::Array{T,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemDofs = Array{Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}},1}(undef,length(FE))
    for j = 1 : nFE
        if j < nFE
            @assert FEB[j].FES == FE[j]
        end
        xItemDofs[j] = Dofmap4AssemblyType(FE[j], DofitemAT4Operator(AT, AP.operators[j]))
    end
    xItemRegions::Union{VectorOfConstants{Int32}, Array{Int32,1}} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = length(xItemVolumes)

    # check if action has a nonlinear action kernel
    action = AP.action
    @assert typeof(action.kernel) <: UserData{AbstractNLActionKernel}

    # prepare assembly
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
    offsets = zeros(Int,length(FE)+1)
    maxdofs::Int = 0
    for j = 1 : nFE
        ncomponents[j] = get_ncomponents(eltype(FE[j]))
        maxdofs = max(maxdofs, max_num_targets_per_source(xItemDofs[j]))
        offsets[j+1] = offsets[j] + size(basisevaler[1,j,1,1].cvals,1)
    end
    action_resultdim::Int = action.argsizes[1]
    maxdofs2::Int = max_num_targets_per_source(xItemDofs[end])

 
    maxnweights::Int = 0
    for j = 1 : length(qf)
        maxnweights = max(maxnweights, length(qf[j].w))
    end
    action_input = Array{Array{T,1},1}(undef,maxnweights)
    for j = 1 : maxnweights
        action_input[j] = zeros(T,offsets[end-1]) # heap for action input
    end
    action_input2 = zeros(T,offsets[end-1])

    if verbosity > 0
        println("  Assembling ($APT,$AT,$T) into matrix (transposed = $transposed_assembly)")
        println("   skip_preps = $skip_preps")
        println("    operators = $(AP.operators)")
        println("      regions = $(AP.regions)")
        println("       factor = $factor")
        println("       action = $(AP.action.name) (apply_to = [operators(FEB),operators(argument 1)] size = $(action.argsizes))")
        println("        qf[1] = $(qf[1].name) ")
        println("           EG = $EG")
    end

    # loop over items
    EG4item::Int = 0
    EG4dofitem1::Array{Int,1} = [1,1] # EG id of the current item with respect to operator
    EG4dofitem2::Array{Int,1} = [1,1] # EG id of the current item with respect to operator
    dofitems1::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found
    dofitems2::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found
    itempos4dofitem1::Array{Int,1} = [1,1] # local item position in dofitem
    itempos4dofitem2::Array{Int,1} = [1,1] # local item position in dofitem
    orientation4dofitem1::Array{Int,1} = [1,2] # local orientation
    orientation4dofitem2::Array{Int,1} = [1,2] # local orientation
    coefficient4dofitem1::Array{T,1} = [0,0] # coefficients for operator
    coefficient4dofitem2::Array{T,1} = [0,0] # coefficients for operator
    ndofs4dofitem::Int = 0 # number of dofs for item
    dofitem::Int = 0
    coeffs::Array{T,1} = zeros(T,maxdofs)
    dofs::Array{Int,1} = zeros(Int,maxdofs)
    dofs2::Array{Int,1} = zeros(Int,maxdofs2)
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations
    basisevaler4dofitem = basisevaler[1,1,1,1]
    basisevaler4dofitem1 = Array{Any,1}(undef,nFE)
    for j = 1 : nFE
        basisevaler4dofitem1[j] = basisevaler[1,j,1,1]
    end
    basisevaler4dofitem2 = basisevaler[1,end,1,1]
    basisvals::Array{T,3} = basisevaler4dofitem.cvals
    temp::T = 0 # some temporary variable
    localmatrix::Array{T,2} = zeros(T,maxdofs,maxdofs2)
    acol::Int = 0
    arow::Int = 0
    fdof::Int = 0

    # note: at the moment we expect that all FE[1:end-1] are the same !
    # otherweise more than one MatrixBlock has to be assembled and we need more offset information
    # hence, this only can handle nonlinearities at the moment that depend on one unknown of the PDEsystem

    regions::Array{Int,1} = AP.regions
    allitems::Bool = (regions == [0])
    nregions::Int = length(regions)
    for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if allitems || xItemRegions[item] == regions[r]

        # get dofitem informations for ansatz function
        EG4item = dii4op[1](dofitems1, EG4dofitem1, itempos4dofitem1, coefficient4dofitem1, orientation4dofitem1, item)

        # get quadrature weights for integration domain
        weights = qf[EG4dofitem1[1]].w

        # fill action input with evluation of current solution
        # given by coefficient vectors
        for FEid = 1 : nFE - 1

            # get information on dofitems
            for di = 1 : length(dofitems1)
                dofitem = dofitems1[di]
                if dofitem != 0
                    # update FEbasisevaler on dofitem
                    basisevaler4dofitem = basisevaler[EG4dofitem1[di],FEid,itempos4dofitem1[di],orientation4dofitem1[di]]
                    update!(basisevaler4dofitem, dofitem)

                    # update coeffs on dofitem
                    ndofs4dofitem = ndofs4EG[FEid][EG4dofitem1[di]]
                    for j=1:ndofs4dofitem
                        fdof = xItemDofs[FEid][j,dofitem]
                        coeffs[j] = FEB[FEid][fdof] * coefficient4dofitem1[di]
                    end

                    for i in eachindex(weights)
                        if FEid == 1 && di == 1
                            fill!(action_input[i], 0)
                        end
                        eval!(action_input[i], basisevaler4dofitem, coeffs, i, offsets[FEid])
                    end  
                end
            end
        end

        # at this point the action_input at each quadrature point contains information on the last solution
        # also no jump operators for the test function are allowed currently

        # get dof information for test function
        dii4op[end](dofitems2, EG4dofitem2, itempos4dofitem2, coefficient4dofitem2, orientation4dofitem2, item)
        di = 1
        dj = 1
        dofitem1 = dofitems1[di]
        dofitem2 = dofitems2[dj]
        ndofs4item1 = ndofs4EG[1][EG4dofitem1[di]]
        ndofs4item2 = ndofs4EG[2][EG4dofitem2[dj]]

        # update FEbasisevalers for ansatz function
        for FEid = 1 : nFE - 1
            basisevaler4dofitem1[FEid] = basisevaler[EG4dofitem1[di],FEid,itempos4dofitem1[di],orientation4dofitem1[di]]
            update!(basisevaler4dofitem1[FEid],dofitem1)
        end

        # update FEbasisevalers for test function
        basisevaler4dofitem2 = basisevaler[EG4dofitem2[dj],end,itempos4dofitem2[dj],orientation4dofitem2[dj]]
        basisvals = basisevaler4dofitem2.cvals
        update!(basisevaler4dofitem2,dofitem2)

        # update action on dofitem
        update!(action, basisevaler4dofitem1[1], dofitem1, item, regions[r])

        # update dofs
        for j=1:ndofs4item1
            dofs[j] = xItemDofs[1][j,dofitem1]
        end
        for j=1:ndofs4item2
            dofs2[j] = xItemDofs[end][j,dofitem2]
        end

        for i in eachindex(weights)
            for dof_i = 1 : ndofs4item1

                for FEid = 1 : nFE - 1
                    eval!(action_input2, basisevaler4dofitem1[FEid], dof_i, i, offsets[FEid])
                end

                apply_action!(action_result, action_input[i], action_input2, action, i)
                action_result .*= weights[i]

                for dof_j = 1 : ndofs4item2
                    temp = 0
                    for k = 1 : action_resultdim
                        temp += action_result[k] * basisvals[k,dof_j,i]
                    end
                    temp *= coefficient4dofitem2[dj]
                    localmatrix[dof_i,dof_j] += temp
                end
            end 
        end

        localmatrix .*= xItemVolumes[item] * factor

        # copy localmatrix into global matrix
        for dof_i = 1 : ndofs4item1
            arow = dofs[dof_i] + offsetX
            for dof_j = 1 : ndofs4item2
                if localmatrix[dof_i,dof_j] != 0
                    acol = dofs2[dof_j] + offsetY
                    if transposed_assembly == true
                        _addnz(A,acol,arow,localmatrix[dof_i,dof_j],1)
                    else 
                        _addnz(A,arow,acol,localmatrix[dof_i,dof_j],1)  
                    end
                end
            end
        end
        
        fill!(localmatrix,0.0)
        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop

    return nothing
end


## wrapper for FEMatrixBlock to avoid use of setindex! functions of FEMAtrixBlock
function assemble!(
    A::FEMatrixBlock,
    AP::AssemblyPattern{APT,T,AT},
    FEB::Array{<:FEVectorBlock,1};
    verbosity::Int = 0,
    factor = 1,
    transposed_assembly::Bool = false,
    skip_preps::Bool = false) where {APT <: APT_NonlinearForm, T <: Real, AT <: AbstractAssemblyType}

    assemble!(A.entries, AP, FEB; verbosity = verbosity, factor = factor, transposed_assembly = transposed_assembly, offsetX = A.offsetX, offsetY = A.offsetY, skip_preps = skip_preps)
end




"""
````
assemble!(
    b::AbstractVector,
    AP::NonlinearForm{T, AT},
    FEB::Array{<:FEVectorBlock,1};         # coefficients for each operator
    verbosity::Int = 0,
    factor = 1,
    transposed_assembly::Bool = false
````

Assembly of a NonlinearForm AP into given AbstractVector (e.g. FEMatrixBlock).
"""
function assemble!(
    b::AbstractVector,
    AP::AssemblyPattern{APT,T,AT},
    FEB::Array{<:FEVectorBlock,1};
    verbosity::Int = 0,
    factor = 1,
    transposed_assembly::Bool = false,
    skip_preps::Bool = false,
    offset = 0) where {APT <: APT_NonlinearForm, T <: Real, AT <: AbstractAssemblyType}

    # extract finite element spaces and operators
    FE = AP.FES
    nFE::Int = length(FE)
    
    # get adjacencies
    xItemVolumes::Array{T,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemDofs = Array{Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}},1}(undef,length(FE))
    for j = 1 : nFE
        xItemDofs[j] = Dofmap4AssemblyType(FE[j], DofitemAT4Operator(AT, AP.operators[j]))
    end
    xItemRegions::Union{VectorOfConstants{Int32}, Array{Int32,1}} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems::Int = length(xItemVolumes)

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
    offsets = zeros(Int,length(FE)+1)
    maxdofs::Int = 0
    for j = 1 : nFE
        ncomponents[j] = get_ncomponents(eltype(FE[j]))
        maxdofs = max(maxdofs, max_num_targets_per_source(xItemDofs[j]))
        offsets[j+1] = offsets[j] + size(basisevaler[1,j,1,1].cvals,1)
    end
    action_resultdim::Int = action.argsizes[1]
    maxdofs2::Int = max_num_targets_per_source(xItemDofs[end])

 
    maxnweights::Int = 0
    for j = 1 : length(qf)
        maxnweights = max(maxnweights, length(qf[j].w))
    end
    action_input = Array{Array{T,1},1}(undef,maxnweights)
    for j = 1 : maxnweights
        action_input[j] = zeros(T,offsets[end-1]) # heap for action input
    end

    if verbosity > 0
        println("  Assembling ($APT,$AT,$T) into vector (action evaluated with given coefficients)")
        println("   skip_preps = $skip_preps")
        println("    operators = $(AP.operators)")
        println("      regions = $(AP.regions)")
        println("       factor = $factor")
        println("       action = $(AP.action.name) (size = $(action.argsizes))")
        println("        qf[1] = $(qf[1].name) ")
        println("           EG = $EG")
    end

    # loop over items
    EG4item::Int = 0
    EG4dofitem::Array{Int,1} = [1,1] # EG id of the current item with respect to operator
    dofitems::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found
    itempos4dofitem::Array{Int,1} = [1,1] # local item position in dofitem
    coefficient4dofitem::Array{T,1} = [0,0] # coefficients for operator
    orientation4dofitem::Array{Int,1} = [1,1] # local item position in dofitem
    ndofs4dofitem::Int = 0 # number of dofs for item
    dofitem::Int = 0
    coeffs::Array{T,1} = zeros(T,maxdofs)
    dofs::Array{Int,1} = zeros(Int,maxdofs)
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations
    basisevaler4dofitem = basisevaler[1,1,1,1]
    basisvals::Array{T,3} = basisevaler4dofitem.cvals
    temp::T = 0 # some temporary variable
    localb = zeros(T,maxdofs)
    bdof::Int = 0


    # note: at the moment we expect that all FE[1:end-1] are the same !
    # otherweise more than one MatrixBlock has to be assembled and we need more offset information
    # hence, this only can handle nonlinearities at the moment that depend on one unknown of the PDEsystem

    regions::Array{Int,1} = AP.regions
    allitems::Bool = (regions == [0])
    nregions::Int = length(regions)
    for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if allitems || xItemRegions[item] == regions[r]

        # get dofitem informations for ansatz function
        EG4item = dii4op[1](dofitems, EG4dofitem, itempos4dofitem, coefficient4dofitem, orientation4dofitem, item)

        # get quadrature weights for integration domain
        weights = qf[EG4dofitem[1]].w

        # fill action input with evluation of current solution
        # given by coefficient vectors
        for FEid = 1 : nFE - 1

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
                        bdof = xItemDofs[FEid][j,dofitem]
                        coeffs[j] = FEB[FEid][bdof] * coefficient4dofitem[di]
                    end

                    if FEid == 1 && di == 1
                        for i in eachindex(weights)
                            fill!(action_input[i], 0)
                        end
                    end
                    for i in eachindex(weights)
                        eval!(action_input[i], basisevaler4dofitem, coeffs, i, offsets[FEid])
                    end  
                end
            end
        end

        # at this point the action_input at each quadrature point contains information on the last solution
        # also no jump operators for the test function are allowed currently
        di = 1
        dofitem = dofitems[di]
        ndofs4item = ndofs4EG[1][EG4dofitem[di]]

        # update FEbasisevalers for test function
        basisevaler4dofitem = basisevaler[EG4dofitem[di],end,itempos4dofitem[di],di]
        basisvals = basisevaler4dofitem.cvals
        update!(basisevaler4dofitem,dofitem)

        # update action on dofitem
        update!(action, basisevaler4dofitem, dofitem, item, regions[r])

        # update dofs
        for j=1:ndofs4item
            dofs[j] = xItemDofs[1][j,dofitem]
        end

        for i in eachindex(weights)
            apply_action!(action_result, action_input[i], action, i)
            action_result .*= weights[i]

            for dof_j = 1 : ndofs4item
                temp = 0
                for k = 1 : action_resultdim
                    temp += action_result[k] * basisvals[k,dof_j,i]
                end
                temp *= coefficient4dofitem[di]
                localb[dof_j] += temp
            end
        end

        localb .*= xItemVolumes[item] * factor

        # copy localmatrix into global matrix
        for dof_i = 1 : ndofs4item
            bdof = dofs[dof_i] + offset
            b[bdof] += localb[dof_i]          
        end
        
        fill!(localb,0.0)
        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop

    return nothing
end


## wrapper for FEVectorBlock to avoid use of setindex! functions of FEMAtrixBlock
function assemble!(
    b::FEVectorBlock,
    AP::AssemblyPattern{APT,T,AT},
    FEB::Array{<:FEVectorBlock,1};
    verbosity::Int = 0,
    factor = 1,
    transposed_assembly::Bool = false,
    skip_preps::Bool = false) where {APT <: APT_NonlinearForm, T <: Real, AT <: AbstractAssemblyType}

    assemble!(b.entries, AP, FEB; verbosity = verbosity, factor = factor, transposed_assembly = transposed_assembly, offset = b.offset, skip_preps = skip_preps)
end






