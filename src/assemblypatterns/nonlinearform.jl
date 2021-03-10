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

    return AssemblyPattern{APT_NonlinearForm, T, AT}(FES,operators,action,regions)
end



"""
````
assemble!(
    A::AbstractArray{T,2},
    AP::AssemblyPattern{APT,T,AT};
    FEB::Array{<:FEVectorBlock,1};         # coefficients of current solution for each operator
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

    # prepare assembly
    FE = AP.FES
    nFE = length(FE)
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
    maxnweights = get_maxnqweights(AM)
    action_input = Array{Array{T,1},1}(undef,maxnweights)
    for j = 1 : maxnweights
        action_input[j] = zeros(T,action.argsizes[3]) # heap for action input
    end
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output

    if verbosity > 0
        println("  Assembling ($APT,$AT,$T) into matrix (transposed = $transposed_assembly)")
        println("   skip_preps = $skip_preps")
        println("    operators = $(AP.operators)")
        println("      regions = $(AP.regions)")
        println("       factor = $factor")
        println("       action = $(AP.action.name) (apply_to = [operators(FEB),operators(argument 1)] size = $(action.argsizes))")
        println("        qf[1] = $(AM.qf[1].name) ")
        
    end

    # loop over items
    offsets = zeros(Int,nFE+1)
    maxdofs = get_maxndofs(AM)
    basisevaler::FEBasisEvaluator = get_basisevaler(AM, 1, 1)
    basisevaler2::FEBasisEvaluator = get_basisevaler(AM, nFE, 1)
    basisvals::Array{T,3} = basisevaler.cvals
    for j = 1 : nFE
        basisevaler = get_basisevaler(AM, j, 1)
        offsets[j+1] = offsets[j] + size(basisevaler.cvals,1)
    end
    action_input2 = zeros(T,offsets[end-1])
    maxdofitems::Array{Int,1} = get_maxdofitems(AM)
    localmatrix::Array{T,2} = zeros(T,get_maxndofs(AM,1),get_maxndofs(AM,nFE))
    coeffs = zeros(T,maximum(maxdofs))
    weights::Array{T,1} = get_qweights(AM)
    itemfactor::T = 0
    arow::Int = 0
    acol::Int = 0
    regions::Array{Int,1} = AP.regions
    allitems::Bool = (regions == [0])
    nregions::Int = length(regions)

    # note: at the moment we expect that all FE[1:end-1] are the same !
    # otherwise more than one MatrixBlock has to be assembled and we need more offset information
    # hence, this only can handle nonlinearities at the moment that depend on one unknown of the PDEsystem

    for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if allitems || xItemRegions[item] == regions[r]

        # update assembly manager (also updates necessary basisevaler)
        update!(AP.AM, item)
        weights = get_qweights(AM)

        # fill action input with evluation of current solution
        # assemble all but the last operators into action_input
        for FEid = 1 : nFE - 1
            for di = 1 : maxdofitems[FEid]
                if AM.dofitems[FEid][di] != 0
                    # get correct basis evaluator for dofitem (was already updated by AM)
                    basisevaler = get_basisevaler(AM, FEid, di)
    
                    # update action on dofitem
                    update!(action, basisevaler, AM.dofitems[FEid][di], item, regions[r])

                    # get coefficients of FE number FEid on current dofitem
                    get_coeffs!(coeffs, FEB[FEid], AM, FEid, di)
                    coeffs .*= AM.coeff4dofitem[FEid][di]

                    # write evaluation of operator of current FE into action_input
                    for i in eachindex(weights)
                        eval!(action_input[i], basisevaler, coeffs, i, offsets[FEid])
                    end  
                end
            end
        end

        # update action on dofitem
        basisevaler2 = get_basisevaler(AM, nFE, 1)
        basisvals = basisevaler2.cvals
        update!(action, basisevaler, item, item, regions[r])

        for i in eachindex(weights)
            for dof_i = 1 : get_ndofs(AM, 1, 1)

                # evaluate operators of ansatz function
                for FEid = 1 : nFE - 1
                    basisevaler = get_basisevaler(AM, FEid, 1)
                    eval!(action_input2, basisevaler, dof_i, i, offsets[FEid])
                end

                # apply nonlinear action
                apply_action!(action_result, action_input[i], action_input2, action, i)  
                action_result .*= weights[i]

                # multiply test function operator evaluation
                for dof_j = 1 : get_ndofs(AM, nFE, 1)
                    temp = 0
                    for k = 1 : action_resultdim
                        temp += action_result[k] * basisvals[k,dof_j,i]
                    end
                    localmatrix[dof_i,dof_j] += temp
                end
            end 
            fill!(action_input[i],0)
        end

        itemfactor = xItemVolumes[item] * factor * AM.coeff4dofitem[nFE][1]

        # copy localmatrix into global matrix
        for dof_i = 1 : get_ndofs(AM, 1, 1)
            arow = get_dof(AM, 1, 1, dof_i) + offsetX
            for dof_j = 1 : get_ndofs(AM, nFE, 1)
                if localmatrix[dof_i,dof_j] != 0
                    acol = get_dof(AM, nFE, 1, dof_j) + offsetY
                    if transposed_assembly == true
                        _addnz(A,acol,arow,localmatrix[dof_i,dof_j],itemfactor)
                    else 
                        _addnz(A,arow,acol,localmatrix[dof_i,dof_j],itemfactor)  
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
    FEB::Array{<:FEVectorBlock,1};         # coefficients of current solution for each operator
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

    # prepare assembly
    FE = AP.FES
    nFE = length(FE)
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
    maxnweights = get_maxnqweights(AM)
    action_input = Array{Array{T,1},1}(undef,maxnweights)
    for j = 1 : maxnweights
        action_input[j] = zeros(T,action.argsizes[3]) # heap for action input
    end
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output

    if verbosity > 0
        println("  Assembling ($APT,$AT,$T) into vector (action evaluated with given coefficients)")
        println("   skip_preps = $skip_preps")
        println("    operators = $(AP.operators)")
        println("      regions = $(AP.regions)")
        println("       factor = $factor")
        println("       action = $(AP.action.name) (size = $(action.argsizes))")
        println("        qf[1] = $(AM.qf[1].name) ")
        
    end

    # loop over items
    offsets = zeros(Int,nFE+1)
    maxdofs = get_maxndofs(AM)
    basisevaler::FEBasisEvaluator = get_basisevaler(AM, 1, 1)
    basisvals::Array{T,3} = basisevaler.cvals
    for j = 1 : nFE
        basisevaler = get_basisevaler(AM, j, 1)
        offsets[j+1] = offsets[j] + size(basisevaler.cvals,1)
    end
    maxdofitems::Array{Int,1} = get_maxdofitems(AM)
    localb::Array{T,1} = zeros(T,get_maxndofs(AM,nFE))
    coeffs = zeros(T,maximum(maxdofs))
    weights::Array{T,1} = get_qweights(AM)
    regions::Array{Int,1} = AP.regions
    allitems::Bool = (regions == [0])
    nregions::Int = length(regions)

    # note: at the moment we expect that all FE[1:end-1] are the same !
    # otherwise more than one MatrixBlock has to be assembled and we need more offset information
    # hence, this only can handle nonlinearities at the moment that depend on one unknown of the PDEsystem

    for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if allitems || xItemRegions[item] == regions[r]

        # update assembly manager (also updates necessary basisevaler)
        update!(AP.AM, item)
        weights = get_qweights(AM)

        # fill action input with evluation of current solution
        # assemble all but the last operators into action_input
        for FEid = 1 : nFE - 1
            for di = 1 : maxdofitems[FEid]
                if AM.dofitems[FEid][di] != 0
                    # get correct basis evaluator for dofitem (was already updated by AM)
                    basisevaler = get_basisevaler(AM, FEid, di)
    
                    # update action on dofitem
                    update!(action, basisevaler, AM.dofitems[FEid][di], item, regions[r])

                    # get coefficients of FE number FEid on current dofitem
                    get_coeffs!(coeffs, FEB[FEid], AM, FEid, di)
                    coeffs .*= AM.coeff4dofitem[FEid][di]

                    # write evaluation of operator of current FE into action_input
                    for i in eachindex(weights)
                        eval!(action_input[i], basisevaler, coeffs, i, offsets[FEid])
                    end  
                end
            end
        end

        # update action on dofitem
        basisevaler = get_basisevaler(AM, nFE, 1)
        basisvals = basisevaler.cvals
        update!(action, basisevaler, item, item, regions[r])

        for i in eachindex(weights)
            apply_action!(action_result, action_input[i], action, i)
            action_result .*= weights[i]

            for dof_j = 1 : get_ndofs(AM, nFE, 1)
                temp = 0
                for k = 1 : action_resultdim
                    temp += action_result[k] * basisvals[k,dof_j,i]
                end
                localb[dof_j] += temp
            end
            fill!(action_input[i],0)
        end

        localb .*= xItemVolumes[item] * factor * AM.coeff4dofitem[nFE][1]

        # copy localmatrix into global matrix
        for dof_i = 1 : get_ndofs(AM, nFE, 1)
            b[get_dof(AM, nFE, 1, dof_i) + offset] += localb[dof_i]          
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