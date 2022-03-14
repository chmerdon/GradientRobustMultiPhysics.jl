
"""
$(TYPEDEF)

nonlinearform assembly pattern type
"""
abstract type APT_NonlinearForm <: AssemblyPatternType end # nonlinear form whose action also gets a current solution as input to evaluate some linearised form

function Base.show(io::IO, ::Type{APT_NonlinearForm})
    print(io, "NonlinearForm")
end

function DiscreteNonlinearForm(
    T::Type{<:Real},
    AT::Type{<:AssemblyType},
    FES::Array{FESpace,1}, 
    operators::Array{DataType,1},
    action::AbstractAction;
    newton_args::Array{Int,1} = 1 : (length(operators)-1),
    name = "NLF",
    regions::Array{Int,1} = [0])

    AP = AssemblyPattern{APT_NonlinearForm, T, AT}(name,FES,operators,action,1:(length(operators)-1),regions)
    AP.newton_args = newton_args
    return AP
end



"""
````
full_assemble!(
    A::AbstractArray{T,2},                 # target matrix
    b::AbstractArray{T,1},                 # target rhs
    AP::AssemblyPattern{APT,T,AT};         # NonlinearForm pattern
    FEB::Array{<:FEVectorBlock,1};         # coefficients of current solution for each operator
    factor = 1,                            # factor that is multiplied
    transposed_assembly::Bool = false)     # transpose result?
    where {APT <: APT_NonlinearForm, T, AT}
````

Assembly (of Newton terms) of a NonlinearForm assembly pattern (assembles both matrix and rhs!).
"""
function full_assemble!(
    A::AbstractArray{T,2},
    b::AbstractArray{T,1},
    AP::AssemblyPattern{APT,T,AT},
    FEB::Array{<:FEVectorBlock{T,Tv,Ti},1};
    factor = 1,
    transposed_assembly::Bool = false,
    skip_preps::Bool = false,
    offsetX = 0,
    offsetY = 0) where {APT <: APT_NonlinearForm, T <: Real, AT <: AssemblyType, Tv, Ti}

    # prepare assembly
    FE = AP.FES
    nFE = length(FE)
    if !skip_preps
        prepare_assembly!(AP)
    end
    AM::AssemblyManager{T} = AP.AM
    FEAT = EffAT4AssemblyType(assemblytype(FE[1]),AT)
    xItemVolumes::Array{T,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(FEAT)]
    xItemRegions::GridRegionTypes{Ti} = FE[1].xgrid[GridComponentRegions4AssemblyType(FEAT)]
    nitems = length(xItemVolumes)

    # prepare action
    jac_handler = AP.action
    action_resultdim::Int = jac_handler.argsizes[1]
    maxnweights = get_maxnqweights(AM)
    action_input = Array{Array{T,1},1}(undef,maxnweights)
    for j = 1 : maxnweights
        action_input[j] = zeros(T,jac_handler.argsizes[3]) # heap for action input
    end
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output

    if AP.regions != [0]
        @logmsg MoreInfo "Assembling $(AP.name) for current $((p->p.name).(FEB)) into matrix ($AT in regions = $(AP.regions))"
    else
        @logmsg MoreInfo "Assembling $(AP.name) for current $((p->p.name).(FEB)) into matrix ($AT)"
    end
    @debug AP

    # loop over items
    newton_args = AP.newton_args

    offsets = zeros(Int,nFE+1)
    maxdofs = get_maxndofs(AM)
    basisevaler = get_basisevaler(AM, 1, 1)
    basisevaler2 = get_basisevaler(AM, nFE, 1)
    basisvals::Union{SharedCValView{T},Array{T,3}} = basisevaler.cvals
    for j = 1 : nFE
        basisevaler = get_basisevaler(AM, j, 1)
        offsets[j+1] = offsets[j] + size(basisevaler.cvals,1)
    end
    action_input2::Array{T,1} = zeros(T, jac_handler.argsizes[3])
    maxdofitems::Array{Int,1} = get_maxdofitems(AM)
    localb::Array{T,1} = zeros(T,get_maxndofs(AM,nFE))
    localmatrix::Array{T,2} = zeros(T,get_maxndofs(AM,newton_args[1]),get_maxndofs(AM,nFE))
    coeffs = zeros(T,maximum(maxdofs))
    weights::Array{T,1} = get_qweights(AM)
    ndofitems::Int = get_maxdofitems(AM)[nFE]
    itemfactor::T = 0
    arow::Int = 0
    acol::Int = 0
    regions::Array{Int,1} = AP.regions
    allitems::Bool = (regions == [0])
    nregions::Int = length(regions)

    # note: at the moment we expect that all FE[1:end-1] are the same !
    # otherwise more than one MatrixBlock has to be assembled and we need more offset information
    # hence, this only can handle nonlinearities at the moment that depend on one unknown of the PDEsystem

    loop_allocations = @allocated for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if allitems || xItemRegions[item] == regions[r]

        # update assembly manager (also updates necessary basisevaler)
        update_assembly!(AM, item)
        weights = get_qweights(AM)

        # fill action input with evaluation of current solution
        # assemble all but the last operators into action_input
        for FEid = 1 : nFE - 1
            for di = 1 : maxdofitems[FEid]
                if AM.dofitems[FEid][di] != 0
                    # get correct basis evaluator for dofitem (was already updated by AM)
                    basisevaler = get_basisevaler(AM, FEid, di)

                    # get coefficients of FE number FEid on current dofitem
                    FEB[FEid][AM.xItemDofs[FEid][1 + AM.dofoffset4dofitem[FEid][di], AM.dofitems[FEid][di]]]
                    get_coeffs!(coeffs, FEB[FEid], AM, FEid, di)
                    coeffs .*= AM.coeff4dofitem[FEid][di]

                    # write evaluation of operator of current FE into action_input
                    for i in eachindex(weights)
                        eval_febe!(action_input[i], basisevaler, coeffs, i, offsets[FEid])
                    end  
                end
            end
        end

        for di = 1: ndofitems
            dofitem = AM.dofitems[nFE][di]
            if dofitem > 0
                # update action on dofitem (not needed yet)
                basisevaler2 = get_basisevaler(AM, nFE, di)
                basisvals = basisevaler2.cvals
        
                if is_itemdependent(jac_handler)
                    jac_handler.item[1] = item
                    jac_handler.item[2] = dofitem
                    jac_handler.item[3] = xItemRegions[item]
                    jac_handler.item[4] = di
                end
        
                for i in eachindex(weights)

                    # get local jacobian
                    if is_xdependent(jac_handler)
                        update_trafo!(basisevaler2.L2G, item)
                        eval_trafo!(jac_handler.x,basisevaler.L2G, basisevaler2.xref[i])
                    end
                    eval_jacobian!(jac_handler, action_input[i])
                    jac = jac_handler.jac
                    value = jac_handler.val

                    for dof_i = 1 : get_ndofs(AM, newton_args[1], 1)

                        # evaluate operators of ansatz function (but only of those operators in the current block)
                        # (only of the newton arguments that are associated to the current matrix block)
                        for newarg in newton_args
                            basisevaler = get_basisevaler(AM, newarg, 1)
                            eval_febe!(action_input2, basisevaler, dof_i, i, offsets[newarg])
                        end

                        # multiply with jacobian
                        mul!(action_result,jac,action_input2)

                        # multiply test function operator evaluation
                        for dof_j = 1 : get_ndofs(AM, nFE, 1)
                            temp = 0
                            for k = 1 : action_resultdim
                                temp += action_result[k] * basisvals[k,dof_j,i]
                            end
                            localmatrix[dof_i,dof_j] += temp * weights[i]
                        end
                    end 

                    if 1 in newton_args
                        for dof_j = 1 : get_ndofs(AM, nFE, 1)
                            # multiply with jacobian
                            mul!(action_result,jac,action_input[i])

                            temp = 0
                            for k = 1 : action_resultdim
                                temp += (action_result[k] - value[k]) * basisvals[k,dof_j,i]
                            end
                            localb[dof_j] += temp * weights[i]
                        end
                    end
                end

                itemfactor = xItemVolumes[item] * factor * AM.coeff4dofitem[nFE][1]

                # copy localmatrix into global matrix
                for dof_i = 1 : get_ndofs(AM, newton_args[1], 1)
                    arow = get_dof(AM, newton_args[1], 1, dof_i) + offsetY # offsetY refers to the newton argument offset
                    for dof_j = 1 : get_ndofs(AM, nFE, di)
                        acol = get_dof(AM, nFE, di, dof_j) + offsetX # offsetX refers to the test function offset
                        if transposed_assembly == true
                            _addnz(A,acol,arow,localmatrix[dof_i,dof_j],itemfactor)
                        else 
                            _addnz(A,arow,acol,localmatrix[dof_i,dof_j],itemfactor)  
                        end
                    end
                end
                fill!(localmatrix,0.0)

                if 1 in newton_args
                    localb .*= itemfactor
                    # copy localb into global rhs
                    for dof_i = 1 : get_ndofs(AM, nFE, 1)
                        b[get_dof(AM, nFE, 1, dof_i) + offsetX] += localb[dof_i]      
                    end
                    fill!(localb,0.0)
                end
            end
        end

        for i in eachindex(weights)
            fill!(action_input[i],0)
        end

        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
    
    AP.last_allocations = loop_allocations

    return nothing
end


## wrapper for FEMatrixBlock to avoid use of setindex! functions of FEMAtrixBlock
function full_assemble!(
    A::FEMatrixBlock,
    b::FEVectorBlock,
    AP::AssemblyPattern{APT,T,AT},
    FEB::Array{<:FEVectorBlock{T,Tv,Ti},1};
    factor = 1,
    fixed_arguments = nothing, # ignored
    transposed_assembly::Bool = false,
    skip_preps::Bool = false) where {APT <: APT_NonlinearForm, T <: Real, AT <: AssemblyType, Tv, Ti}

    @assert A.offsetX == b.offset "A and b do not match"

    full_assemble!(A.entries, b.entries, AP, FEB; factor = factor, transposed_assembly = transposed_assembly, offsetX = A.offsetX, offsetY = A.offsetY, skip_preps = skip_preps)
end




#### OLD STUFF BELOW

# """
# ````
# assemble!(
#     A::AbstractArray{T,2},                 # target matrix
#     AP::AssemblyPattern{APT,T,AT};         # NonlinearForm pattern
#     FEB::Array{<:FEVectorBlock,1};         # coefficients of current solution for each operator
#     factor = 1,                            # factor that is multiplied
#     transposed_assembly::Bool = false)     # transpose result?
#     where {APT <: APT_NonlinearForm, T, AT}
# ````

# Assembly of a NonlinearForm assembly pattern into given two-dimensional AbstractArray (e.g. FEMatrixBlock).
# """
# function assemble!(
#     A::AbstractArray{T,2},
#     AP::AssemblyPattern{APT,T,AT},
#     FEB::Array{<:FEVectorBlock{T,Tv,Ti},1};
#     factor = 1,
#     transposed_assembly::Bool = false,
#     skip_preps::Bool = false,
#     offsetX = 0,
#     offsetY = 0) where {APT <: APT_NonlinearForm, T <: Real, AT <: AssemblyType, Tv, Ti}
    

#     # prepare assembly
#     FE = AP.FES
#     nFE = length(FE)
#     if !skip_preps
#         prepare_assembly!(AP)
#     end
#     AM::AssemblyManager{T} = AP.AM
#     xItemVolumes::Array{Tv,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
#     xItemRegions::GridRegionTypes{Ti} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
#     nitems = length(xItemVolumes)

#     # prepare action
#     action = AP.action
#     action_resultdim::Int = action.argsizes[1]
#     maxnweights = get_maxnqweights(AM)
#     action_input = Array{Array{T,1},1}(undef,maxnweights)
#     for j = 1 : maxnweights
#         action_input[j] = zeros(T,action.argsizes[3]) # heap for action input
#     end
#     action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output

#     if AP.regions != [0]
#         @logmsg MoreInfo "Assembling $(AP.name) for current $((p->p.name).(FEB)) into matrix ($AT in regions = $(AP.regions))"
#     else
#         @logmsg MoreInfo "Assembling $(AP.name) for current $((p->p.name).(FEB)) into matrix ($AT)"
#     end
#     @debug AP


#     # loop over items
#     newton_args = AP.newton_args
#     offsets = zeros(Int,nFE+1)
#     maxdofs = get_maxndofs(AM)
#     basisevaler = get_basisevaler(AM, 1, 1)
#     basisevaler2 = get_basisevaler(AM, nFE, 1)
#     basisvals::Union{SharedCValView{T},Array{T,3}} = basisevaler.cvals
#     for j = 1 : nFE
#         basisevaler = get_basisevaler(AM, j, 1)
#         offsets[j+1] = offsets[j] + size(basisevaler.cvals,1)
#     end
#     action_input2::Array{T,1} = zeros(T,offsets[end-1])
#     maxdofitems::Array{Int,1} = get_maxdofitems(AM)
#     localmatrix::Array{T,2} = zeros(T,get_maxndofs(AM,newton_args[1]),get_maxndofs(AM,nFE))
#     coeffs = zeros(T,maximum(maxdofs))
#     weights::Array{T,1} = get_qweights(AM)
#     itemfactor::T = 0
#     arow::Int = 0
#     acol::Int = 0
#     regions::Array{Int,1} = AP.regions
#     allitems::Bool = (regions == [0])
#     nregions::Int = length(regions)

#     # note: at the moment we expect that all FE[1:end-1] are the same !
#     # otherwise more than one MatrixBlock has to be assembled and we need more offset information
#     # hence, this only can handle nonlinearities at the moment that depend on one unknown of the PDEsystem

#     loop_allocations = @allocated for item = 1 : nitems
#     for r = 1 : nregions
#     # check if item region is in regions
#     if allitems || xItemRegions[item] == regions[r]

#         # update assembly manager (also updates necessary basisevaler)
#         update_assembly!(AM, item)
#         weights = get_qweights(AM)

#         # fill action input with evaluation of current solution
#         # assemble all but the last operators into action_input
#         for FEid = 1 : nFE - 1
#             for di = 1 : maxdofitems[FEid]
#                 if AM.dofitems[FEid][di] != 0
#                     # get correct basis evaluator for dofitem (was already updated by AM)
#                     basisevaler = get_basisevaler(AM, FEid, di)
    
#                     # update action on dofitem
#                     # update_action!(action, basisevaler, AM.dofitems[FEid][di], item, regions[r])

#                     # get coefficients of FE number FEid on current dofitem
#                     get_coeffs!(coeffs, FEB[FEid], AM, FEid, di)
#                     coeffs .*= AM.coeff4dofitem[FEid][di]

#                     # write evaluation of operator of current FE into action_input
#                     for i in eachindex(weights)
#                         eval_febe!(action_input[i], basisevaler, coeffs, i, offsets[FEid])
#                     end  
#                 end
#             end
#         end

#         # update action on dofitem (not needed yet)
#         basisevaler2 = get_basisevaler(AM, nFE, 1)
#         basisvals = basisevaler2.cvals
#         update_action!(action, basisevaler, item, item, regions[r])

#         for i in eachindex(weights)
#             for dof_i = 1 : get_ndofs(AM, newton_args[1], 1)

#                 # evaluate operators of ansatz function (but only of those operators in the current block)
#                 # (only of the newton arguments that are associated to the current matrix block)
#                 for newarg in newton_args
#                     basisevaler = get_basisevaler(AM, newarg, 1)
#                     eval_febe!(action_input2, basisevaler, dof_i, i, offsets[newarg])
#                 end

#                 # apply nonlinear action
#                 apply_action!(action_result, action_input[i], action_input2, action, i, nothing)  
#                 action_result .*= weights[i]

#                 # multiply test function operator evaluation
#                 for dof_j = 1 : get_ndofs(AM, nFE, 1)
#                     temp = 0
#                     for k = 1 : action_resultdim
#                         temp += action_result[k] * basisvals[k,dof_j,i]
#                     end
#                     localmatrix[dof_i,dof_j] += temp
#                 end
#             end 
#             fill!(action_input[i],0)
#         end

#         itemfactor = xItemVolumes[item] * factor * AM.coeff4dofitem[nFE][1]

#         # copy localmatrix into global matrix
#         for dof_i = 1 : get_ndofs(AM, newton_args[1], 1)
#             arow = get_dof(AM, newton_args[1], 1, dof_i) + offsetY # offsetY refers to the newton argument offset
#             for dof_j = 1 : get_ndofs(AM, nFE, 1)
#                # if localmatrix[dof_i,dof_j] != 0
#                     acol = get_dof(AM, nFE, 1, dof_j) + offsetX # offsetX refers to the test function offset
#                     if transposed_assembly == true
#                         _addnz(A,acol,arow,localmatrix[dof_i,dof_j],itemfactor)
#                     else 
#                         _addnz(A,arow,acol,localmatrix[dof_i,dof_j],itemfactor)  
#                     end
#                # end
#             end
#         end
        
#         fill!(localmatrix,0.0)
#         break; # region for loop
#     end # if in region    
#     end # region for loop
#     end # item for loop

#     AP.last_allocations = loop_allocations

#     return nothing
# end


# ## wrapper for FEMatrixBlock to avoid use of setindex! functions of FEMAtrixBlock
# function assemble!(
#     A::FEMatrixBlock,
#     AP::AssemblyPattern{APT,T,AT},
#     FEB::Array{<:FEVectorBlock{T,Tv,Ti},1};
#     factor = 1,
#     fixed_arguments = nothing, # ignored
#     transposed_assembly::Bool = false,
#     skip_preps::Bool = false) where {APT <: APT_NonlinearForm, T <: Real, AT <: AssemblyType, Tv, Ti}

#     assemble!(A.entries, AP, FEB; factor = factor, transposed_assembly = transposed_assembly, offsetX = A.offsetX, offsetY = A.offsetY, skip_preps = skip_preps)
# end




# """
# ````
# assemble!(
#     b::AbstractVector,                     # target vector
#     AP::AssemblyPattern{APT,T,AT},         # NonlinearForm pattern
#     FEB::Array{<:FEVectorBlock,1};         # coefficients of current solution for each operator
#     factor = 1)                            # factor that is multiplied
#     where {APT <: APT_NonlinearForm, T, AT}
# ````

# Assembly of a NonlinearForm AP into given AbstractVector (e.g. FEMatrixBlock).
# """
# function assemble!(
#     b::AbstractVector,
#     AP::AssemblyPattern{APT,T,AT},
#     FEB::Array{<:FEVectorBlock{T,Tv,Ti},1};
#     factor = 1,
#     skip_preps::Bool = false,
#     offset = 0) where {APT <: APT_NonlinearForm, T <: Real, AT <: AssemblyType, Tv, Ti}

#     # prepare assembly
#     FE = AP.FES
#     nFE = length(FE)
#     if !skip_preps
#         prepare_assembly!(AP)
#     end
#     AM::AssemblyManager{T} = AP.AM
#     xItemVolumes::Array{Tv,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
#     xItemRegions::GridRegionTypes{Ti} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
#     nitems = length(xItemVolumes)

#     # prepare action
#     action = AP.action
#     action_resultdim::Int = action.argsizes[1]
#     maxnweights = get_maxnqweights(AM)
#     action_input = Array{Array{T,1},1}(undef,maxnweights)
#     for j = 1 : maxnweights
#         action_input[j] = zeros(T,action.argsizes[3]) # heap for action input
#     end
#     action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output

#     if AP.regions != [0]
#         @logmsg MoreInfo "Assembling $(AP.name) for current $((p->p.name).(FEB)) into vector ($AT in regions = $(AP.regions))"
#     else
#         @logmsg MoreInfo "Assembling $(AP.name) for current $((p->p.name).(FEB)) into vector ($AT)"
#     end
#     @debug AP AM.qf[1]

#     # loop over items
#     offsets::Array{Int,1} = zeros(Int,nFE+1)
#     maxdofs::Array{Int,1} = get_maxndofs(AM)
#     basisevaler = get_basisevaler(AM, 1, 1)
#     basisvals::Union{SharedCValView{T},Array{T,3}} = basisevaler.cvals
#     for j = 1 : nFE
#         basisevaler = get_basisevaler(AM, j, 1)
#         offsets[j+1] = offsets[j] + size(basisevaler.cvals,1)
#     end
#     maxdofitems::Array{Int,1} = get_maxdofitems(AM)
#     localb::Array{T,1} = zeros(T,get_maxndofs(AM,nFE))
#     coeffs::Array{T,1} = zeros(T,maximum(maxdofs))
#     weights::Array{T,1} = get_qweights(AM)
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

#         # fill action input with evluation of current solution
#         # assemble all but the last operators into action_input
#         for FEid = 1 : nFE - 1
#             for di = 1 : maxdofitems[FEid]
#                 if AM.dofitems[FEid][di] != 0
#                     # get correct basis evaluator for dofitem (was already updated by AM)
#                     basisevaler = get_basisevaler(AM, FEid, di)
    
#                     # update action on dofitem (not needed for these type of actions yet)
#                     # update_action!(action, basisevaler, AM.dofitems[FEid][di], item, regions[r])

#                     # get coefficients of FE number FEid on current dofitem
#                     get_coeffs!(coeffs, FEB[FEid], AM, FEid, di)
#                     coeffs .*= AM.coeff4dofitem[FEid][di]

#                     # write evaluation of operator of current FE into action_input
#                     for i in eachindex(weights)
#                         eval_febe!(action_input[i], basisevaler, coeffs, i, offsets[FEid])
#                     end  
#                 end
#             end
#         end

#         # update action on dofitem (not needed yet)
#         basisevaler = get_basisevaler(AM, nFE, 1)
#         basisvals = basisevaler.cvals
#         update_action!(action, basisevaler, item, item, regions[r])

#         for i in eachindex(weights)
#             apply_action!(action_result, action_input[i], action, i, nothing)
#             action_result .*= weights[i]

#             for dof_j = 1 : get_ndofs(AM, nFE, 1)
#                 temp = 0
#                 for k = 1 : action_resultdim
#                     temp += action_result[k] * basisvals[k,dof_j,i]
#                 end
#                 localb[dof_j] += temp
#             end
#             fill!(action_input[i],0)
#         end

#         localb .*= xItemVolumes[item] * factor * AM.coeff4dofitem[nFE][1]

#         # copy localmatrix into global matrix
#         for dof_i = 1 : get_ndofs(AM, nFE, 1)
#             b[get_dof(AM, nFE, 1, dof_i) + offset] += localb[dof_i]          
#         end
        
#         fill!(localb,0.0)
#         break; # region for loop
#     end # if in region    
#     end # region for loop
#     end # item for loop

#     AP.last_allocations = loop_allocations

#     return nothing
# end


# ## wrapper for FEVectorBlock to avoid use of setindex! functions of FEMAtrixBlock
# function assemble!(
#     b::FEVectorBlock,
#     AP::AssemblyPattern{APT,T,AT},
#     FEB::Array{<:FEVectorBlock{T,Tv,Ti},1};
#     factor = 1,
#     fixed_arguments = [], # ignored
#     skip_preps::Bool = false) where {APT <: APT_NonlinearForm, T <: Real, AT <: AssemblyType, Tv, Ti}

#     assemble!(b.entries, AP, FEB; factor = factor, offset = b.offset, skip_preps = skip_preps)
# end




# """
# ````
# assemble!(
#     AP::AssemblyPattern{APT,T,AT},         # NonlinearForm pattern
#     FEB::Array{<:FEVectorBlock,1};         # coefficients of current solution for each operator
#     FEBtest::FEVectorBlock;                # coefficients of test function for test function operator
#     factor = 1)                            # factor that is multiplied
#     where {APT <: APT_NonlinearForm, T, AT}
# ````

# Evaluation of a NonlinearForm AP for given coefficients of ansatz and test function.
# """
# function evaluate(
#     AP::AssemblyPattern{APT,T,AT},
#     FEB::Array{<:FEVectorBlock,1},
#     FEBtest::FEVectorBlock{T,Tv,Ti};
#     factor = 1,
#     skip_preps::Bool = false,
#     offset = 0) where {APT <: APT_NonlinearForm, T <: Real, AT <: AssemblyType, Tv, Ti}

#     # prepare assembly
#     FE = AP.FES
#     nFE = length(FE)
#     if !skip_preps
#         prepare_assembly!(AP)
#     end
#     AM::AssemblyManager{T} = AP.AM
#     xItemVolumes::Array{Tv,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
#     xItemRegions::GridRegionTypes{Ti} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
#     nitems = length(xItemVolumes)

#     # prepare action
#     action = AP.action
#     action_resultdim::Int = action.argsizes[1]
#     maxnweights = get_maxnqweights(AM)
#     action_input = Array{Array{T,1},1}(undef,maxnweights)
#     for j = 1 : maxnweights
#         action_input[j] = zeros(T,action.argsizes[3]) # heap for action input
#     end
#     action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output

#     if AP.regions != [0]
#         @logmsg MoreInfo "Assembling $(AP.name) for current $((p->p.name).(FEB)) into vector ($AT in regions = $(AP.regions))"
#     else
#         @logmsg MoreInfo "Assembling $(AP.name) for current $((p->p.name).(FEB)) into vector ($AT)"
#     end
#     @debug AP

#     # loop over items
#     offsets = zeros(Int,nFE+1)
#     maxdofs = get_maxndofs(AM)
#     basisevaler::FEBasisEvaluator{T,Tv,Ti} = get_basisevaler(AM, 1, 1)
#     for j = 1 : nFE
#         basisevaler = get_basisevaler(AM, j, 1)
#         offsets[j+1] = offsets[j] + size(basisevaler.cvals,1)
#     end
#     maxdofitems::Array{Int,1} = get_maxdofitems(AM)
#     coeffs = zeros(T,maximum(maxdofs))
#     weights::Array{T,1} = get_qweights(AM)
#     regions::Array{Int,1} = AP.regions
#     allitems::Bool = (regions == [0])
#     nregions::Int = length(regions)

#     eval_result::Float64 = 0
#     for item = 1 : nitems
#     for r = 1 : nregions
#     # check if item region is in regions
#     if allitems || xItemRegions[item] == regions[r]

#         # update assembly manager (also updates necessary basisevaler)
#         update_assembly!(AM, item)
#         weights = get_qweights(AM)

#         # fill action input with evluation of current solution
#         # assemble all but the last operators into action_input
#         for FEid = 1 : nFE - 1
#             for di = 1 : maxdofitems[FEid]
#                 if AM.dofitems[FEid][di] != 0
#                     # get correct basis evaluator for dofitem (was already updated by AM)
#                     basisevaler = get_basisevaler(AM, FEid, di)
    
#                     # update action on dofitem (not needed for these type of actions yet)
#                     # update_action!(action, basisevaler, AM.dofitems[FEid][di], item, regions[r])

#                     # get coefficients of FE number FEid on current dofitem
#                     get_coeffs!(coeffs, FEB[FEid], AM, FEid, di)
#                     coeffs .*= AM.coeff4dofitem[FEid][di]

#                     # write evaluation of operator of current FE into action_input
#                     for i in eachindex(weights)
#                         eval_febe!(action_input[i], basisevaler, coeffs, i, offsets[FEid])
#                     end  
#                 end
#             end
#         end

#         # update action on dofitem (not needed yet)
#         basisevaler = get_basisevaler(AM, nFE, 1)
#         update_action!(action, basisevaler, item, item, regions[r])

#         # get coefficients of test function
#         get_coeffs!(coeffs, FEBtest, AM, nFE, 1)

#         for i in eachindex(weights)
#             apply_action!(action_result, action_input[i], action, i, nothing)
#             action_result .*= weights[i] * xItemVolumes[item] * factor 

#             fill!(action_input[i],0)
#             eval_febe!(action_input[i], basisevaler, coeffs, i)

#             for k = 1 : action_resultdim
#                 eval_result += action_result[k] * action_input[i][k]
#             end
#             fill!(action_input[i],0)
#         end
#         break; # region for loop
#     end # if in region    
#     end # region for loop
#     end # item for loop

#     return eval_result
# end