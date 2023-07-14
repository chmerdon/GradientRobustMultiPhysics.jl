#####################
# SegmentIntegrator #
#####################


struct SegmentIntegrator{T <: Real, Tv <: Real, Ti <: Integer, FEType <: AbstractFiniteElement, EG <: AbstractElementGeometry, FEOP <: AbstractFunctionOperator, AT <: AssemblyType, SG <: AbstractElementGeometry, ACT <: AbstractAction}
    FEBE::FEEvaluator{T,Tv,Ti} ## evaluates the FE basis on the segment (needs recomputation of quadrature points on entering cell)
    FEB::FEVectorBlock{T,Tv,Ti} # holds coefficients
    xItemDofs::DofMapTypes{Ti} # holds dof numbers
    action::ACT # additional action to postprocess operator evaluation
    action_input::Array{T,1}
    cvol::Base.RefValue{T}
    xrefSG::Array{Array{T,1},1} ## xref from qf
    qf::QuadratureRule{T,SG} ## quadrature formula for the segment geometry
end


function SegmentIntegrator{T}(EG,FEOP,SG,FES::FESpace{Tv,Ti,FEType,FEAT}, FEB::FEVectorBlock{T,Tv,Ti}, order::Int, action::AbstractAction = NoAction(); AT = ON_CELLS) where {T, Tv, Ti, FEType, FEAT}
    qf_SG = QuadratureRule{T, SG}(order)

    dimfill = dim_element(EG) - dim_element(SG)
    @assert dimfill >= 0

    xrefSG = deepcopy(qf_SG.xref)
    if dimfill > 0
        new_xref = Array{Array{T,1},1}(undef,length(qf_SG.xref))
        for i = 1 : length(qf_SG.xref)
            new_xref[i] = zeros(T,dim_element(EG))
        end
        qf_aux = SQuadratureRule{Float64, EG, dim_element(EG), length(qf_SG.xref)}("aux", new_xref, qf_SG.w)
        FEBE = FEEvaluator(FES, FEOP, qf_aux; T = T, AT = AT)
    else
        FEBE = FEEvaluator(FES, FEOP, qf_SG; T = T, AT = AT)
    end
    DM = Dofmap4AssemblyType(FEB.FES, AT)


    if typeof(action) <: NoAction
        action_input = zeros(T,size(FEBE.cvals,1))
    else
        action_input = zeros(T,action.argsizes[2])
    end
    cvol::T = 0

    return SegmentIntegrator{T,Tv,Ti,FEType,EG,FEOP,AT,SG,typeof(action)}(FEBE, FEB, DM, action, action_input, Ref(cvol), xrefSG, qf_SG)
end

function integrate!(
    result::Array{T,1},
    SI::SegmentIntegrator{T,Tv,Ti,FEType,EG,FEOP,AT,SG,ACT},
    w::Array{Array{T,1},1},    # world coordinates
    b::Array{Array{T,1},1},    # barycentric coordinates (w.r.t. item geometry)
    item::Int # cell in which the segment lies (completely)
    ) where {T,Tv,Ti,FEType,EG,FEOP,AT,SG,ACT}

    FEBE = SI.FEBE
    FEB = SI.FEB

    xref::Array{Array{T,1},1} = FEBE.xref
    xrefSG::Array{Array{T,1},1} = SI.xrefSG

    for i = 1 : length(xref)
        fill!(xref[i],0)
        for k = 1 : length(xref[1])
            for j = 1 : length(b) - 1
                xref[i][k] += xrefSG[i][j] * b[j][k]
            end
            xref[i][k] += (1-sum(xrefSG[i])) * b[end][k]
        end
    end

    # update basis evaluations on new quadrature points
    relocate_xref!(FEBE, xref)
    
    # update operator eveluation on item
    update_basis!(FEBE, item)
    @show 

    # compute volume
    action = SI.action
    cvol::Base.Ref{T} = SI.cvol
    if SG <: AbstractElementGeometry1D
        cvol[] = sqrt((w[1][1] - w[2][1])^2 + (w[1][2] - w[2][2])^2)
    else
        @error "This segment geometry is not implemented!"
    end

    # update_action
    if is_itemdependent(action)
        action.item[1] = item
        action.item[2] = item
        action.item[3] = 0 # todo
    end

    # do the integration
    qf = SI.qf
    action_input::Array{T,1} = SI.action_input
#    action_result::Array{T,1} = SI.action_result
    coeffs::Array{T,1} = FEB.entries
    basisvals::Union{SharedCValView{T},Array{T,3}} = FEBE.cvals
    xItemDofs::DofMapTypes{Int32} = SI.xItemDofs
    weights::Array{T,1} = qf.w

    fill!(result,0)

    for i in eachindex(weights)

        # apply_action
        if !(typeof(action) <: NoAction)

            if is_xdependent(action)
                update_trafo!(FEBE.L2G, item)
                eval_trafo!(action.x, FEBE.L2G, xref[i])
            end
            if is_xrefdependent(action)
                action.xref = xref
            end

            # evaluate operator
            fill!(action_input,0)
            for dof_i = 1 : size(basisvals,2), k = 1 : length(action_input)
                action_input[k] += coeffs[xItemDofs[dof_i,item] + SI.FEB.offset] * basisvals[k,dof_i,i]
            end

            eval_action!(action,action_input)

            # accumulate
            result .+= action.val * weights[i]
        else
            for dof_i = 1 : size(basisvals,2), k = 1 : length(action_input)
                result[k] += coeffs[xItemDofs[dof_i,item] + SI.FEB.offset] * basisvals[k,dof_i,i] * weights[i]
            end
        end

    end

    # multiply volume
    result .*= cvol[]
    return nothing
end