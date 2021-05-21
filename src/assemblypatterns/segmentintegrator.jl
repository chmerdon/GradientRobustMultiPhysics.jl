#####################
# SegmentIntegrator #
#####################


struct SegmentIntegrator{T <: Real, FEType <: AbstractFiniteElement, EG <: AbstractElementGeometry, FEOP <: AbstractFunctionOperator, AT <: AbstractAssemblyType, SG <: AbstractElementGeometry, ACT <: AbstractAction}
    FEBE::FEBasisEvaluator{T,FEType,EG,FEOP,AT} ## evaluates the FE basis on the segment (needs recomputation of quadrature points on entering cell)
    FEB::FEVectorBlock # holds coefficients
    xItemDofs::DofMapTypes # holds dof numbers
    action::ACT # additional action to postprocess operator evaluation
    action_input::Array{T,1}
    action_result::Array{T,1}
    cvol::Base.RefValue{T}
    xrefSG::Array{Array{T,1},1} ## xref from qf
    qf::QuadratureRule{T,SG} ## quadrature formula for the segment geometry
end


function SegmentIntegrator{T,FEType,EG,FEOP,AT,SG}(FES::FESpace, FEB::FEVectorBlock, order::Int, action::AbstractAction) where {T, FEType, EG, FEOP, AT,SG}
    qf_SG = QuadratureRule{T, SG}(order)

    dimfill = dim_element(EG) - dim_element(SG)
    @assert dimfill >= 0

    if dimfill > 0
        xref = Array{Array{Float64,1},1}(undef,length(qf_SG.xref))
        for i = 1 : length(qf_SG.xref)
            xref[i] = zeros(Float64,dim_element(EG))
        end
        FEBE = FEBasisEvaluator{Float64,FEType,EG,FEOP,AT}(FES, xref; mutable = true)
    else
        FEBE = FEBasisEvaluator{Float64,FEType,EG,FEOP,AT}(FES, qf_SG; mutable = true)
    end
    DM = Dofmap4AssemblyType(FEB.FES, AT)


    if typeof(action) <: NoAction
        action_input = zeros(Float64,size(FEBE.cvals,1))
        action_result = action_input
    else
        action_input = zeros(Float64,action.argsizes[2])
        action_result = zeros(Float64,action.argsizes[1])
    end
    cvol::T = 0

    return SegmentIntegrator{T,FEType,EG,FEOP,AT,SG,typeof(action)}(FEBE, FEB, DM, action, action_input, action_result, Ref(cvol), qf_SG.xref, qf_SG)
end

function integrate!(
    result::Array{T,1},
    SI::SegmentIntegrator{T},
    w::Array{Array{T,1},1},    # world coordinates
    b::Array{Array{T,1},1},    # barycentric coordinates (w.r.t. item geometry)
    item::Int # cell in which the segment lies (completely)
    ) where {T}

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
    update!(FEBE, item)

    # compute volume
    action = SI.action
    cvol::Base.Ref{Float64} = SI.cvol
    if typeof(SI).parameters[6] <: AbstractElementGeometry1D
        cvol[] = sqrt((w[1][1] - w[2][1])^2 + (w[1][2] - w[2][2])^2)
    else
        @error "This segment geometry is not implemented!"
    end

    # update_action
    update!(action, FEBE, item, item, 0) # region is missing currently

    # do the integration
    qf = SI.qf
    action_input::Array{T,1} = SI.action_input
    action_result::Array{T,1} = SI.action_result
    coeffs::Array{T,1} = FEB.entries
    basisvals::Array{T,3} = FEBE.cvals
    xItemDofs::DofMapTypes = SI.xItemDofs
    weights::Array{T,1} = qf.w

    fill!(result,0)

    for i in eachindex(weights)

        # apply_action
        if !(typeof(action) <: NoAction)
            # evaluate operator
            fill!(SI.action_input,0)
            for dof_i = 1 : size(basisvals,2), k = 1 : length(action_input)
                action_input[k] += coeffs[xItemDofs[dof_i,item] + SI.FEB.offset] * basisvals[k,dof_i,i]
            end

            apply_action!(action_result,action_input,action,i,xref[i])

            # accumulate
            for k = 1 : length(action_result)
                result[k] += action_result[k] * weights[i]
            end
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