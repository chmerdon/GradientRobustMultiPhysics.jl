##################
# PointEvaluator #
##################


struct PointEvaluator{T <: Real, FEType <: AbstractFiniteElement, EG <: AbstractElementGeometry, FEOP <: AbstractFunctionOperator, AT <: AbstractAssemblyType, ACT <: AbstractAction}
    FEBE::FEBasisEvaluator{T,FEType,EG,FEOP,AT} ## evaluates the FE basis on the segment (needs recomputation of quadrature points on entering cell)
    FEB::FEVectorBlock{T} # holds coefficients
    xItemDofs::DofMapTypes # holds dof numbers
    action::ACT # additional action to postprocess operator evaluation
    action_input::Array{T,1}
end


function PointEvaluator{T,FEType,EG,FEOP,AT}(FES::FESpace, FEB::FEVectorBlock, action::AbstractAction = NoAction()) where {T, FEType, EG, FEOP, AT}
    qf = QuadratureRule{T, EG}(0) # dummy quadrature
    
    FEBE = FEBasisEvaluator{Float64,FEType,EG,FEOP,AT}(FES, qf; mutable = true)
    
    DM = Dofmap4AssemblyType(FEB.FES, AT)

    if typeof(action) <: NoAction
        action_input = zeros(Float64,size(FEBE.cvals,1))
    else
        action_input = zeros(Float64,action.argsizes[2])
    end

    return PointEvaluator{T,FEType,EG,FEOP,AT,typeof(action)}(FEBE, FEB, DM, action, action_input)
end

function evaluate!(
    result::AbstractArray{T,1},
    PE::PointEvaluator{T},
    xref::AbstractArray{T,1},
    item::Int # cell used to evaluate local coordinates
    ) where {T}

    FEBE = PE.FEBE
    FEB = PE.FEB

    # update basis evaluations on new quadrature points
    relocate_xref!(FEBE, [xref])
    
    # update operator eveluation on item
    update!(FEBE, item)

    # evaluate
    action = PE.action
    action_input::Array{T,1} = PE.action_input
    coeffs::Array{T,1} = FEB.entries
    xItemDofs::DofMapTypes = PE.xItemDofs

    fill!(result,0)
    if !(typeof(action) <: NoAction)
        # update_action
        update!(action, FEBE, item, item, 0) # region is missing currently

        # evaluate operator
        fill!(action_input,0)
        @time eval!(action_input, FEBE, coeffs, xItemDofs, 1)
        apply_action!(result,action_input,action,1,xref)
    else
        @time eval!(result, FEBE, coeffs, xItemDofs, 1)
    end

    return nothing
end