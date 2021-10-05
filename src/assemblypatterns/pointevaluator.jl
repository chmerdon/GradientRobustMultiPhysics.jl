##################
# PointEvaluator #
##################


struct PointEvaluator{T <: Real, Tv <: Real, Ti <: Integer, FEType <: AbstractFiniteElement, EG <: AbstractElementGeometry, FEOP <: AbstractFunctionOperator, AT <: AssemblyType, ACT <: AbstractAction, BVT}
    FEBE::FEBasisEvaluator{T,Tv,Ti,FEType,EG,FEOP,AT} ## evaluates the FE basis on the segment (needs recomputation of quadrature points on entering cell)
    basisvals::BVT
    FEB::FEVectorBlock{T,Tv,Ti} # holds coefficients
    xItemDofs::DofMapTypes{Ti} # holds dof numbers
    action::ACT # additional action to postprocess operator evaluation
    action_input::Array{T,1}
end


function PointEvaluator{T}(EG, FEOP, FES::FESpace{Tv,Ti,FEType,FEAT}, FEB::FEVectorBlock{T,Tv,Ti}, action::AbstractAction = NoAction(); AT = ON_CELLS) where {T, Tv, Ti, FEType, FEAT}
    qf = QuadratureRule{T, EG}(0) # dummy quadrature
    
    FEBE = FEBasisEvaluator{T,EG,FEOP,AT}(FES, qf; mutable = true)
    
    DM = Dofmap4AssemblyType(FEB.FES, AT)

    if typeof(action) <: NoAction
        action_input = zeros(T,size(FEBE.cvals,1))
    else
        action_input = zeros(T,action.argsizes[2])
    end

    return PointEvaluator{T,Tv,Ti,FEType,EG,FEOP,AT,typeof(action),typeof(FEBE.cvals)}(FEBE, FEBE.cvals, FEB, DM, action, action_input)
end

function evaluate!(
    result::AbstractArray{T,1},
    PE::PointEvaluator{T, Tv, Ti, FEType, EG, FEOP, AT, ACT, BVT},
    xref::Array{T,1},
    item::Int # cell used to evaluate local coordinates
    ) where  {T, Tv, Ti, FEType, EG, FEOP, AT, ACT, BVT}

    FEBE = PE.FEBE
    FEB = PE.FEB

    # update basis evaluations at xref
    relocate_xref!(FEBE, xref)
    
    # update operator eveluation on item
    update_febe!(FEBE, item)

    # evaluate
    action = PE.action
    action_input::Array{T,1} = PE.action_input
    coeffs::Array{T,1} = FEB.entries
    basisvals::BVT = PE.basisvals
    xItemDofs::DofMapTypes{Ti} = PE.xItemDofs

    fill!(result,0)
    if !(ACT <: NoAction)
        # update_action
        update_action!(action, FEBE, item, item, 0) # region is missing currently

        # evaluate operator
        fill!(action_input,0)
        for dof_i = 1 : size(basisvals,2), k = 1 : length(action_input)
            action_input[k] += coeffs[xItemDofs[dof_i,item] + FEB.offset] * basisvals[k,dof_i,1]
        end
        apply_action!(result,action_input,action,1,xref)
    else
        for dof_i = 1 : size(basisvals,2), k = 1 : length(result)
            result[k] += coeffs[xItemDofs[dof_i,item] + FEB.offset] * basisvals[k,dof_i,1]
        end
    end

    return nothing
end