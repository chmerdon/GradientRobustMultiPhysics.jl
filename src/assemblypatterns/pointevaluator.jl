##################
# PointEvaluator #
##################


"""
$(TYPEDEF)

structure that allows to evaluate a FEVectorBlock at arbitrary points
"""
struct PointEvaluator{T <: Real, Tv <: Real, Ti <: Integer, FEType <: AbstractFiniteElement, FEOP <: AbstractFunctionOperator, AT <: AssemblyType, ACT <: AbstractAction}
    FEBE::Array{FEBasisEvaluator{T,Tv,Ti,FEType},1} ## evaluates the FE basis on the possible element geometries
    FEB::FEVectorBlock{T,Tv,Ti} # holds coefficients
    EG::Array{DataType,1} # Element geometries
    xItemGeometries::GridEGTypes
    xItemDofs::DofMapTypes{Ti} # holds dof numbers
    action::ACT # additional action to postprocess operator evaluation
    action_input::Array{T,1}
end


"""
````
function PointEvaluator(FEB::FEVectorBlock, FEOP::AbstractFunctionOperator, action::AbstractAction = NoAction(); AT = ON_CELLS)
````
constructor for PointEvaluator that evaluate the given FEVectorBlock with the specified operator (possibly postprocessed by an action) at arbitrary points
inside entities of the given assembly type
"""
function PointEvaluator(FEB::FEVectorBlock{T,Tv,Ti}, FEOP,action::AbstractAction = NoAction(); AT = ON_CELLS) where {T, Tv, Ti}
    
    xgrid = FEB.FES.xgrid
    EG = xgrid[GridComponentUniqueGeometries4AssemblyType(AT)]
    xItemGeometries = xgrid[GridComponentGeometries4AssemblyType(AT)]
    qf = QuadratureRule{T, EG[1]}(0) # dummy quadrature

    FEType = eltype(FEB.FES)
    FEBE = Array{FEBasisEvaluator{T,Tv,Ti,FEType},1}(undef,length(EG))
    
    for j = 1 : length(EG)
        FEBE[j] = FEBasisEvaluator{T,EG[j],FEOP,AT}(FEB.FES, qf)
    end
    
    DM = Dofmap4AssemblyType(FEB.FES, AT)

    if typeof(action) <: NoAction
        action_input = zeros(T,size(FEBE[1].cvals,1))
    else
        action_input = zeros(T,action.argsizes[2])
    end

    return PointEvaluator{T,Tv,Ti,FEType,FEOP,AT,typeof(action)}(FEBE, FEB, EG, xItemGeometries, DM, action, action_input)
end

"""
````
function evaluate(PE::PointEvaluator)
````
Returns the function
    (result,xref,cell) --> evaluate!(result,PE,xref,cell)

(e.g. to be used as a callback function in vectorplot!)
"""
function evaluate(PE::PointEvaluator)
    return (result,xref,item) -> evaluate!(result,PE,xref,item)
end

"""
````
function evaluate!(
    result,                     # target for result
    PE::PointEvaluator,         
    xref,                       # local coordinates inside item
    item                        # item number
    ) where  {T, Tv, Ti, FEType, FEOP, AT, ACT}
````
Evaluates the PointEvaluator at the point with the given local coordinates insides the item with the specified item number.
(To get the local coordinates, currently a CellFinder has to be maintained manually, this might change in future.)

"""
function evaluate!(
    result,
    PE::PointEvaluator{T, Tv, Ti, FEType, FEOP, AT, ACT},
    xref,
    item # cell used to evaluate local coordinates
    ) where  {T, Tv, Ti, FEType, FEOP, AT, ACT}

    iEG::Int = findfirst(isequal(PE.xItemGeometries[item]), PE.EG)
    FEBE = PE.FEBE[iEG]
    FEB = PE.FEB

    # update basis evaluations at xref
    relocate_xref!(FEBE, xref)
    
    # update operator eveluation on item
    update_febe!(FEBE, item)

    # evaluate
    action = PE.action
    action_input::Array{T,1} = PE.action_input
    coeffs::Array{T,1} = FEB.entries
    basisvals::Array{T,3} = FEBE.cvals
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