module FEOperator

using FiniteElements
using ExtendableGrids
using FEXGrid
using QuadratureRules
using ExtendableSparse
using SparseArrays
using ForwardDiff # for FEBasisEvaluator

# would be usefull, but not working like this at the moment
#GridComponentDofs4AssemblyType(::Type{AbstractAssemblyTypeCELL}) = CellDofs
#GridComponentDofs4AssemblyType(::Type{AbstractAssemblyTypeFACE}) = FaceDofs
#GridComponentDofs4AssemblyType(::Type{AbstractAssemblyTypeBFACE}) = BFaceDofs

# instead currently, this is used:
FEPropertyDofs4AssemblyType(FE::AbstractFiniteElement,::Type{AbstractAssemblyTypeCELL}) = FE.CellDofs
FEPropertyDofs4AssemblyType(FE::AbstractFiniteElement,::Type{AbstractAssemblyTypeFACE}) = FE.FaceDofs
FEPropertyDofs4AssemblyType(FE::AbstractFiniteElement,::Type{AbstractAssemblyTypeBFACE}) = FE.BFaceDofs
FEPropertyDofs4AssemblyType(FE::AbstractFiniteElement,::Type{AbstractAssemblyTypeBFACECELL}) = FE.CellDofs


abstract type AbstractFEFunctionOperator end # to dispatch which evaluator of the FE_basis_caller is used
abstract type Identity <: AbstractFEFunctionOperator end # 1*v_h
abstract type Gradient <: AbstractFEFunctionOperator end # D_geom(v_h)
abstract type SymmetricGradient <: AbstractFEFunctionOperator end # eps_geom(v_h)
abstract type Laplacian <: AbstractFEFunctionOperator end # L_geom(v_h)
abstract type Hessian <: AbstractFEFunctionOperator end # D^2(v_h)
abstract type Curl <: AbstractFEFunctionOperator end # only 2D: Curl(v_h) = D(v_h)^\perp
abstract type Rotation <: AbstractFEFunctionOperator end # only 3D: Rot(v_h) = D \times v_h
abstract type Divergence <: AbstractFEFunctionOperator end # div(v_h)
abstract type Trace <: AbstractFEFunctionOperator end # tr(v_h)
abstract type Deviator <: AbstractFEFunctionOperator end # dev(v_h)


NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Identity}) = 0
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Gradient}) = 1
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{SymmetricGradient}) = 1
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Laplacian}) = 2
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Hessian}) = 2
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Curl}) = 1
NeededDerivative4Operator(::Type{<:AbstractHcurlFiniteElement},::Type{Curl}) = 0
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Rotation}) = 1
NeededDerivative4Operator(::Type{<:AbstractHcurlFiniteElement},::Type{Rotation}) = 0
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Divergence}) = 1
NeededDerivative4Operator(::Type{<:AbstractHdivFiniteElement},::Type{Divergence}) = 0
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Trace}) = 0
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Deviator}) = 0

QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{Identity}) = 0
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{Gradient}) = -1
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{SymmetricGradient}) = -1
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{Laplacian}) = -2
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{Hessian}) = -2


export AbstractFEFunctionOperator
export Identity, Gradient, SymmetricGradient, Laplacian, Hessian, Curl, Rotation, Divergence, Trace, Deviator
export NeededDerivatives4Operator

include("FEBasisEvaluator.jl")
export FEBasisEvaluator, update!

include("AbstractAction.jl")
export DoNothingAction
export MultiplyScalarAction
export MultiplyVectorAction
export MultiplyMatrixAction
export RegionWiseMultiplyScalarAction
export RegionWiseMultiplyVectorAction
export FunctionAction
export XFunctionAction


abstract type AbstractFEForm end
# for each item: I(item) = Int_item action(FEfunction) dx
abstract type ItemIntegrals <: AbstractFEForm end 
# for each v_h: F(v_h) = Int_omega action(v_h) dx
abstract type LinearForm <: AbstractFEForm end   
# for each v_h,u_h from same FE: A(v_h,u_h) = sum_item Int_item action(u_h) dot v_h dx
abstract type SymmetricBilinearForm <: AbstractFEForm end 
# todo: for each v_h,u_h: B(v_h,u_h) = sum_item Int_item action(u_h) dot action(v_h) dx
abstract type MixedBilinearForm <: AbstractFEForm end 

export AbstractFEForm,ItemIntegrals,LinearForm,SymmetricBilinearForm,ASymmetricBilinearForm
export assemble!

# some shortcuts for assemble! defined at the bottom
export StiffnessMatrix!, RightHandSide!


# unique functions that only selects uniques in specified regions
function unique(xItemGeometries, xItemRegions, xItemDofs, regions)
    nitems = 0
    try
        nitems = num_sources(xItemGeometries)
    catch
        nitems = length(xItemGeometries)
    end      
    EG::Array{DataType,1} = []
    ndofs4EG::Array{Int32,1} = []
    iEG = 0
    cellEG = Triangle2D
    for item = 1 : nitems
        for j = 1 : length(regions)
            if xItemRegions[item] == regions[j]
                cellEG = xItemGeometries[item]
                iEG = 0
                for k = 1 : length(EG)
                    if cellEG == EG[k]
                        iEG = k
                        break;
                    end
                end
                if iEG == 0
                    append!(EG, [xItemGeometries[item]])
                    append!(ndofs4EG, num_targets(xItemDofs,item))
                end  
                break; # rest of for loop can be skipped
            end    
        end
    end    
    return EG, ndofs4EG
end

function prepareOperatorAssembly(form::Type{<:AbstractFEForm}, AT::Type{<:AbstractAssemblyType}, operator::Type{<:AbstractFEFunctionOperator}, FE::AbstractFiniteElement, regions::Array{Int}, NumberType::Type{<:Real}, nrfactors::Int, bonus_quadorder::Int, talkative::Bool)
    xItemGeometries = FE.xgrid[GridComponentTypes4AssemblyType(AT)]
    xItemRegions = FE.xgrid[GridComponentRegions4AssemblyType(AT)]
    xItemDofs = FEPropertyDofs4AssemblyType(FE,AT)

    # find unique ElementGeometries
    EG, ndofs4EG = unique(xItemGeometries, xItemRegions, xItemDofs, regions)

    # find proper quadrature QuadratureRules
    # and construct matching FEBasisEvaluators
    qf = Array{QuadratureRule,1}(undef,length(EG))
    basisevaler = Array{Array{FEBasisEvaluator,1},1}(undef,length(EG))
    quadorder = 0
    for j = 1 : length(EG)
        basisevaler[j] = Array{FEBasisEvaluator,1}(undef,1)
        quadorder = bonus_quadorder + nrfactors*(FiniteElements.get_polynomialorder(typeof(FE), EG[j]) + QuadratureOrderShift4Operator(typeof(FE),operator))
        qf[j] = QuadratureRule{NumberType,EG[j]}(quadorder);
        basisevaler[j][1] = FEBasisEvaluator{NumberType,typeof(FE),EG[j],operator,AT}(FE, qf[j])
    end        
    if talkative
        println("\nASSEMBLY PREPARATION")
        println("=====================")
        println("      form = $form")
        println("        FE = $(FE.name), ndofs = $(FE.ndofs)")
        println("  operator = $operator")
        println("   regions = $regions")
        println("  uniqueEG = $EG")
        for j = 1 : length(EG)
            println("\nQuadratureRule [$j] for $(EG[j]):")
            QuadratureRules.show(qf[j])
        end
    end
    dofitem4item(item) = item
    EG4item(item) = xItemGeometries[item]
    FEevaler4item(item) = 1
    return EG, ndofs4EG, qf, basisevaler, EG4item, dofitem4item, FEevaler4item
end

function prepareOperatorAssembly(form::Type{<:AbstractFEForm}, AT::Type{<:AbstractAssemblyTypeBFACECELL}, operator::Type{<:AbstractFEFunctionOperator}, FE::AbstractFiniteElement, regions::Array{Int}, NumberType::Type{<:Real}, nrfactors::Int, bonus_quadorder::Int, talkative::Bool)
    # find proper quadrature QuadratureRules
    xItemGeometries = FE.xgrid[GridComponentTypes4AssemblyType(AT)]
    xCellGeometries = FE.xgrid[CellGeometries]

    # find unique ElementGeometries
    # todo: we need unique CellGeometriy/BFaceGeometrie pairs in BFace regions

    qf = Array{QuadratureRule,1}(undef,length(EG))
    basisevaler = Array{Array{FEBasisEvaluator,1},1}(undef,length(EG))
    quadorder = 0
    nfaces4cell = 0
    for j = 1 : length(EG)
        itemEG = facetype_of_cellface(EG[j],1)
        nfaces4cell = nfaces_per_cell(EG[j])
        basisevaler[j] = Array{FEBasisEvaluator,1}(undef,nfaces4cell)
        quadorder = bonus_quadorder + nrfactors*FiniteElements.get_polynomialorder(typeof(FE), itemEG)
        qf[j] = QuadratureRule{NumberType,itemEG}(quadorder);
        qfxref = qf[j].xref
        xrefFACE2CELL = xrefFACE2xrefCELL(EG[j])
        for k = 1 : nfaces4cell
            for i = 1 : length(qfxref)
                qf[j].xref[i] = xrefFACE2CELL[k](qfxref[i])
            end
            basisevaler[j][k] = FEBasisEvaluator{NumberType,typeof(FE),EG[j],operator,AT}(FE, qf[j])
        end    
        for i = 1 : length(qfxref)
            qf[j].xref[i] = qfxref[i]
        end
    end        
    if talkative
        println("\nASSEMBLY PREPARATION")
        println("=====================")
        println("      form = $form")
        println("        FE = $(FE.name), ndofs = $(FE.ndofs)")
        println("  operator = $operator")
        println("   regions = $regions")
        println("  uniqueEG = $EG")
        for j = 1 : length(EG)
            println("\nQuadratureRule [$j] for $(EG[j]):")
            QuadratureRules.show(qf[j])
        end
    end
    dofitem4item(item) = FE.xgrid[FaceCells][1,FE.xgrid[BFaces][item]]
    EG4item(item) = xCellGeometries[FE.xgrid[FaceCells][1,FE.xgrid[BFaces][item]]]
    FEevaler4item(item) = FE.xgrid[BFaceCellPos][item]
    return EG, qf, basisevaler, EG4item , dofitem4item, FEevaler4item
end


function assemble!(
    b::Array{<:Real},
    form::Type{ItemIntegrals},
    AT::Type{<:AbstractAssemblyType},
    operator::Type{<:AbstractFEFunctionOperator},
    FEF::FEFunction,
    action::AbstractAction;
    bonus_quadorder::Int = 0,
    talkative::Bool = false,
    regions::Array{Int} = [1])

    NumberType = eltype(b)
    FE = FEF.FEType
    xItemNodes = FE.xgrid[GridComponentNodes4AssemblyType(AT)]
    xItemVolumes = FE.xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemGeometries = FE.xgrid[GridComponentTypes4AssemblyType(AT)]
    xItemDofs = FEPropertyDofs4AssemblyType(FE,AT)
    xItemRegions = FE.xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = num_sources(xItemNodes)

    EG, ndofs4EG, qf, basisevaler, EG4item, dofitem4item, evaler4item = prepareOperatorAssembly(form, AT, operator, FE, regions, NumberType, 1, bonus_quadorder, talkative)

    # collect FE and FEBasisEvaluator information
    ncomponents = FiniteElements.get_ncomponents(typeof(FE))
    cvals_resultdim = size(basisevaler[1][1].cvals[1],1)
    @assert size(b,2) == cvals_resultdim

    # loop over items
    itemET = xItemGeometries[1] # type of the current item
    iEG = 1 # index to the correct unique geometry
    ndofs4item = 0 # number of dofs for item
    evalnr = 0 # evaler number that has to be used for current item
    dofitem = 0 # itemnr where the dof numbers can be found
    dofs = zeros(Int32,max_num_targets_per_source(xItemDofs))
    action_input = zeros(NumberType,cvals_resultdim) # heap for action input
    action_result = zeros(NumberType,action.resultdim) # heap for action output
    cvali::Array{NumberType,2} = [[] []] # pointer to FEoperatorvalue at quadrature point i

    for item::Int32 = 1 : nitems
    for r = 1 : length(regions)
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        # item number for dof provider
        dofitem = dofitem4item(item)

        # find index for CellType
        itemET = EG4item(item)
        iEG = findfirst(isequal(itemET), EG)
        ndofs4item = ndofs4EG[iEG]

        # update FEbasisevaler
        evalnr = evaler4item(item)
        update!(basisevaler[iEG][evalnr],dofitem)

        # update action
        update!(action, basisevaler[iEG][evalnr], item)

        # update dofs
        dofs[1:ndofs4item] = xItemDofs[:,dofitem]

        for i in eachindex(qf[iEG].w)
            cvali = basisevaler[iEG][evalnr].cvals[i]
            # apply action to FEFunction
            fill!(action_input,0)
            for dof_i = 1 : ndofs4item
                for k = 1 : cvals_resultdim
                    action_input[k] += FEF.coefficients[dofs[dof_i]] * cvali[k,dof_i]
                end    
            end 
            apply_action!(action_result, action_input, action, i)
            for j = 1 : action.resultdim
                b[item,j] += action_result[j] * qf[iEG].w[i] * xItemVolumes[item]
            end
        end  
        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
end


function assemble!(
    b::Array{<:Real},
    form::Type{LinearForm},
    AT::Type{<:AbstractAssemblyType},
    operator::Type{<:AbstractFEFunctionOperator},
    FE::AbstractFiniteElement,
    action::AbstractAction;
    bonus_quadorder::Int = 0,
    talkative::Bool = false,
    regions::Array{Int} = [1])

    NumberType = eltype(b)
    xItemNodes = FE.xgrid[GridComponentNodes4AssemblyType(AT)]
    xItemVolumes = FE.xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemGeometries = FE.xgrid[GridComponentTypes4AssemblyType(AT)]
    xItemDofs = FEPropertyDofs4AssemblyType(FE,AT)
    xItemRegions = FE.xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = num_sources(xItemNodes)
    
    EG, ndofs4EG, qf, basisevaler, EG4item, dofitem4item, evaler4item = prepareOperatorAssembly(form, AT, operator, FE, regions, NumberType, 1, bonus_quadorder, talkative)

    # collect FE and FEBasisEvaluator information
    ncomponents = FiniteElements.get_ncomponents(typeof(FE))
    cvals_resultdim = size(basisevaler[1][1].cvals[1],1)
    @assert size(b,2) == cvals_resultdim

    # loop over items
    itemET = xItemGeometries[1] # type of the current item
    iEG = 1 # index to the correct unique geometry
    ndofs4item = 0 # number of dofs for item
    evalnr = 0 # evaler number that has to be used for current item
    dofitem = 0 # itemnr where the dof numbers can be found
    dofs = zeros(Int32,max_num_targets_per_source(xItemDofs))
    temp = 0 # some temporary variable
    action_input = zeros(NumberType,cvals_resultdim) # heap for action input
    action_result = zeros(NumberType,action.resultdim) # heap for action output
    cvali::Array{NumberType,2} = [[] []] # pointer to FEoperatorvalue at quadrature point i
    for item::Int32 = 1 : nitems
    for r = 1 : length(regions)
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        # item number for dof provider
        dofitem = dofitem4item(item)

        # find index for CellType
        itemET = EG4item(item)
        iEG = findfirst(isequal(itemET), EG)
        ndofs4item = ndofs4EG[iEG]

        # update FEbasisevaler
        evalnr = evaler4item(item)
        update!(basisevaler[iEG][evalnr],dofitem)

        # update action
        update!(action, basisevaler[iEG][evalnr], item)

        # update dofs
        dofs[1:ndofs4item] = xItemDofs[:,dofitem]

        for i in eachindex(qf[iEG].w)
            cvali = basisevaler[iEG][evalnr].cvals[i]

            for dof_i = 1 : ndofs4item
                # apply action
                for k = 1 : cvals_resultdim
                    action_input[k] = cvali[k,dof_i]
                end    
                apply_action!(action_result, action_input, action, i)

                for j = 1 : cvals_resultdim
                   b[dofs[dof_i],j] += action_result[j] * qf[iEG].w[i] * xItemVolumes[item]
                end
            end 
        end  
        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
end

function assemble!(
    A::AbstractSparseMatrix,
    form::Type{SymmetricBilinearForm},
    AT::Type{<:AbstractAssemblyType},
    operator::Type{<:AbstractFEFunctionOperator},
    FE::AbstractFiniteElement,
    action::AbstractAction;
    bonus_quadorder::Int = 0,
    talkative::Bool = false,
    regions::Array{Int} = [1])

    # collect grid information
    NumberType = eltype(A)
    xItemNodes = FE.xgrid[GridComponentNodes4AssemblyType(AT)]
    xItemVolumes = FE.xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemGeometries = FE.xgrid[GridComponentTypes4AssemblyType(AT)]
    xItemDofs = FEPropertyDofs4AssemblyType(FE,AT)
    xItemRegions = FE.xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = num_sources(xItemNodes)
    
    EG, ndofs4EG, qf, basisevaler, EG4item, dofitem4item, evaler4item = prepareOperatorAssembly(form, AT, operator, FE, regions, NumberType, 2, bonus_quadorder, talkative)

    # collect FE and FEBasisEvaluator information
    ncomponents = FiniteElements.get_ncomponents(typeof(FE))
    cvals_resultdim = size(basisevaler[1][1].cvals[1],1)

    # loop over items
    itemET = xItemGeometries[1] # type of the current item
    iEG = 1 # index to the correct unique geometry
    ndofs4item = 0 # number of dofs for item
    evalnr = 0 # evaler number that has to be used for current item
    dofitem = 0 # itemnr where the dof numbers can be found
    dofs = zeros(Int32,max_num_targets_per_source(xItemDofs))
    temp = 0 # some temporary variable
    action_input = zeros(NumberType,cvals_resultdim) # heap for action input
    action_result = zeros(NumberType,action.resultdim) # heap for action output
    cvali::Array{NumberType,2} = [[] []] # pointer to FEoperatorvalue at quadrature point i
    @time for item::Int32 = 1 : nitems
    for r = 1 : length(regions)
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        dofitem = dofitem4item(item)

        # find index for CellType
        itemET = EG4item(item)
        iEG = findfirst(isequal(itemET), EG)
        ndofs4item = ndofs4EG[iEG]

        # update FEbasisevaler
        evalnr = evaler4item(item)
        update!(basisevaler[iEG][evalnr],dofitem)

        # update action
        update!(action, basisevaler[iEG][evalnr], item)

        # update dofs
        dofs[1:ndofs4item] = xItemDofs[:,dofitem]

        for i in eachindex(qf[iEG].w)
            cvali = basisevaler[iEG][evalnr].cvals[i]
           
            for dof_i = 1 : ndofs4item
                # apply action to first argument
                for k = 1 : cvals_resultdim
                    action_input[k] = cvali[k,dof_i]
                end    
                apply_action!(action_result, action_input, action, i)

                for dof_j = 1 : ndofs4item
                    temp = 0
                    for k = 1 : action.resultdim
                        temp += action_result[k]*cvali[k,dof_j]
                    end
                    A[dofs[dof_i],dofs[dof_j]] += temp * qf[iEG].w[i] * xItemVolumes[item]
                end
            end 
        end 
        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
end

# We can also define some shortcuts

function RightHandSide!(FE,action; operator::Type{<:AbstractFEFunctionOperator} = Identity, talkative::Bool = false, bonus_quadorder::Int = 0)
    b = zeros(Float64,FE.ndofs,1)
    assemble!(b, LinearForm, AbstractAssemblyTypeCELL, operator, FE, action; talkative = talkative, bonus_quadorder = bonus_quadorder)
    return b
end    

function StiffnessMatrix!(FE,action; operator::Type{<:AbstractFEFunctionOperator} = Gradient, talkative::Bool = false, bonus_quadorder::Int = 0)
    ndofs = FE.ndofs
    A = ExtendableSparseMatrix{Float64,Int32}(ndofs,ndofs)
    FEOperator.assemble!(A, SymmetricBilinearForm, AbstractAssemblyTypeCELL, operator, FE, action; talkative = talkative, bonus_quadorder = bonus_quadorder)
    return A
end

    

end
