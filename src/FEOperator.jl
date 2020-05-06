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


abstract type AbstractFEVectorOperator end # to dispatch which evaluator of the FE_basis_caller is used
abstract type Identity <: AbstractFEVectorOperator end # 1*v_h
abstract type Gradient <: AbstractFEVectorOperator end # D_geom(v_h)
abstract type SymmetricGradient <: AbstractFEVectorOperator end # eps_geom(v_h)
abstract type Laplacian <: AbstractFEVectorOperator end # L_geom(v_h)
abstract type Hessian <: AbstractFEVectorOperator end # D^2(v_h)
abstract type Curl <: AbstractFEVectorOperator end # only 2D: Curl(v_h) = D(v_h)^\perp
abstract type Rotation <: AbstractFEVectorOperator end # only 3D: Rot(v_h) = D \times v_h
abstract type Divergence <: AbstractFEVectorOperator end # div(v_h)
abstract type Trace <: AbstractFEVectorOperator end # tr(v_h)
abstract type Deviator <: AbstractFEVectorOperator end # dev(v_h)


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

Length4Operator(::Type{Identity}, xdim::Int) = 1
Length4Operator(::Type{Divergence}, xdim::Int) = 1
Length4Operator(::Type{Trace}, xdim::Int) = 1
Length4Operator(::Type{Curl}, xdim::Int) = (xdim == 2) ? 1 : xdim
Length4Operator(::Type{Gradient}, xdim::Int) = xdim
Length4Operator(::Type{SymmetricGradient}, xdim::Int) = (xdim == 2) ? 3 : 6
Length4Operator(::Type{Hessian}, xdim::Int) = xdim*xdim

QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{Identity}) = 0
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{Gradient}) = -1
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{SymmetricGradient}) = -1
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{Laplacian}) = -2
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{Hessian}) = -2


export AbstractFEVectorOperator
export Identity, Gradient, SymmetricGradient, Laplacian, Hessian, Curl, Rotation, Divergence, Trace, Deviator
export NeededDerivatives4Operator

include("FEOperator_FEBasisEvaluator.jl")
export FEBasisEvaluator, update!

include("FEOperator_AbstractAction.jl")
export DoNothingAction
export MultiplyScalarAction
export MultiplyVectorAction
export MultiplyMatrixAction
export RegionWiseMultiplyScalarAction
export RegionWiseMultiplyVectorAction
export FunctionAction
export XFunctionAction
export RegionWiseXFunctionAction


abstract type AbstractAssemblyForm{AT<:AbstractAssemblyType} end
# for each v_h: F(v_h) = Int_regions action(v_h) dx
struct LinearForm{AT} <: AbstractAssemblyForm{AT}
    FE::AbstractFiniteElement
    operator::Type{<:AbstractFEVectorOperator}
    action::AbstractAction
    bonus_quadorder::Int
    regions::Array{Int,1}
end   

function LinearForm(AT::Type{<:AbstractAssemblyType},
    FE::AbstractFiniteElement,
    operator::Type{<:AbstractFEVectorOperator},
    action::AbstractAction;
    bonus_quadorder::Int = 0,
    regions::Array{Int,1} = [0])
    return LinearForm{AT}(FE,operator,action,bonus_quadorder,regions)
end

abstract type AbstractFEForm end
# for each item: I(item) = Int_item action(FEVector) dx
abstract type ItemIntegrals <: AbstractFEForm end     

# for each v_h,u_h from same FE: A(v_h,u_h) = sum_item Int_item action(u_h) dot v_h dx
abstract type SymmetricBilinearForm <: AbstractFEForm end 
# todo: for each v_h,u_h: B(v_h,u_h) = sum_item Int_item action(u_h) dot action(v_h) dx
abstract type MixedBilinearForm <: AbstractFEForm end 

export AbstractFEForm,ItemIntegrals,LinearForm,SymmetricBilinearForm,ASymmetricBilinearForm
export assemble!
export L2Error


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


function prepareOperatorAssembly(form::AbstractAssemblyForm{AT}, operator::Type{<:AbstractFEVectorOperator}, FE::AbstractFiniteElement, regions::Array{Int32,1}, NumberType::Type{<:Real}, nrfactors::Int, bonus_quadorder::Int, talkative::Bool) where AT <: AbstractAssemblyType
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
        println("\nASSEMBLY PREPARATION $form")
        println("====================================")
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

function prepareOperatorAssembly(form::Type{<:AbstractFEForm}, AT::Type{<:AbstractAssemblyType}, operator::Type{<:AbstractFEVectorOperator}, FE::AbstractFiniteElement, regions::Array{Int32,1}, NumberType::Type{<:Real}, nrfactors::Int, bonus_quadorder::Int, talkative::Bool)
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
        println("\nASSEMBLY PREPARATION $form")
        println("====================================")
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

function prepareOperatorAssembly(form::Type{<:AbstractFEForm}, AT::Type{<:AbstractAssemblyTypeBFACECELL}, operator::Type{<:AbstractFEVectorOperator}, FE::AbstractFiniteElement, regions::Array{Int32,1}, NumberType::Type{<:Real}, nrfactors::Int, bonus_quadorder::Int, talkative::Bool)
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
        println("\nASSEMBLY PREPARATION $form")
        println("====================================")
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
    operator::Type{<:AbstractFEVectorOperator},
    FEB::FEVectorBlock,
    action::AbstractAction;
    bonus_quadorder::Int = 0,
    talkative::Bool = false,
    regions::Array{Int,1} = [0])

    NumberType = eltype(b)
    FE = FEB.FEType
    xItemNodes = FE.xgrid[GridComponentNodes4AssemblyType(AT)]
    xItemVolumes = FE.xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemGeometries = FE.xgrid[GridComponentTypes4AssemblyType(AT)]
    xItemDofs = FEPropertyDofs4AssemblyType(FE,AT)
    xItemRegions = FE.xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = num_sources(xItemNodes)

    if regions == [0]
        try
            regions = Array{Int32,1}(Base.unique(xItemRegions[:]))
        catch
            regions = [xItemRegions[1]]
        end        
    else
        regions = Array{Int32,1}(regions)    
    end
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
            # apply action to FEVector
            fill!(action_input,0)
            for dof_i = 1 : ndofs4item
                for k = 1 : cvals_resultdim
                    action_input[k] += FEB[dofs[dof_i]] * cvali[k,dof_i]
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

function assemble!(b::AbstractArray{<:Real,2}, LF::LinearForm{AT}; talkative::Bool = false) where AT <: AbstractAssemblyType
    FE = LF.FE
    operator = LF.operator
    action = LF.action
    bonus_quadorder = LF.bonus_quadorder
    regions = LF.regions

    NumberType = eltype(b)
    xItemNodes = FE.xgrid[GridComponentNodes4AssemblyType(AT)]
    xItemVolumes = FE.xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemGeometries = FE.xgrid[GridComponentTypes4AssemblyType(AT)]
    xItemDofs = FEPropertyDofs4AssemblyType(FE,AT)
    xItemRegions = FE.xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = num_sources(xItemNodes)
    
    if regions == [0]
        try
            regions = Array{Int32,1}(Base.unique(xItemRegions[:]))
        catch
            regions = [xItemRegions[1]]
        end        
    else
        regions = Array{Int32,1}(regions)    
    end
    EG, ndofs4EG, qf, basisevaler, EG4item, dofitem4item, evaler4item = prepareOperatorAssembly(LF, operator, FE, regions, NumberType, 1, bonus_quadorder, talkative)

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
    A::AbstractArray{<:Real,2},
    form::Type{SymmetricBilinearForm},
    AT::Type{<:AbstractAssemblyType},
    operator::Type{<:AbstractFEVectorOperator},
    FE::AbstractFiniteElement,
    action::AbstractAction;
    bonus_quadorder::Int = 0,
    talkative::Bool = false,
    regions::Array{Int,1} = [0])

    # collect grid information
    NumberType = eltype(A)
    xItemNodes = FE.xgrid[GridComponentNodes4AssemblyType(AT)]
    xItemVolumes = FE.xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemGeometries = FE.xgrid[GridComponentTypes4AssemblyType(AT)]
    xItemDofs = FEPropertyDofs4AssemblyType(FE,AT)
    xItemRegions = FE.xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = num_sources(xItemNodes)
    
    if regions == [0]
        try
            regions = Array{Int32,1}(Base.unique(xItemRegions[:]))
        catch
            regions = [xItemRegions[1]]
        end        
    else
        regions = Array{Int32,1}(regions)    
    end
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


function L2Error(discrete_function::FEVectorBlock{<:Real}, exact_function::Function, operator::Type{<:AbstractFEVectorOperator}; talkative::Bool = false, bonus_quadorder::Int = 0)
    function L2error_function(result,input,x)
        exact_function(result,x)
        result[1] = (result[1] - input[1])^2
        for j=2:length(input)
            result[1] += (result[j] - input[j])^2
        end    
        for j=2:length(result)
            result[2] = 0.0
        end    
    end    
    dim = size(discrete_function.FEType.xgrid[Coordinates],1)
    L2error_action = XFunctionAction(L2error_function,Length4Operator(operator,dim)*get_ncomponents(typeof(discrete_function.FEType)),dim)
    error4cell = zeros(Float64,num_sources(discrete_function.FEType.xgrid[CellNodes]),L2error_action.resultdim)
    assemble!(error4cell, ItemIntegrals, AbstractAssemblyTypeCELL, operator, discrete_function, L2error_action; talkative = talkative, bonus_quadorder = bonus_quadorder)

    error = 0.0;
    for j=1 : size(error4cell,1)
        error += error4cell[j,1]
    end
    return sqrt(error)
end

    

end
