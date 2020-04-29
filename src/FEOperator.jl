module FEOperator

using FiniteElements
using ExtendableGrids
using FEXGrid
using QuadratureRules
using ExtendableSparse
using SparseArrays
using ForwardDiff

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

include("AbstractCoefficient.jl")
export ConstantCoefficient
export RegionWiseConstantCoefficient
export FunctionCoefficient



abstract type AbstractFEForm end
abstract type LinearForm <: AbstractFEForm end
abstract type SymmetricBilinearForm <: AbstractFEForm end
abstract type ASymmetricBilinearForm <: AbstractFEForm end

export AbstractFEForm,LinearForm,SymmetricBilinearForm,ASymmetricBilinearForm
export assemble!


# unique functions that only selects uniques in specified regions
function unique(xItemGeometries, xItemRegions, regions)
    nitems = 0
    try
        nitems = num_sources(xItemGeometries)
    catch
        nitems = length(xItemGeometries)
    end      
    #EG = unique(xItemGeometries)
    EG = []
    for item = 1 : nitems
        if indexin(xItemRegions[item], regions) != Nothing
            if findfirst(isequal(xItemGeometries[item]), EG) == nothing
                append!(EG, [xItemGeometries[item]])
            end    
        end
    end    
    return EG
end

function prepareOperatorAssembly(form::Type{<:AbstractFEForm}, AT::Type{<:AbstractAssemblyType}, operator::Type{<:AbstractFEFunctionOperator}, FE::AbstractFiniteElement, regions::Array{Int}, NumberType::Type{<:Real}, nrfactors::Int, bonus_quadorder::Int, talkative::Bool)
    xItemGeometries = FE.xgrid[GridComponentTypes4AssemblyType(AT)]
    xItemRegions = FE.xgrid[GridComponentRegions4AssemblyType(AT)]

    # find unique ElementGeometries
    EG = unique(xItemGeometries, xItemRegions, regions)

    # find proper quadrature QuadratureRules
    # and construct matching FEBasisEvaluators
    qf = Array{QuadratureRule,1}(undef,length(EG))
    basisevaler = Array{Array{FEBasisEvaluator,1},1}(undef,length(EG))
    quadorder = 0
    for j = 1 : length(EG)
        basisevaler[j] = Array{FEBasisEvaluator,1}(undef,1)
        quadorder = bonus_quadorder + nrfactors*FiniteElements.get_polynomialorder(typeof(FE), EG[j])
        qf[j] = QuadratureRule{NumberType,EG[j]}(quadorder);
        basisevaler[j][1] = FEBasisEvaluator{NumberType,typeof(FE),EG[j],operator,AT}(FE, qf[j])
    end        
    if talkative
        println("\nASSEMBLY PREPARATION")
        println("=====================")
        println("      form = $form")
        println("  operator = $operator")
        println("   regions = $regions")
        println("  uniqueEG = $EG")
        for j = 1 : length(EG)
            println("\nQuadratureRule [$j] for $(EG[j]):")
            QuadratureRules.show(qf[j])
        end
    end
    return EG, qf, basisevaler, (item) -> xItemGeometries[item], (item) -> item, (item) -> 1
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
        println("  operator = $operator")
        println("   regions = $regions")
        println("  uniqueEG = $EG")
        for j = 1 : length(EG)
            println("\nQuadratureRule [$j] for $(EG[j]):")
            QuadratureRules.show(qf[j])
        end
    end
    return EG, qf, basisevaler, (item) -> xCellGeometries[FE.xgrid[FaceCells][1,FE.xgrid[BFaces][item]]] ,(item) -> FE.xgrid[FaceCells][1,FE.xgrid[BFaces][item]], (item) -> FE.xgrid[BFaceCellPos][item]
end


function assemble!(
    b::Array{<:Real},
    form::Type{LinearForm},
    AT::Type{<:AbstractAssemblyType},
    operator::Type{<:AbstractFEFunctionOperator},
    FE::AbstractFiniteElement,
    coefficient::AbstractCoefficient;
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
    
    EG, qf, basisevaler, EG4item, dofitem4item, evaler4item = prepareOperatorAssembly(form, AT, operator, FE, regions, NumberType, 1, bonus_quadorder, talkative)

    # collect FE and FEBasisEvaluator information
    ncomponents = FiniteElements.get_ncomponents(typeof(FE))
    cvals_resultdim = size(basisevaler[1][1].cvals[1],2)
    @assert size(b,2) == cvals_resultdim

    # loop over items
    itemET = xItemGeometries[1] # type of the current item
    iEG = 1 # index to the correct unique geometry
    ndofs4item = 0 # number of dofs for item
    evalnr = 0 # evaler number that has to be used for current item
    dofitem = 0 # itemnr where the dof numbers can be found
    temp = 0 # some temporary variable
    coeff_input = zeros(NumberType,cvals_resultdim) # heap for coefficient input
    coeff_result = zeros(NumberType,coefficient.resultdim) # heap for coefficient output
    for item::Int32 = 1 : nitems
    # check if item region is in regions
    if indexin(xItemRegions[item], regions) != Nothing

        # item number for dof provider
        dofitem = dofitem4item(item)

        # find index for CellType
        itemET = EG4item(item)
        iEG = findfirst(isequal(itemET), EG)
        ndofs4item = num_targets(xItemDofs,dofitem)

        # update FEbasisevaler
        evalnr = evaler4item(item)
        update!(basisevaler[iEG][evalnr],dofitem)

        # update coefficient
        update!(coefficient, basisevaler[iEG][evalnr], item)

        for i in eachindex(qf[iEG].w)
            for dof_i = 1 : ndofs4item
                # apply coefficient
                for k = 1 : cvals_resultdim
                    coeff_input[k] = basisevaler[iEG][evalnr].cvals[i][dof_i,k]
                end    
                apply_coefficient!(coeff_result, coeff_input, coefficient, i)

                for j = 1 : cvals_resultdim
                    b[xItemDofs[dof_i,dofitem],j] += coeff_result[j] * qf[iEG].w[i] * xItemVolumes[item]
                end
            end 
        end  
    end # if in region    
    end # for loop
end

function assemble!(A::AbstractSparseMatrix, form::Type{SymmetricBilinearForm}, AT::Type{<:AbstractAssemblyType}, operator::Type{<:AbstractFEFunctionOperator}, FE::AbstractFiniteElement, coefficient::AbstractCoefficient; bonus_quadorder::Int = 0, talkative::Bool = false, regions::Array{Int} = [1])

    # collect grid information
    NumberType = eltype(A)
    xItemNodes = FE.xgrid[GridComponentNodes4AssemblyType(AT)]
    xItemVolumes = FE.xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemGeometries = FE.xgrid[GridComponentTypes4AssemblyType(AT)]
    xItemDofs = FEPropertyDofs4AssemblyType(FE,AT)
    xItemRegions = FE.xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = num_sources(xItemNodes)
    
    EG, qf, basisevaler, EG4item, dofitem4item, evaler4item = prepareOperatorAssembly(form, AT, operator, FE, regions, NumberType, 2, bonus_quadorder, talkative)

    # collect FE and FEBasisEvaluator information
    ncomponents = FiniteElements.get_ncomponents(typeof(FE))
    cvals_resultdim = size(basisevaler[1][1].cvals[1],2)

    # loop over items
    itemET = xItemGeometries[1] # type of the current item
    iEG = 1 # index to the correct unique geometry
    ndofs4item = 0 # number of dofs for item
    evalnr = 0 # evaler number that has to be used for current item
    dofitem = 0 # itemnr where the dof numbers can be found
    temp = 0 # some temporary variable
    coeff_input = zeros(NumberType,cvals_resultdim) # heap for coefficient input
    coeff_result = zeros(NumberType,coefficient.resultdim) # heap for coefficient output
    for item::Int32 = 1 : nitems
    # check if item region is in regions
    if indexin(xItemRegions[item], regions) != Nothing

        dofitem = dofitem4item(item)

        # find index for CellType
        itemET = EG4item(item)
        iEG = findfirst(isequal(itemET), EG)
        ndofs4item = num_targets(xItemDofs,dofitem)

        # update FEbasisevaler
        evalnr = evaler4item(item)
        update!(basisevaler[iEG][evalnr],dofitem)

        # update coefficient
        update!(coefficient, basisevaler[iEG][evalnr], item)

        for i in eachindex(qf[iEG].w)
           
            for dof_i = 1 : ndofs4item

                # apply coefficient to first argument
                for k = 1 : cvals_resultdim
                    coeff_input[k] = basisevaler[iEG][evalnr].cvals[i][dof_i,k]
                end    
                apply_coefficient!(coeff_result, coeff_input, coefficient, i)

                for dof_j = 1 : ndofs4item
                    temp = 0
                    for k = 1 : coefficient.resultdim
                        temp += coeff_result[k]*basisevaler[iEG][evalnr].cvals[i][dof_j,k]
                    end
                    A[xItemDofs[dof_i,dofitem],xItemDofs[dof_j,dofitem]] += temp * qf[iEG].w[i] * xItemVolumes[item]
                end
            end 
        end 
    end # if in region    
    end # for loop
end

end
