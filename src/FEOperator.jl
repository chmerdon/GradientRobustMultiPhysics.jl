module FEOperator

using FiniteElements
using XGrid
using FEXGrid
using QuadratureRules
using ExtendableSparse
using SparseArrays

# would be usefull, but not working like this at the moment
#GridComponentDofs4AssemblyType(::Type{AbstractAssemblyTypeCELL}) = CellDofs
#GridComponentDofs4AssemblyType(::Type{AbstractAssemblyTypeFACE}) = FaceDofs
#GridComponentDofs4AssemblyType(::Type{AbstractAssemblyTypeBFACE}) = BFaceDofs

# instead currently, this is used:
FEPropertyDofs4AssemblyType(FE::AbstractFiniteElement,::Type{AbstractAssemblyTypeCELL}) = FE.CellDofs
FEPropertyDofs4AssemblyType(FE::AbstractFiniteElement,::Type{AbstractAssemblyTypeFACE}) = FE.FaceDofs
FEPropertyDofs4AssemblyType(FE::AbstractFiniteElement,::Type{AbstractAssemblyTypeBFACE}) = FE.BFaceDofs


abstract type AbstractFEFunctionOperator end # to dispatch which evaluator of the FE_basis_caller is used
abstract type Identity <: AbstractFEFunctionOperator end
abstract type Gradient <: AbstractFEFunctionOperator end
abstract type Curl <: AbstractFEFunctionOperator end
abstract type Divergence <: AbstractFEFunctionOperator end
abstract type SymmetricGradient <: AbstractFEFunctionOperator end
abstract type Trace <: AbstractFEFunctionOperator end
abstract type Jump <: AbstractFEFunctionOperator end
abstract type NormalJump <: AbstractFEFunctionOperator end
abstract type TangentJump <: AbstractFEFunctionOperator end

export AbstractFEFunctionOperator, Identity, Gradient, Curl, Divergence, SymmetricGradient, Trace, Jump, NormalJump, TangentialJump


include("FEBasisEvaluator.jl")
export FEBasisEvaluator, eval!, update!


abstract type AbstractFEOperator end
abstract type LinearForm <: AbstractFEOperator end
abstract type SymmetricBilinearForm <: AbstractFEOperator end
abstract type ASymmetricBilinearForm <: AbstractFEOperator end

export AbstractFEOperator,LinearForm,SymmetricBilinearForm,ASymmetricBilinearForm
export assemble!



function assemble!(b::Array{<:Real}, ::Type{LinearForm}, AT::Type{<:AbstractAssemblyType}, ::Type{<:AbstractFEFunctionOperator}, FE::AbstractFiniteElement, coefficient::Real = 1.0; bonus_quadorder::Int = 0, talkative::Bool = false)

    NumberType = eltype(b)
    xCoords = FE.xgrid[Coordinates]
    dim = size(xCoords,1)
    xItemNodes = FE.xgrid[GridComponentNodes4AssemblyType(AT)]
    xItemVolumes = FE.xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemTypes = FE.xgrid[GridComponentTypes4AssemblyType(AT)]
    xItemDofs = FEPropertyDofs4AssemblyType(FE,AT)
    nitems = num_sources(xItemNodes)
    
    # find proper quadrature rules
    EG = unique(xItemTypes)
    qf = Array{QuadratureRule,1}(undef,length(EG))
    basisevaler = Array{FEBasisEvaluator,1}(undef,length(EG))
    quadorder = 0
    for j = 1 : length(EG)
        quadorder = bonus_quadorder + FiniteElements.get_polynomialorder(typeof(FE), EG[j])
        qf[j] = QuadratureRule{NumberType,EG[j]}(quadorder);
        basisevaler[j] = FEBasisEvaluator{NumberType,typeof(FE),EG[j],Identity}(FE, qf[j], AT)
    end    
    if talkative
        println("ASSEMBLE_OPERATOR")
        println("=================")
        println("  type = LinearForm")
        println("nitems = $nitems")
        for j = 1 : length(EG)
            println("QuadratureRule [$j] for $(EG[j]):")
            QuadratureRules.show(qf[j])
        end
    end

    # loop over items
    itemET = xItemTypes[1]
    iEG = 1
    ndofs4item = 0
    dof = 0
    ncomponents = FiniteElements.get_ncomponents(typeof(FE))
    for item = 1 : nitems
        # find index for CellType
        itemET = xItemTypes[item]
        iEG = findfirst(isequal(itemET), EG)
        ndofs4item = num_targets(xItemDofs,item)

        # update FEbasisevaler
        update!(basisevaler[iEG],item)

        for i in eachindex(qf[iEG].w)

            # evaluate FEbasis at quadrature point
            eval!(basisevaler[iEG], i)

            for dof_i = 1 : ndofs4item
                for j = 1 : ncomponents
                    b[xItemDofs[dof_i,item],j] += coefficient * basisevaler[iEG].cvals[dof_i,j] * qf[iEG].w[i] * xItemVolumes[item]
                end
            end 
        end  
    end
end


function assemble!(A::AbstractSparseMatrix, ::Type{SymmetricBilinearForm}, AT::Type{<:AbstractAssemblyType}, ::Type{<:AbstractFEFunctionOperator}, FE::AbstractFiniteElement, coefficient::Real = 1.0; bonus_quadorder::Int = 0, talkative::Bool = false)

    NumberType = eltype(A)
    xCoords = FE.xgrid[Coordinates]
    dim = size(xCoords,1)
    xItemNodes = FE.xgrid[GridComponentNodes4AssemblyType(AT)]
    xItemVolumes = FE.xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemTypes = FE.xgrid[GridComponentTypes4AssemblyType(AT)]
    xItemDofs = FEPropertyDofs4AssemblyType(FE,AT)
    nitems = num_sources(xItemNodes)
    
    # find proper quadrature rules
    EG = unique(xItemTypes)
    qf = Array{QuadratureRule,1}(undef,length(EG))
    basisevaler = Array{FEBasisEvaluator,1}(undef,length(EG))
    quadorder = 0
    for j = 1 : length(EG)
        quadorder = bonus_quadorder + 2*FiniteElements.get_polynomialorder(typeof(FE), EG[j])
        qf[j] = QuadratureRule{NumberType,EG[j]}(quadorder);
        basisevaler[j] = FEBasisEvaluator{NumberType,typeof(FE),EG[j],Identity}(FE, qf[j], AT)
    end    
    if talkative
        println("ASSEMBLE_OPERATOR")
        println("=================")
        println("  type = SymmetricBilinearForm")
        println("nitems = $nitems")
        for j = 1 : length(EG)
            println("QuadratureRule [$j] for $(EG[j]):")
            QuadratureRules.show(qf[j])
        end
    end

    # loop over items
    itemET = xItemTypes[1]
    iEG = 1
    ndofs4item = 0
    dof = 0
    ncomponents = FiniteElements.get_ncomponents(typeof(FE))
    temp = 0
    Base.show(xItemDofs)
    for item = 1 : nitems
        # find index for CellType
        itemET = xItemTypes[item]
        iEG = findfirst(isequal(itemET), EG)
        ndofs4item = num_targets(xItemDofs,item)

        # update FEbasisevaler
        update!(basisevaler[iEG],item)

        for i in eachindex(qf[iEG].w)

            # evaluate FEbasis at quadrature point
            eval!(basisevaler[iEG], i)

            for dof_i = 1 : ndofs4item, dof_j = 1 : ndofs4item
                temp = 0
                for k = 1 : ncomponents
                    temp += basisevaler[iEG].cvals[dof_i,k]*basisevaler[iEG].cvals[dof_j,k]
                end
                A[xItemDofs[dof_i,item],xItemDofs[dof_j,item]] += coefficient * temp * qf[iEG].w[i] * xItemVolumes[item]
            end 
        end  
    end
end

end
