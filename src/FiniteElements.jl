module FiniteElements

using ExtendableGrids
using ExtendableSparse
using SparseArrays
using FEXGrid
using QuadratureRules
using ForwardDiff # for FEBasisEvaluator
using Printf


 #######################################################################################################
 #######################################################################################################
 ### FFFFF II NN    N II TTTTTT EEEEEE     EEEEEE LL     EEEEEE M     M EEEEEE NN    N TTTTTT SSSSSS ###
 ### FF    II N N   N II   TT   EE         EE     LL     EE     MM   MM EE     N N   N   TT   SS     ###
 ### FFFF  II N  N  N II   TT   EEEEE      EEEEE  LL     EEEEE  M M M M EEEEE  N  N  N   TT    SSSS  ###
 ### FF    II N   N N II   TT   EE         EE     LL     EE     M  M  M EE     N   N N   TT       SS ###
 ### FF    II N    NN II   TT   EEEEEE     EEEEEE LLLLLL EEEEEE M     M EEEEEE N    NN   TT   SSSSSS ###
 #######################################################################################################
 #######################################################################################################

 abstract type AbstractFiniteElement end
 export AbstractFiniteElement


include("FiniteElements_FESpaces.jl")
export FESpace

export AbstractH1FiniteElement
export getH1P1FiniteElement, getH1CRFiniteElement, getH1MINIFiniteElement, getH1P2FiniteElement

export AbstractH1FiniteElementWithCoefficients
export getH1BRFiniteElement

export AbstractHdivFiniteElement
export getHdivRT0FiniteElement

export AbstractHcurlFiniteElement

export get_ncomponents, reconstruct!

get_polynomialorder(::Type{<:AbstractFiniteElement}, ::Type{<:Vertex0D}) = 0;


include("FiniteElements_FEBlockArrays.jl");
export FEVectorBlock, FEVector
export FEMatrixBlock, FEMatrix, fill!



include("FiniteElements_FEBasisEvaluator.jl")
export FEBasisEvaluator, update!

export AbstractFunctionOperator
export Identity, ReconstructionIdentity
export NormalFlux, TangentFlux
export Gradient, SymmetricGradient, TangentialGradient
export Divergence
export Curl, Rotation
export Laplacian, Hessian
export Trace, Deviator

export NeededDerivatives4Operator, QuadratureOrderShift4Operator, FEPropertyDofs4AssemblyType
export DefaultDirichletBoundaryOperator4FE


include("FiniteElements_Interpolations.jl")
export interpolate! # must be defined separately by each FEdefinition
export nodevalues! # = P1interpolation, abstract averaging method that works for any element, but can be overwritten by FEdefinition to something simpler

end #module