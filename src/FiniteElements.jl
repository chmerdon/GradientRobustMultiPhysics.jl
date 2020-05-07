module FiniteElements

using ExtendableGrids
using ExtendableSparse
using FEXGrid


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

  # subtype for Hdiv-conforming elements
  abstract type AbstractHdivFiniteElement <: AbstractFiniteElement end
    # lowest order
    #include("FEdefinitions/HDIV_RT0.jl");
    #include("FEdefinitions/HDIV_ABF0.jl");

    # second order
    #include("FEdefinitions/HDIV_RT1.jl");
    #include("FEdefinitions/HDIV_BDM1.jl");

  # subtype for H1 conforming elements (also Crouzeix-Raviart)
  abstract type AbstractH1FiniteElement <: AbstractFiniteElement end
    # lowest order
    include("FEdefinitions/H1_P1.jl");
    #include("FEdefinitions/H1_Q1.jl");
    #include("FEdefinitions/H1_MINI.jl");
    #include("FEdefinitions/H1_CR.jl");
    # second order
    include("FEdefinitions/H1_P2.jl");
    #include("FEdefinitions/H1_P2B.jl");

    abstract type AbstractH1FiniteElementWithCoefficients <: AbstractH1FiniteElement end
    #   include("FEdefinitions/H1_BR.jl");

 
  # subtype for L2 conforming elements
  abstract type AbstractL2FiniteElement <: AbstractFiniteElement end
    #include("FEdefinitions/L2_P0.jl");
    #include("FEdefinitions/L2_P1.jl");
 
  # subtype for Hcurl-conforming elements
  abstract type AbstractHcurlFiniteElement <: AbstractFiniteElement end
    # TODO

export AbstractFiniteElement, AbstractH1FiniteElementWithCoefficients, AbstractH1FiniteElement, AbstractHdivFiniteElement, AbstractHcurlFiniteElement
export get_ncomponents, getH1P1FiniteElement, getH1P2FiniteElement
export interpolate!

# show function for FiniteElement
function show(FE::AbstractFiniteElement)
	println("\nFiniteElement information")
	println("=========================")
	println("   name = " * FE.name)
	println("  ndofs = $(FE.ndofs)")
end


include("FEBlockArrays.jl");
export FEVectorBlock, FEVector, FEMatrix

function interpolate!(Target::FEVectorBlock, exact_function!::Function; dofs = [], verbosity::Int = 0)
    if verbosity > 0
        println("\nINTERPOLATING")
        println("=============")
        println("     target = $(Target.name)")
        println("         FE = $(Target.FEType.name) (ndofs = $(Target.FEType.ndofs))")
    end
    interpolate!(Target, Target.FEType, exact_function!; dofs = dofs)
end


end #module
