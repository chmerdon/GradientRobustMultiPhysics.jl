  # subtype for Hdiv-conforming elements
  abstract type AbstractHdivFiniteElement <: AbstractFiniteElement end
    # lowest order
    include("FEdefinitions/Hdiv_RT0.jl");

    # second order
    #include("FEdefinitions/HDIV_RT1.jl");
    #include("FEdefinitions/HDIV_BDM1.jl");

  # subtype for H1 conforming elements (also Crouzeix-Raviart)
  abstract type AbstractH1FiniteElement <: AbstractFiniteElement end
    # lowest order
    include("FEdefinitions/H1_P1.jl");
    include("FEdefinitions/H1_MINI.jl");
    #include("FEdefinitions/H1_CR.jl");
    # second order
    include("FEdefinitions/H1_P2.jl");
    #include("FEdefinitions/H1_P2B.jl");

    abstract type AbstractH1FiniteElementWithCoefficients <: AbstractH1FiniteElement end
      include("FEdefinitions/H1v_BR.jl");

 
  # subtype for L2 conforming elements
  abstract type AbstractL2FiniteElement <: AbstractFiniteElement end
    include("FEdefinitions/L2_P0.jl"); # currently masked as H1 element
    #include("FEdefinitions/L2_P1.jl");
 
  # subtype for Hcurl-conforming elements
  abstract type AbstractHcurlFiniteElement <: AbstractFiniteElement end
    # TODO

# show function for FiniteElement
function show(FE::AbstractFiniteElement)
	println("\nFiniteElement information")
	println("=========================")
	println("   name = " * FE.name)
	println("  ndofs = $(FE.ndofs)")
end