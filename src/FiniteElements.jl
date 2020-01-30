module FiniteElements

using Grid
using LinearAlgebra

export AbstractFiniteElement, AbstractH1FiniteElement, AbstractHdivFiniteElement, AbstractHdivRTFiniteElement, AbstractHdivBDMFiniteElement, AbstractHcurlFiniteElement

 #######################################################################################################
 #######################################################################################################
 ### FFFFF II NN    N II TTTTTT EEEEEE     EEEEEE LL     EEEEEE M     M EEEEEE NN    N TTTTTT SSSSSS ###
 ### FF    II N N   N II   TT   EE         EE     LL     EE     MM   MM EE     N N   N   TT   SS     ###
 ### FFFF  II N  N  N II   TT   EEEEE      EEEEE  LL     EEEEE  M M M M EEEEE  N  N  N   TT    SSSS  ###
 ### FF    II N   N N II   TT   EE         EE     LL     EE     M  M  M EE     N   N N   TT       SS ###
 ### FF    II N    NN II   TT   EEEEEE     EEEEEE LLLLLL EEEEEE M     M EEEEEE N    NN   TT   SSSSSS ###
 #######################################################################################################
 #######################################################################################################

# top level abstract type
abstract type AbstractFiniteElement end

  # subtype for H1 conforming elements (also Crouzeix-Raviart)
  abstract type AbstractH1FiniteElement <: AbstractFiniteElement end
    include("FEdefinitions/H1_P1.jl");
    include("FEdefinitions/H1_MINI.jl");
    include("FEdefinitions/H1_P2.jl");
    include("FEdefinitions/H1_CR.jl");
    include("FEdefinitions/H1_BR.jl");
 
  # subtype for L2 conforming elements
  abstract type AbstractL2FiniteElement <: AbstractFiniteElement end
    include("FEdefinitions/L2_P0.jl");
 
  # subtype for Hdiv-conforming elements
  abstract type AbstractHdivFiniteElement <: AbstractFiniteElement end
    include("FEdefinitions/HDIV_RT0.jl");

  # subtype for Hcurl-conforming elements
  abstract type AbstractHcurlFiniteElement <: AbstractFiniteElement end
    # TODO


# basis function updates on cells and faces (to multiply e.g. by normal vector or reconstruction coefficients etc.)
function get_basis_coefficients_on_cell!(coefficients, FE::AbstractFiniteElement, cell::Int64, ::Grid.AbstractElemType)
    # default is one, replace if needed for special FE
    fill!(coefficients,1.0)
end    
function get_basis_coefficients_on_face!(coefficients, FE::AbstractFiniteElement, face::Int64, ::Grid.AbstractElemType)
    # default is one, replace if needed for special FE
    fill!(coefficients,1.0)
end    

function Hdivreconstruction_available(FE::AbstractFiniteElement)
    return false
end


# show function for FiniteElement
function show(FE::AbstractFiniteElement)
    nd = get_ndofs(FE);
    ET = FE.grid.elemtypes[1];
	mdc = get_ndofs4elemtype(FE,ET);
	mdf = get_ndofs4elemtype(FE,Grid.get_face_elemtype(ET));
	po = get_polynomial_order(FE);

	println("FiniteElement information");
	println("         name : " * FE.name);
	println("        ndofs : $(nd)")
	println("    polyorder : $(po)");
	println("  maxdofs c/f : $(mdc)/$(mdf)")
end

function show_dofmap(FE::AbstractFiniteElement)
    println("FiniteElement cell dofmap for " * FE.name);
    ET = FE.grid.elemtypes[1];
    ETF = Grid.get_face_elemtype(ET);
    ndofs4cell = FiniteElements.get_ndofs4elemtype(FE, ET)
    dofs = zeros(Int64,ndofs4cell)
	for cell = 1 : size(FE.grid.nodes4cells,1)
        print(string(cell) * " | ");
        FiniteElements.get_dofs_on_cell!(dofs, FE, cell, ET)
        for j = 1 : ndofs4cell
            print(string(dofs[j]) * " ");
        end
        println("");
	end
	
	println("\nFiniteElement face dofmap for " * FE.name);
    ndofs4face = FiniteElements.get_ndofs4elemtype(FE, ETF)
    dofs2 = zeros(Int64,ndofs4face)
	for face = 1 : size(FE.grid.nodes4faces,1)
        print(string(face) * " | ");
        FiniteElements.get_dofs_on_face!(dofs2, FE, face, ETF)
        for j = 1 : ndofs4face
            print(string(dofs2[j]) * " ");
        end
        println("");
	end
end

# creates a zero vector of the correct length for the FE
function createFEVector(FE::AbstractFiniteElement)
    return zeros(get_ndofs(FE));
end

end #module
