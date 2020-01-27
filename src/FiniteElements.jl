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
function set_basis_coefficients_on_cell!(coefficients, FE::AbstractFiniteElement, cell::Int64)
    # default is one, replace if needed for special FE
    fill!(coefficients,1.0)
end    
function set_basis_coefficients_on_face!(coefficients, FE::AbstractFiniteElement, face::Int64)
    # default is one, replace if needed for special FE
    fill!(coefficients,1.0)
end    

function Hdivreconstruction_available(FE::AbstractFiniteElement)
    return false
end


# mapper for Int{N} to Val{N}
# dummy_array needed to avoid constructing Val{N} in each call
# length of dummy array corressponds to be longest number of dofs
dummy_array = [Val(1),Val(2),Val(3),Val(4),Val(5),Val(6),Val(7),Val(8),Val(9),Val(10),Val(11),Val(12)];

# all elements allow for evaluation of basis functions on cells
#get_globaldof4cell(FE::AbstractFiniteElement, cell, N::Int64) = FiniteElements.get_globaldof4cell(FE, cell, dummy_array[N])

# H1 elements allow for continuous (= cell-independent) basis function evaluations on faces
#get_globaldof4face(FE::AbstractFiniteElement, face, N::Int64) = FiniteElements.get_globaldof4face(FE, face, dummy_array[N])

# Hdiv elements should allow for continuous evaluations of normal-fluxes
# maybe a good idea to offer a function for this?
# 

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

# # transformation for H1 elements on a tetrahedron
# should go to ELemType3DTetrahedron later
# function local2global_tetrahedron()
#     A = Matrix{Float64}(undef,3,3)
#     b = Vector{Float64}(undef,3)
#     return function fix_cell(grid,cell)
#         b[1] = grid.coords4nodes[grid.nodes4cells[cell,1],1]
#         b[2] = grid.coords4nodes[grid.nodes4cells[cell,1],2]
#         b[2] = grid.coords4nodes[grid.nodes4cells[cell,1],3]
#         A[1,1] = grid.coords4nodes[grid.nodes4cells[cell,2],1] - b[1]
#         A[1,2] = grid.coords4nodes[grid.nodes4cells[cell,3],1] - b[1]
#         A[1,3] = grid.coords4nodes[grid.nodes4cells[cell,4],1] - b[1]
#         A[2,1] = grid.coords4nodes[grid.nodes4cells[cell,2],2] - b[2]
#         A[2,2] = grid.coords4nodes[grid.nodes4cells[cell,3],2] - b[2]
#         A[2,3] = grid.coords4nodes[grid.nodes4cells[cell,4],2] - b[2]
#         A[3,1] = grid.coords4nodes[grid.nodes4cells[cell,2],3] - b[3]
#         A[3,2] = grid.coords4nodes[grid.nodes4cells[cell,3],3] - b[3]
#         A[3,3] = grid.coords4nodes[grid.nodes4cells[cell,4],3] - b[3]
#         return function closure(xref)
#             x = A*xref + b
#         end
#     end    
# end

end #module
