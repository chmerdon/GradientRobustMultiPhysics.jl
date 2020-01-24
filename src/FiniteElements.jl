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
    # subtype for Raviart-Thomas elements (only normal face fluxes)
    abstract type AbstractHdivRTFiniteElement <: AbstractHdivFiniteElement end
      include("FEdefinitions/HDIV_RT0.jl");
    # subtype for Brezzi-Douglas elements (different handling of tangential face fluxes?)
    abstract type AbstractHdivBDMFiniteElement <: AbstractHdivFiniteElement end
      # TODO

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
get_globaldof4cell(FE::AbstractFiniteElement, cell, N::Int64) = FiniteElements.get_globaldof4cell(FE, cell, dummy_array[N])

# H1 elements allow for continuous (= cell-independent) basis function evaluations on faces
get_globaldof4face(FE::AbstractFiniteElement, face, N::Int64) = FiniteElements.get_globaldof4face(FE, face, dummy_array[N])

# Hdiv elements should allow for continuous evaluations of normal-fluxes
# maybe a good idea to offer a function for this?
# 

# show function for FiniteElement
function show(FE::AbstractFiniteElement)
	nd = get_ndofs(FE);
	mdc = get_maxndofs4cell(FE);
	mdf = get_maxndofs4face(FE);
	po = get_polynomial_order(FE);

	println("FiniteElement information");
	println("         name : " * FE.name);
	println("        ndofs : $(nd)")
	println("    polyorder : $(po)");
	println("  maxdofs c/f : $(mdc)/$(mdf)")
end

function show_dofmap(FE::AbstractFiniteElement)
	println("FiniteElement cell dofmap for " * FE.name);
	for cell = 1 : size(FE.grid.nodes4cells,1)
        print(string(cell) * " | ");
        for j = 1 : FiniteElements.get_maxndofs4cell(FE)
            print(string(FiniteElements.get_globaldof4cell(FE, cell, j)) * " ");
        end
        println("");
	end
	
	println("\nFiniteElement face dofmap for " * FE.name);
	for face = 1 : size(FE.grid.nodes4faces,1)
        print(string(face) * " | ");
        for j = 1 : FiniteElements.get_maxndofs4face(FE)
            print(string(FiniteElements.get_globaldof4face(FE, face, j)) * " ");
        end
        println("");
	end
end

# creates a zero vector of the correct length for the FE
function createFEVector(FE::AbstractFiniteElement)
    return zeros(get_ndofs(FE));
end

# transformation for H1 elements on a line
function local2global_line()
    A = Matrix{Float64}(undef,1,1)
    b = Vector{Float64}(undef,1)
    x = Vector{Real}(undef,1)
    return function fix_cell(grid,cell)
        b[1] = grid.coords4nodes[grid.nodes4cells[cell,1],1]
        A[1,1] = grid.coords4nodes[grid.nodes4cells[cell,2],1] - b[1]
        return function closure(xref)
            x[1] = A[1,1]*xref[1] + b[1]
            return x
        end
    end    
end

# transformation for H1 elements on a triangle
function local2global_triangle()
    A = Matrix{Float64}(undef,2,2)
    b = Vector{Float64}(undef,2)
    x = Vector{Real}(undef,2)
    return function fix_cell(grid,cell)
        b[1] = grid.coords4nodes[grid.nodes4cells[cell,1],1]
        b[2] = grid.coords4nodes[grid.nodes4cells[cell,1],2]
        A[1,1] = grid.coords4nodes[grid.nodes4cells[cell,2],1] - b[1]
        A[1,2] = grid.coords4nodes[grid.nodes4cells[cell,3],1] - b[1]
        A[2,1] = grid.coords4nodes[grid.nodes4cells[cell,2],2] - b[2]
        A[2,2] = grid.coords4nodes[grid.nodes4cells[cell,3],2] - b[2]
        return function closure(xref)
            x[1] = A[1,1]*xref[1] + A[1,2]*xref[2] + b[1]
            x[2] = A[2,1]*xref[1] + A[2,2]*xref[2] + b[2]
            # faster than: x = A*xref + b
            return x
        end
    end    
end


# Piola transformation for Hdiv elements on a triangle
function Piola_triangle!(grid)
    det = 0.0;
    return function closure(A,cell)
        A[1,1] = grid.coords4nodes[grid.nodes4cells[cell,2],1] - grid.coords4nodes[grid.nodes4cells[cell,1],1]
        A[1,2] = grid.coords4nodes[grid.nodes4cells[cell,3],1] - grid.coords4nodes[grid.nodes4cells[cell,1],1]
        A[2,1] = grid.coords4nodes[grid.nodes4cells[cell,2],2] - grid.coords4nodes[grid.nodes4cells[cell,1],2]
        A[2,2] = grid.coords4nodes[grid.nodes4cells[cell,3],2] - grid.coords4nodes[grid.nodes4cells[cell,1],2]
        det = A[1,1]*A[2,2] - A[1,2]*A[2,1]
        return det
    end    
end


# exact tinversion of transformation for H1 elements on a triangle
function local2global_tinv_jacobian_triangle(A!,det!,grid,cell)
    # transposed inverse of A
    A![2,2] = grid.coords4nodes[grid.nodes4cells[cell,2],1] - grid.coords4nodes[grid.nodes4cells[cell,1],1]
    A![2,1] = -grid.coords4nodes[grid.nodes4cells[cell,3],1] + grid.coords4nodes[grid.nodes4cells[cell,1],1]
    A![1,2] = -grid.coords4nodes[grid.nodes4cells[cell,2],2] + grid.coords4nodes[grid.nodes4cells[cell,1],2]
    A![1,1] = grid.coords4nodes[grid.nodes4cells[cell,3],2] - grid.coords4nodes[grid.nodes4cells[cell,1],2]
        
    # divide by  determinant
    det! = A![1,1]*A![2,2] - A![1,2]*A![2,1]
    
    A![1] = A![1]/det!
    A![2] = A![2]/det!
    A![3] = A![3]/det!
    A![4] = A![4]/det!
end


# exact tinversion of transformation for H1 elements on a line
function local2global_tinv_jacobian_line(A!,det!,grid,cell)
    # transposed inverse of A
    det! = grid.coords4nodes[grid.nodes4cells[cell,2],1] - grid.coords4nodes[grid.nodes4cells[cell,1],1]
    A![1,1] = 1/det!
end



# transformation for H1 elements on a tetrahedron
function local2global_tetrahedron()
    A = Matrix{Float64}(undef,3,3)
    b = Vector{Float64}(undef,3)
    return function fix_cell(grid,cell)
        b[1] = grid.coords4nodes[grid.nodes4cells[cell,1],1]
        b[2] = grid.coords4nodes[grid.nodes4cells[cell,1],2]
        b[2] = grid.coords4nodes[grid.nodes4cells[cell,1],3]
        A[1,1] = grid.coords4nodes[grid.nodes4cells[cell,2],1] - b[1]
        A[1,2] = grid.coords4nodes[grid.nodes4cells[cell,3],1] - b[1]
        A[1,3] = grid.coords4nodes[grid.nodes4cells[cell,4],1] - b[1]
        A[2,1] = grid.coords4nodes[grid.nodes4cells[cell,2],2] - b[2]
        A[2,2] = grid.coords4nodes[grid.nodes4cells[cell,3],2] - b[2]
        A[2,3] = grid.coords4nodes[grid.nodes4cells[cell,4],2] - b[2]
        A[3,1] = grid.coords4nodes[grid.nodes4cells[cell,2],3] - b[3]
        A[3,2] = grid.coords4nodes[grid.nodes4cells[cell,3],3] - b[3]
        A[3,3] = grid.coords4nodes[grid.nodes4cells[cell,4],3] - b[3]
        return function closure(xref)
            x = A*xref + b
        end
    end    
end

end #module
