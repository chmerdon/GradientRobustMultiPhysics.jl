module FiniteElements

using Grid
using LinearAlgebra
using ForwardDiff


# Finite Element structure
# = container for set of basis functions and their gradients
#   local dof numbers, coordinates for dofs etc.
#
# todo/ideas:
# - broken elements (just replace dofs4cells, how to handle dofs4faces?)
# - define dual functionals for dofs?
# - Hdiv spaces (RT, BDM)
# - Hcurl spaces (Nedelec)
# - elasticity (Kouhia-Stenberg)
# - Stokes (P2B)
# - reconstruction (2nd set of bfun and bfun_grad? precompute coefficients?)
struct FiniteElement{T <: Real}
    name::String;                 # name of finite element (used in messages)
    grid::Grid.Mesh;              # link to grid
    polynomial_order::Int;        # polyonomial degree of basis functions (used for quadrature)
    ncomponents::Int;             # length of return value ofr basis functions, 1 = scalar, >1 vector-valued
    ndofs::Int;                   # total number of degrees of freedom
    dofs4cells::Array{Int64,2};   # dof numbers for each cell
    dofs4faces::Array{Int64,2};   # dof numbers for each face
    xref4dofs4cell::Array{T,2};   # coordinates for degrees of freedom
    loc2glob_trafo::Function;     # local2global trafo calculation (or should this better be part of grid?)
    bfun_ref::Vector{Function};   # basis functions evaluated in local coordinates
    bfun_grad!::Vector{Function}; # gradients of basis functions (either exactly given, or ForwardDiff of bfun) matrix
end

# FE into broken pieces
# by rewriting celldofs and facedofs
function BreakFEIntoPieces(FE)
    ndofs = prod(size(FE.dofs4cells));
    dofs4cells = zeros(Int64,size(FE.dofs4cells));
    dofs4faces = zeros(Int64,size(FE.grid.nodes4faces,1),0)
    dofs4cells[:] = 1:ndofs;
    return FiniteElement(FE.name * " (broken)", FE.grid, FE.polynomial_order, FE.ncomponents, ndofs, dofs4cells, dofs4faces, FE.xref4dofs4cell, FE.loc2glob_trafo, FE.bfun_ref, FE.bfun_grad!);
end
   
# wrapper for ForwardDiff & DiffResults
function FDgradient(bfun::Function, x::Vector{T}, xdim = 1) where T <: Real
    if xdim == 1
        DRresult = DiffResults.GradientResult(Vector{T}(undef, length(x)));
    else
        DRresult = DiffResults.DiffResult(Vector{T}(undef, length(x)),Matrix{T}(undef,length(x),xdim));
    end
    function closure(result,x,xref,grid,cell)
        f(a) = bfun(a,grid,cell);
        if xdim == 1
            ForwardDiff.gradient!(DRresult,f,x);
        else
            ForwardDiff.jacobian!(DRresult,f,x);
        end    
        result[:] = DiffResults.gradient(DRresult);
    end    
end


function local2global_line()
    A = Matrix{Float64}(undef,1,1)
    b = Vector{Float64}(undef,1)
    return function fix_cell(grid,cell)
        b[1] = grid.coords4nodes[grid.nodes4cells[cell,1],1]
        A[1,1] = grid.coords4nodes[grid.nodes4cells[cell,2],1] - b[1]
        return function closure(xref)
            x = A*xref + b
        end
    end    
end

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


function fast_inv_and_transpose_2D!()
    det = 0.0
    temp = 0.0;
    return function closure!(A::Matrix)
        # compute determinant
        det = (A[1,1]*A[2,2] - A[1,2]*A[2,1])
        # invert and transpose
        temp = A[2,2];
        A[2,2] = A[1,1]/det
        A[1,1] = temp/det
        temp = A[2,1]
        A[2,1] = -A[1,2]/det
        A[1,2] = -temp/det
    end  
end



function FDgradient2(loc2glob_trafo::Function, bfun::Function, x::Vector{T}, ncomponents = 1) where T <: Real
    xdim = length(x)
    if ncomponents == 1
        DRresult1 = DiffResults.GradientResult(Vector{T}(undef, xdim));
    else
        DRresult1 = DiffResults.DiffResult(Vector{T}(undef, xdim),Matrix{T}(undef,xdim,xdim));
    end
    DRresult2 = DiffResults.DiffResult(Vector{T}(undef, xdim),Matrix{T}(undef,xdim,xdim))
    if xdim == 2
        tinv = fast_inv_and_transpose_2D!()
    end    
    offset = 0
    function closure(result,xref,grid,cell)
        # compute derivative of glob2loc_trafo
        ForwardDiff.jacobian!(DRresult2,loc2glob_trafo(grid,cell),xref);
        # transpose and invert matrix by own function (todo: 1D, 3D)
        tinv(DiffResults.gradient(DRresult2))
        # compute derivative of f evaluated at xref and multiply
        f(a) = bfun(a,grid,cell);
        if ncomponents == 1
            ForwardDiff.gradient!(DRresult1,f,xref);
            result[:] = DiffResults.gradient(DRresult2)*DiffResults.gradient(DRresult1)
        else
            ForwardDiff.jacobian!(DRresult1,f,xref);
            offset = 0
            fill!(result,0.0)
            for j=1:ncomponents
                for k=1:xdim
                    for l=1:xdim
                        result[offset+k] += DiffResults.gradient(DRresult2)[k,l]*DiffResults.gradient(DRresult1)[j,l];
                    end    
                end  
                offset += length(x)
            end   
        end
    end    
end    
                  
   
 #######################################################################################################
 #######################################################################################################
 ### FFFFF II NN    N II TTTTTT EEEEEE     EEEEEE LL     EEEEEE M     M EEEEEE NN    N TTTTTT SSSSSS ###
 ### FF    II N N   N II   TT   EE         EE     LL     EE     MM   MM EE     N N   N   TT   SS     ###
 ### FFFF  II N  N  N II   TT   EEEEE      EEEEE  LL     EEEEE  M M M M EEEEE  N  N  N   TT    SSSS  ###
 ### FF    II N   N N II   TT   EE         EE     LL     EE     M  M  M EE     N   N N   TT       SS ###
 ### FF    II N    NN II   TT   EEEEEE     EEEEEE LLLLLL EEEEEE M     M EEEEEE N    NN   TT   SSSSSS ###
 #######################################################################################################
 #######################################################################################################

 include("FiniteElements_Lagrange.jl")
 include("FiniteElements_CrouzeixRaviart.jl")
 include("FiniteElements_BernardiRaugel.jl")
 include("FiniteElements_MINI.jl")
 include("FiniteElements_P2B.jl")
 


end # module
