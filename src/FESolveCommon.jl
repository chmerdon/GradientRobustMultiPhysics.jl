module FESolveCommon

export computeBestApproximation!, computeFEInterpolation!, eval_interpolation_error!, eval_L2_interpolation_error!

using SparseArrays
using ExtendableSparse
using LinearAlgebra
using BenchmarkTools
using FiniteElements
using ForwardDiff
using Grid
using Quadrature



# MASS matrices on cells
include("FEoperators/CELL_UdotV.jl");
include("FEoperators/CELL_FdotUdotV.jl");

# MASS matrices on boundary faces
include("FEoperators/BFACE_UdotV.jl");
include("FEoperators/BFACE_UndotVn.jl");
#include("FEoperators/HDIV_bface_UxV.jl");

# STIFFNESS matrices on cells
include("FEoperators/CELL_DUdotDV.jl");
include("FEoperators/CELL_MdotDUdotDV.jl");

# LINEAR FUNCTIONALS on cells
include("FEoperators/CELL_FdotV.jl");
include("FEoperators/CELL_FdotDIVV.jl");
include("FEoperators/CELL_FdotDV.jl");
include("FEoperators/CELL_L2_FplusLA.jl");
include("FEoperators/CELL_L2_FplusDIVA.jl");
include("FEoperators/CELL_L2_CURLA.jl");

# LINEAR FUNCTIONALS on full domain
include("FEoperators/DOMAIN_1dotV.jl");

# LINEAR FUNCTIONALS on (boundary) faces
include("FEoperators/BFACE_FdotV.jl");
include("FEoperators/BFACE_FndotVn.jl");
include("FEoperators/BFACE_FdotVn.jl");
include("FEoperators/FACE_1dotVn.jl");
include("FEoperators/FACE_L2_JumpDA.jl");
include("FEoperators/FACE_L2_JumpA.jl");

# DIV-DIV matrices on cells
include("FEoperators/CELL_DIVUdotDIVV.jl");
include("FEoperators/CELL_UdotDIVV.jl");


function assembleSystem(nu::Real, norm_lhs::String,norm_rhs::String,volume_data!::Function,FE::AbstractFiniteElement,quadrature_order::Int)

    ncells::Int = size(FE.grid.nodes4cells,1);
    nnodes::Int = size(FE.grid.coords4nodes,1);
    celldim::Int = size(FE.grid.nodes4cells,2);
    xdim::Int = size(FE.grid.coords4nodes,2);
    
    Grid.ensure_volume4cells!(FE.grid);
    
    ndofs = FiniteElements.get_ndofs(FE);
    @time begin
        A = ExtendableSparseMatrix{Float64,Int64}(ndofs,ndofs);
        if norm_lhs == "L2"
            print("    |assembling mass matrix...")
            assemble_operator!(A,CELL_UdotV,FE);
        elseif norm_lhs == "H1"
            print("    |assembling stiffness matrix...")
            assemble_operator!(A,CELL_DUdotDV,FE,nu);
        end 
        println("finished")
    end
    
    # compute right-hand side vector
    @time begin
        print("    |assembling rhs...")
        b = FiniteElements.createFEVector(FE);
        if norm_rhs == "L2"
            assemble_operator!(b, CELL_FdotV, FE, volume_data!, quadrature_order)
        elseif norm_rhs == "H1"
            assemble_operator!(b, CELL_FdotDV, FE, volume_data!, quadrature_order)
        end
        println("finished")
    end
    
    
    return A,b
end

function computeDirichletBoundaryData!(val4dofs,FE,Dbids::Vector{Int64},boundary_data!::Vector{Function}, use_L2bestapproximation, quadorder::Vector{Int64})
    nbregions = length(Dbids)
    if use_L2bestapproximation == false
        bdofs = [];
        for j = 1 : nbregions
            if boundary_data![j] != Nothing
                dofs = computeDirichletBoundaryDataByInterpolation!(val4dofs,FE,Dbids[j],boundary_data![j]);    
                append!(bdofs,dofs)
            end    
        end    
        unique!(bdofs)
        return bdofs
    else
        ensure_bfaces!(FE.grid);
        ETF = Grid.get_face_elemtype(FE.grid.elemtypes[1]);
        nbfaces::Int = size(FE.grid.bfaces,1);
        ndofs4face::Int = FiniteElements.get_ndofs4elemtype(FE, ETF);

        # assemble operators
        ndofs = FiniteElements.get_ndofs(FE);
        B = ExtendableSparseMatrix{Float64,Int64}(ndofs,ndofs)
        assemble_operator!(B, BFACE_UdotV, FE, Dbids)
        b = FiniteElements.createFEVector(FE);

        # find all bdofs on selected boundary parts
        bdofs = [];
        dofs = zeros(Int64,ndofs4face)
        for j = 1 : nbregions
            if boundary_data![j] != Nothing
                if typeof(FE) <: AbstractH1FiniteElement
                    assemble_operator!(b, BFACE_FdotV, FE, Dbids[j], boundary_data![j], quadorder[j])
                elseif typeof(FE) <: AbstractHdivFiniteElement
                    assemble_operator!(b, BFACE_FndotVn, FE, Dbids[j], boundary_data![j], quadorder[j])
                else
                end        
                for face in eachindex(FE.grid.bfaces)
                    if FE.grid.bregions[face] == Dbids[j]
                        FiniteElements.get_dofs_on_face!(dofs, FE, FE.grid.bfaces[face], ETF);
                        append!(bdofs,dofs)
                    end    
                end
            end    
        end    

        # solve bestapproximation problem on boundary
        unique!(bdofs)
        val4dofs[bdofs] = B[bdofs,bdofs]\b[bdofs];
        return bdofs
    end    
end

function computeDirichletBoundaryData!(val4dofs,FE,boundary_data!,use_L2bestapproximation = false, quadorder = 1)
    dummy = Vector{Function}(undef, 1)
    dummy[1] = boundary_data!
    computeDirichletBoundaryData!(val4dofs,FE,[0],dummy,use_L2bestapproximation,[quadorder])
end


function computeDirichletBoundaryDataByInterpolation!(val4dofs,FE,Dbid,boundary_data!)

    # find boundary dofs
    xdim = FiniteElements.get_ncomponents(FE);
    ndofs::Int = FiniteElements.get_ndofs(FE);
    
    bdofs = [];
    Grid.ensure_bfaces!(FE.grid);
    Grid.ensure_cells4faces!(FE.grid);
    T = eltype(FE.grid.coords4nodes)
    ET = FE.grid.elemtypes[1]
    ETF = Grid.get_face_elemtype(ET);
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    ndofs4face::Int = FiniteElements.get_ndofs4elemtype(FE, ETF);

    xref = FiniteElements.get_xref4dof(FE, ET)
    temp = zeros(eltype(FE.grid.coords4nodes),xdim);
    dim = size(FE.grid.nodes4cells,2) - 1;
    loc2glob_trafo = Grid.local2global(FE.grid, ET)

    # pre-allocate memory for basis functions
    ncomponents = FiniteElements.get_ncomponents(FE);
    if ncomponents == 1
        basisvals = Array{Array{T,1}}(undef,size(xref,1));
    else
        basisvals = Array{Array{T,2}}(undef,size(xref,1));
    end
    for i = 1:size(xref,1)
        basisvals[i] = FiniteElements.get_basis_on_cell(FE, ET)(xref[i,:])
    end    

    cell::Int = 0;
    j::Int = 1;
    A4bface = Matrix{Float64}(undef,ndofs4face,ndofs4face)
    b4bface = Vector{Float64}(undef,ndofs4face)
    dofs4bface = zeros(Int64,ndofs4face)
    dofs4cell = zeros(Int64,ndofs4cell)
    celldof2facedof = zeros(Int,ndofs4face)
    for i in eachindex(FE.grid.bfaces)

        if (FE.grid.bregions[i] == Dbid)

            # find neighbouring cell
            cell = FE.grid.cells4faces[FE.grid.bfaces[i],1];

            # get dofs
            FiniteElements.get_dofs_on_face!(dofs4bface, FE, FE.grid.bfaces[i], ETF);
            FiniteElements.get_dofs_on_cell!(dofs4cell, FE, cell, ET);

            # find position of face dofs in cell dofs
            for j=1:ndofs4cell, k = 1 : ndofs4face
                if dofs4cell[j] == dofs4bface[k]
                    celldof2facedof[k] = j;
                end   
            end

            # append face dofs to bdofs
            append!(bdofs,dofs4bface);

            # setup trafo
            trafo_on_cell = loc2glob_trafo(cell);

            # setup local system of equations to determine piecewise interpolation of boundary data

            # assemble matrix and right-hand side
            for k = 1:ndofs4face
                for l = 1:ndofs4face
                    A4bface[k,l] = dot(basisvals[celldof2facedof[k]][celldof2facedof[k],:],basisvals[celldof2facedof[k]][celldof2facedof[l],:]);
                end 
                
                # evaluate Dirichlet data
                boundary_data!(temp,trafo_on_cell(xref[celldof2facedof[k],:]));
                b4bface[k] = dot(temp,basisvals[celldof2facedof[k]][celldof2facedof[k],:]);
            end

            # solve
            val4dofs[dofs4bface] = A4bface\b4bface;
            if norm(A4bface*val4dofs[dofs4bface]-b4bface) > eps(1e3)
                println("WARNING: large residual, boundary data may be inexact");
            end
        end    
    end    
    return unique(bdofs)
end

# computes Bestapproximation in approx_norm="L2" or "H1"
# volume_data! for norm="H1" is expected to be the gradient of the function that is bestapproximated
function computeBestApproximation!(val4dofs::Array,approx_norm::String ,volume_data!::Function,boundary_data!,FE::AbstractFiniteElement,quadrature_order::Int, dirichlet_penalty = 1e60)

    println("\nCOMPUTING " * approx_norm * " BESTAPPROXIMATION")
    println(" |   FE = " * FE.name)
    println(" |ndofs = ", FiniteElements.get_ndofs(FE))
    println(" |");
    println(" |PROGRESS")

    # assemble system 
    A, b = assembleSystem(1.0,approx_norm,approx_norm,volume_data!,FE,FiniteElements.get_polynomial_order(FE) + quadrature_order);
    
    # apply boundary data
    bdofs = []
    if boundary_data! != Nothing
        celldim::Int = size(FE.grid.nodes4cells,2) - 1;
        Dbids = sort(unique(FE.grid.bregions));
        bd_data = Vector{Function}(undef,length(Dbids))
        for j = 1 : length(Dbids)
            bd_data[j] = boundary_data!
        end    
        bdofs = computeDirichletBoundaryData!(val4dofs,FE,Dbids,bd_data,celldim > 1,ones(Int64,length(Dbids))*quadrature_order);
        for i = 1 : length(bdofs)
        A[bdofs[i],bdofs[i]] = dirichlet_penalty;
        b[bdofs[i]] = val4dofs[bdofs[i]]*dirichlet_penalty;
        end
    end

    # solve
    val4dofs[:] = A\b;
    
    # compute residual (exclude bdofs)
    residual = A*val4dofs - b
    residual[bdofs] .= 0
    residual = norm(residual);
    println("    |residual=", residual)
    
    return residual
end

# TODO: has to be rewritten!!!
function computeFEInterpolation!(val4dofs::Array,source_function!::Function,FE::AbstractH1FiniteElement)
    ET = FE.grid.elemtypes[1];
    temp = zeros(Float64,FiniteElements.get_ncomponents(FE));
    xref = FiniteElements.get_xref4dof(FE, ET)
    ndofs4cell = FiniteElements.get_ndofs4elemtype(FE, ET);
    loc2glob_trafo = Grid.local2global(FE.grid,ET)
    dofs = zeros(Int64,ndofs4cell)
    ncomponents = FiniteElements.get_ncomponents(FE);
    coefficients = zeros(Float64,ndofs4cell,ncomponents)
    # loop over nodes
    for cell = 1 : size(FE.grid.nodes4cells,1)
        # get trafo
        cell_trafo = loc2glob_trafo(cell)
        
        # get dofs
        FiniteElements.get_dofs_on_cell!(dofs,FE,cell,ET)

        # get coefficients
        FiniteElements.get_basis_coefficients_on_cell!(coefficients, FE, cell, ET)

        for k = 1 : ndofs4cell
            x = cell_trafo(xref[k, :]);
            source_function!(temp,x);
            val4dofs[dofs[k],:] = temp;
        end    
    end
end


function eval_FEfunction(coeffs, FE::AbstractH1FiniteElement)
    ncomponents = FiniteElements.get_ncomponents(FE);
    ET = FE.grid.elemtypes[1];
    ndofs4cell = FiniteElements.get_ndofs4elemtype(FE, ET);
    basisvals = zeros(eltype(FE.grid.coords4nodes),ndofs4cell,ncomponents)
    dofs = zeros(Int64,ndofs4cell)
    coefficients = zeros(Float64,ndofs4cell,ncomponents)
    function closure(result, x, xref, cell)
        fill!(result,0.0)

        # get dofs
        FiniteElements.get_dofs_on_cell!(dofs,FE,cell,ET)

        # get coefficients
        FiniteElements.get_basis_coefficients_on_cell!(coefficients, FE, cell, ET)

        for j = 1 : ndofs4cell
            for k = 1 : ncomponents;
                result[k] += basisvals[j,k] * coeffs[dofs[j]] * coefficients[j,k];
            end    
        end    
    end
end

function eval_L2_interpolation_error!(exact_function!, coeffs_interpolation, FE::AbstractH1FiniteElement)
    ncomponents = FiniteElements.get_ncomponents(FE);
    ET = FE.grid.elemtypes[1]
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    basisvals = zeros(eltype(FE.grid.coords4nodes),ndofs4cell,ncomponents)
    coefficients = zeros(Float64,ndofs4cell,ncomponents);
    dofs = zeros(Int64,ndofs4cell)
    function closure(result, x, xref, cell)
        # evaluate exact function
        exact_function!(result, x);

        # evaluate FE functions
        basisvals = FiniteElements.get_basis_on_cell(FE, ET)(xref)

        # get dofs
        FiniteElements.get_dofs_on_cell!(dofs, FE, cell, ET);

        # get coefficients
        FiniteElements.get_basis_coefficients_on_cell!(coefficients, FE, cell, ET);

        for j = 1 : ndofs4cell
            for k = 1 : ncomponents;
                result[k] -= basisvals[j,k] * coeffs_interpolation[dofs[j]] * coefficients[j,k];
            end    
        end   
        # square for L2 norm
        for j = 1 : length(result)
            result[j] = result[j]^2
        end    
    end
end


function eval_L2_interpolation_error!(exact_function!, coeffs_interpolation, FE::AbstractHdivFiniteElement)
    ncomponents = FiniteElements.get_ncomponents(FE);
    ET = FE.grid.elemtypes[1]
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    basisvals = zeros(eltype(FE.grid.coords4nodes),ndofs4cell,ncomponents)
    coefficients = zeros(Float64,ndofs4cell,ncomponents);
    AT = zeros(Float64,2,2)
    get_Piola_trafo_on_cell! = Grid.local2global_Piola(FE.grid, ET)
    det = 0.0;
    dofs = zeros(Int64,ndofs4cell)
    function closure(result, x, xref, cellIndex)
        # evaluate exact function
        exact_function!(result, x);
        
        # subtract nodal interpolation
        det = get_Piola_trafo_on_cell!(AT,cellIndex);
        basisvals[:] = FiniteElements.get_basis_on_cell(FE, ET)(xref)

        # get dofs
        FiniteElements.get_dofs_on_cell!(dofs, FE, cellIndex, ET)

        # get coefficients
        FiniteElements.get_basis_coefficients_on_cell!(coefficients, FE, cellIndex, ET)

        for j = 1 : ndofs4cell
            for k = 1 : ncomponents
                for l = 1: ncomponents
                    result[k] -= AT[k,l] * basisvals[j,l] * coeffs_interpolation[dofs[j]] * coefficients[j,k] / det;
                end    
            end    
        end   
        # square for L2 norm
        for j = 1 : length(result)
            result[j] = result[j]^2
        end    
    end
end


function eval_at_nodes(val4dofs, FE::AbstractH1FiniteElement, offset::Int64 = 0)
    # evaluate at grid points
    ndofs4node = zeros(size(FE.grid.coords4nodes,1))
    ET = FE.grid.elemtypes[1]
    xref4dofs4cell = Grid.get_reference_cordinates(ET)
    ncomponents = FiniteElements.get_ncomponents(FE);
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    basisvals = zeros(eltype(FE.grid.coords4nodes),ndofs4cell,ncomponents)
    nodevals = zeros(size(FE.grid.coords4nodes,1),ncomponents)
    coefficients = zeros(Float64,ndofs4cell,ncomponents)
    dofs = zeros(Int64,ndofs4cell)
    for j = 1 : size(xref4dofs4cell,1)
        # get basisvals
        basisvals[:] = FiniteElements.get_basis_on_cell(FE,ET)(xref4dofs4cell[j,:])

        for cell = 1 : size(FE.grid.nodes4cells,1)

            # get dofs
            FiniteElements.get_dofs_on_cell!(dofs, FE, cell, ET);

            # get coefficients
            FiniteElements.get_basis_coefficients_on_cell!(coefficients, FE, cell, ET)
        
            for dof = 1 : ndofs4cell;
                for k = 1 : ncomponents
                    nodevals[FE.grid.nodes4cells[cell,j],k] += basisvals[dof,k]*val4dofs[offset+dofs[dof]]*coefficients[dof,k];
                end   
            end
            ndofs4node[FE.grid.nodes4cells[cell,j]] +=1
        end    
    end
    # average
    for k = 1 : ncomponents
        nodevals[:,k] ./= ndofs4node
    end
    return nodevals
end   

function eval_at_nodes(val4dofs, FE::AbstractHdivFiniteElement, offset::Int64 = 0)
    # evaluate at grid points
    ndofs4node = zeros(size(FE.grid.coords4nodes,1))
    ET = FE.grid.elemtypes[1]
    xref4dofs4cell = Grid.get_reference_cordinates(ET)
    ncomponents = FiniteElements.get_ncomponents(FE);
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    basisvals = zeros(eltype(FE.grid.coords4nodes),ndofs4cell,ncomponents)
    nodevals = zeros(size(FE.grid.coords4nodes,1),ncomponents)
    coefficients = zeros(Float64,ndofs4cell,ncomponents)
    AT = zeros(Float64,2,2)
    get_Piola_trafo_on_cell! = Grid.local2global_Piola(FE.grid, ET)
    det = 0.0;
    dofs = zeros(Int64, ndofs4cell)
    for j = 1 : size(xref4dofs4cell,1)

        basisvals[:] = FiniteElements.get_basis_on_cell(FE, ET)(xref4dofs4cell[j,:])

        for cell = 1 : size(FE.grid.nodes4cells,1)
            # get dofs
            FiniteElements.get_dofs_on_cell!(dofs, FE, cell, ET);

            # get coefficients
            FiniteElements.get_basis_coefficients_on_cell!(coefficients, FE, cell, ET);
            
            # get trafo
            det = get_Piola_trafo_on_cell!(AT,cell);

    
            for dof = 1 : ndofs4cell;
                for k = 1 : ncomponents
                    for l = 1: ncomponents
                        nodevals[FE.grid.nodes4cells[cell,j],k] +=  AT[k,l] * basisvals[dof,l]*val4dofs[dofs[dof]]*coefficients[dof,k] / det;
                    end    
                end   
            end
            ndofs4node[FE.grid.nodes4cells[cell,j]] +=1
        end    
    end
    # average
    for k = 1 : ncomponents
        nodevals[:,k] ./= ndofs4node
    end
    return nodevals
end   

end
