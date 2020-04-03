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
include("FEoperators/CELL_EPSUdotEPSV.jl");
include("FEoperators/CELL_CEPSUdotEPSV.jl");
include("FEoperators/CELL_MdotDUdotDV.jl");
include("FEoperators/CELL_FdotDUdotV.jl")

# LINEAR FUNCTIONALS on cells
include("FEoperators/CELL_FdotV.jl");
include("FEoperators/CELL_FdotDIVV.jl");
include("FEoperators/CELL_FdotDV.jl");
include("FEoperators/CELL_L2_FplusA.jl");
include("FEoperators/CELL_L2_FplusLA.jl");
include("FEoperators/CELL_L2_FplusDIVA.jl");
include("FEoperators/CELL_L2_CURLA.jl");

# LINEAR FUNCTIONALS on full domain
include("FEoperators/DOMAIN_1dotV.jl");
include("FEoperators/DOMAIN_L2_FplusA.jl");

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


function computeDirichletBoundaryDataByInterpolation!(val4dofs,FE::FiniteElements.AbstractH1FiniteElement,Dbid,boundary_data!)

    # find boundary dofs
    
    bdofs = [];
    Grid.ensure_bfaces!(FE.grid);
    Grid.ensure_cells4faces!(FE.grid);
    T = eltype(FE.grid.coords4nodes)
    ET = FE.grid.elemtypes[1]
    ETF = Grid.get_face_elemtype(ET);
    ncomponents::Int = FiniteElements.get_ncomponents(FE);
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    ndofs4face::Int = FiniteElements.get_ndofs4elemtype(FE, ETF);

    xref, InterpolationMatrix = FiniteElements.get_xref4dof(FE, ET)
    xdim = size(FE.grid.coords4nodes,2);
    loc2glob_trafo = Grid.local2global(FE.grid, ET)

    cell::Int = 0;
    dofs4bface = zeros(Int64,ndofs4face)
    dofs4cell = zeros(Int64,ndofs4cell)
    coefficients = zeros(Float64,ndofs4cell)
    celldof2facedof = zeros(Int,ndofs4face)
    temp = zeros(eltype(FE.grid.coords4nodes),ncomponents);
    x = zeros(Float64,xdim)
    for i in eachindex(FE.grid.bfaces)
        if (FE.grid.bregions[i] == Dbid)

            # find neighbouring cell
            cell = FE.grid.cells4faces[FE.grid.bfaces[i],1];

            # get dofs
            FiniteElements.get_dofs_on_face!(dofs4bface, FE, FE.grid.bfaces[i], ETF);
            FiniteElements.get_dofs_on_cell!(dofs4cell, FE, cell, ET);
            FiniteElements.get_basis_coefficients_on_cell!(coefficients, FE, cell, ET)

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

            for k = 1:ndofs4face
                # evaluate Dirichlet data
                x[:] = trafo_on_cell(xref[celldof2facedof[k]])
                boundary_data!(temp,x);

                # add function value to dof and subtract interferences of other dofs
                for d = 1 : FiniteElements.get_ncomponents(FE)
                    val4dofs[dofs4bface] += (InterpolationMatrix[d][celldof2facedof[k],celldof2facedof].*coefficients[celldof2facedof,d])*temp[d];
                end    
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

# Interpolation for H1 FiniteElements
# requires FEdefinition to define the two functions
# get_xref4dof : reference coordinats of point evaluation for local dofnumber
# get_interpolation_matrix : to manage interference of other dofs that are nonzero at xref4dof
#                            (usually just the identity matrix, but may be different in case of
#                             additional bubbles are present or if other basis functions are used)
function computeFEInterpolation!(val4dofs::Array,source_function!::Function,FE::AbstractH1FiniteElement)
    
    # get xref and interpolation matrix
    ET = FE.grid.elemtypes[1];
    xref, InterpolationMatrix = FiniteElements.get_xref4dof(FE, ET)
    dofs = zeros(Int64,length(xref))
    coefficients = zeros(Float64,length(xref),FiniteElements.get_ncomponents(FE))
    offsets = 0:length(xref):size(InterpolationMatrix,2)

    # trafo for evaluation of source function
    loc2glob_trafo = Grid.local2global(FE.grid,ET)

    #@time begin    
    temp = zeros(Float64,FiniteElements.get_ncomponents(FE));
    for cell = 1 : size(FE.grid.nodes4cells,1)

        # get trafo
        cell_trafo = loc2glob_trafo(cell)

        # get dofs and coefficients (needed for FE like BR)
        FiniteElements.get_basis_coefficients_on_cell!(coefficients, FE, cell, ET)
        FiniteElements.get_dofs_on_cell!(dofs, FE, cell, ET);
        val4dofs[dofs] .= 0.0 

        for i = 1 : length(xref)
            # evaluate function at reference point for dof
            x = cell_trafo(xref[i]);
            fill!(temp,0.0)
            source_function!(temp,x);

            # add function value to dof and subtract interferences of other dofs
            for d = 1 : FiniteElements.get_ncomponents(FE)
                val4dofs[dofs] += (InterpolationMatrix[d][i,:].*coefficients[:,d])*temp[d];
            end    
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
