module FESolveStokes

export solveStokesProblem!

using ExtendableSparse
using SparseArrays
using LinearAlgebra
using BenchmarkTools
using FiniteElements
using FESolveCommon
using DiffResults
using ForwardDiff
using Grid
using Quadrature

function assemble_divdiv_Matrix!(A::ExtendableSparseMatrix, FE_velocity::FiniteElements.AbstractH1FiniteElement)

    grid = FE_velocity.grid;
    ncells::Int = size(grid.nodes4cells,1);
    xdim::Int = size(grid.coords4nodes,2);
    ndofs4cell_velocity::Int = FiniteElements.get_maxndofs4cell(FE_velocity);
    ncomponents::Int = FiniteElements.get_ncomponents(FE_velocity);
    ndofs_velocity = FiniteElements.get_ndofs(FE_velocity);
    celldim::Int = size(grid.nodes4cells,2);
    
    @assert ncomponents == xdim
    
    T = eltype(grid.coords4nodes);
    quadorder = 2*(FiniteElements.get_polynomial_order(FE_velocity)-1);
    qf = QuadratureFormula{T}(quadorder, xdim);

    # pre-allocate gradients of basis function in reference coordinates
    gradients_xref_cache = zeros(Float64,length(qf.w),xdim*ndofs4cell_velocity,celldim)
    for i in eachindex(qf.w)
        # evaluate gradients of basis function
        gradients_xref_cache[i,:,:] = ForwardDiff.jacobian(FiniteElements.get_all_basis_functions_on_cell(FE_velocity),qf.xref[i]);
    end    
    
    # determine needed trafo
    dim = celldim - 1;
    if dim == 1
        loc2glob_trafo_tinv = FiniteElements.local2global_tinv_jacobian_line
    elseif dim == 2
        loc2glob_trafo_tinv = FiniteElements.local2global_tinv_jacobian_triangle
    end    
    trafo_jacobian = Matrix{T}(undef,xdim,xdim);
        
    # pre-allocate memory for temporary stuff
    gradients4cell = Array{Array{T,1}}(undef,xdim*ndofs4cell_velocity);
    for j = 1 : xdim*ndofs4cell_velocity
        gradients4cell[j] = zeros(T,xdim);
    end
    dofs_velocity = zeros(Int64,ndofs4cell_velocity)
    coefficients_velocity = zeros(Float64,ndofs4cell_velocity,xdim)
    div_i::T = 0.0;
    div_j::T = 0.0;
    det::T = 0.0;
    offsets = [0,ndofs4cell_velocity];
    
    # audrature loop
    @time begin
    for cell = 1 : ncells
      # evaluate tinverted (=transposed + inverted) jacobian of element trafo
      loc2glob_trafo_tinv(trafo_jacobian,det,FE_velocity.grid,cell)
      
      # cache dofs
      for dof_i = 1 : ndofs4cell_velocity
          dofs_velocity[dof_i] = FiniteElements.get_globaldof4cell(FE_velocity, cell, dof_i);
      end 
      
      FiniteElements.set_basis_coefficients_on_cell!(coefficients_velocity,FE_velocity,cell);
      
      for i in eachindex(qf.w)
    
        # multiply tinverted jacobian of element trafo with gradient of basis function
        # which yields (by chain rule) the gradient in x coordinates
        for dof_i = 1 : ndofs4cell_velocity
            for c=1 : xdim, k = 1 : xdim
                gradients4cell[dof_i + offsets[c]][k] = 0.0;
                for j = 1 : xdim
                    gradients4cell[dof_i + offsets[c]][k] += trafo_jacobian[k,j]*gradients_xref_cache[i,dof_i + offsets[c],j] * coefficients_velocity[dof_i,c]
                end    
            end    
        end    
        
        # div x div velocity
        for dof_i = 1 : ndofs4cell_velocity
            div_i = 0.0;
            for c = 1 : xdim
                div_i -= gradients4cell[offsets[c]+dof_i][c];
            end    
            div_i *= qf.w[i] * grid.volume4cells[cell];
            for dof_j = 1 : ndofs4cell_velocity
                div_j = 0.0;
                for c = 1 : xdim
                    div_j -= gradients4cell[offsets[c]+dof_j][c];
                end    
                A[dofs_velocity[dof_i],dofs_velocity[dof_j]] += div_j*div_i;
            end
        end    
      end  
    end
  end  
end



function assemble_Stokes_Operator4FE!(A::ExtendableSparseMatrix, nu::Real, FE_velocity::FiniteElements.AbstractH1FiniteElement, FE_pressure::FiniteElements.AbstractH1FiniteElement, pressure_diagonal = 1e-6)

    grid = FE_velocity.grid;
    ncells::Int = size(grid.nodes4cells,1);
    xdim::Int = size(grid.coords4nodes,2);
    ndofs4cell_velocity::Int = FiniteElements.get_maxndofs4cell(FE_velocity);
    ndofs4cell_pressure::Int = FiniteElements.get_maxndofs4cell(FE_pressure);
    ncomponents::Int = FiniteElements.get_ncomponents(FE_velocity);
    ndofs_velocity = FiniteElements.get_ndofs(FE_velocity);
    ndofs = ndofs_velocity + FiniteElements.get_ndofs(FE_pressure);
    ndofs4cell::Int = ndofs4cell_velocity+ndofs4cell_pressure;
    celldim::Int = size(grid.nodes4cells,2);
    
    @assert ncomponents == xdim
    
    T = eltype(grid.coords4nodes);
    quadorder = maximum([FiniteElements.get_polynomial_order(FE_pressure) + FiniteElements.get_polynomial_order(FE_velocity)-1, 2*(FiniteElements.get_polynomial_order(FE_velocity)-1)]);
    qf = QuadratureFormula{T}(quadorder, xdim);

    # evaluate basis functions at quarature points in reference coordinates
    gradients_xref_cache = zeros(Float64,length(qf.w),xdim*ndofs4cell_velocity,celldim)
    pressure_vals = Array{Array{T,1}}(undef,length(qf.w))
    for i in eachindex(qf.w)
        # evaluate gradients of basis function
        gradients_xref_cache[i,:,:] = ForwardDiff.jacobian(FiniteElements.get_all_basis_functions_on_cell(FE_velocity),qf.xref[i]);
        # evaluate pressure
        pressure_vals[i] = FiniteElements.get_all_basis_functions_on_cell(FE_pressure)(qf.xref[i]);
    end
    
    # get needed trafo
    dim = celldim - 1;
    if dim == 1
        loc2glob_trafo_tinv = FiniteElements.local2global_tinv_jacobian_line
    elseif dim == 2
        loc2glob_trafo_tinv = FiniteElements.local2global_tinv_jacobian_triangle
    end    
    trafo_jacobian = Matrix{T}(undef,xdim,xdim);
    
    
    # pre-allocate memory for temporary stuff
    gradients4cell = Array{Array{T,1}}(undef,xdim*ndofs4cell_velocity);
    for j = 1 : xdim*ndofs4cell_velocity
        gradients4cell[j] = zeros(T,xdim);
    end    
    dofs_velocity = zeros(Int64,ndofs4cell_velocity)
    dofs_pressure = zeros(Int64,ndofs4cell_pressure)
    coefficients_velocity = zeros(Float64,ndofs4cell_velocity,xdim)
    coefficients_pressure = zeros(Float64,ndofs4cell_pressure)
    temp::T = 0.0;
    det::T = 0.0;
    offsets = [0,ndofs4cell_velocity];
    
    # quadrature loop
    @time begin
    for cell = 1 : ncells
      # evaluate tinverted (=transposed + inverted) jacobian of element trafo
      loc2glob_trafo_tinv(trafo_jacobian,det,FE_velocity.grid,cell)
      
      # cache dofs
      for dof_i = 1 : ndofs4cell_velocity
          dofs_velocity[dof_i] = FiniteElements.get_globaldof4cell(FE_velocity, cell, dof_i);
      end 
      for dof_i = 1 : ndofs4cell_pressure
          dofs_pressure[dof_i] = ndofs_velocity + FiniteElements.get_globaldof4cell(FE_pressure, cell, dof_i);
      end    
      
      FiniteElements.set_basis_coefficients_on_cell!(coefficients_velocity,FE_velocity,cell);
      FiniteElements.set_basis_coefficients_on_cell!(coefficients_pressure,FE_pressure,cell);
      
      for i in eachindex(qf.w)
      
        # multiply tinverted jacobian of element trafo with gradient of basis function
        # which yields (by chain rule) the gradient in x coordinates
        for dof_i = 1 : ndofs4cell_velocity
            for c=1 : xdim, k = 1 : xdim
                gradients4cell[dof_i + offsets[c]][k] = 0.0;
                for j = 1 : xdim
                    gradients4cell[dof_i + offsets[c]][k] += trafo_jacobian[k,j]*gradients_xref_cache[i,dof_i + offsets[c],j] * coefficients_velocity[dof_i,c]
                end    
            end    
        end    
         
        # fill sparse array
        for dof_i = 1 : ndofs4cell_velocity
            # stiffness matrix for velocity
            for dof_j = 1 : ndofs4cell_velocity
                temp = 0.0;
                for k = 1 : xdim
                    for c = 1 : xdim
                        temp += gradients4cell[offsets[c]+dof_i][k] * gradients4cell[offsets[c]+dof_j][k];
                    end
                end
                A[dofs_velocity[dof_i],dofs_velocity[dof_j]] += temp * nu * qf.w[i] * grid.volume4cells[cell];
            end
            # pressure x div velocity
            for dof_j = 1 : ndofs4cell_pressure
                temp = 0.0;
                for c = 1 : xdim
                    temp -= gradients4cell[offsets[c]+dof_i][c];
                end    
                temp *= pressure_vals[i][dof_j] * coefficients_pressure[dof_j] * qf.w[i] * grid.volume4cells[cell];
                A[dofs_velocity[dof_i],dofs_pressure[dof_j]] += temp;
                A[dofs_pressure[dof_j],dofs_velocity[dof_i]] += temp;
            end
        end    
        
        # Lagrange multiplier for integral mean
        
        for dof_i = 1 : ndofs4cell_pressure
                temp = pressure_vals[i][dof_i] * coefficients_pressure[dof_i] * qf.w[i] * grid.volume4cells[cell];
                A[dofs_pressure[dof_i],ndofs + 1] += temp;
                A[ndofs + 1, dofs_pressure[dof_i]] += temp;
        end
      end  
    end
  end  
end

function assembleStokesSystem(nu::Real, volume_data!::Function,FE_velocity::FiniteElements.AbstractFiniteElement,FE_pressure::AbstractFiniteElement,quadrature_order::Int)

    grid = FE_velocity.grid;
    ncells::Int = size(grid.nodes4cells,1);
    nnodes::Int = size(grid.coords4nodes,1);
    celldim::Int = size(grid.nodes4cells,2);
    xdim::Int = size(grid.coords4nodes,2);
    ndofs_velocity::Int = FiniteElements.get_ndofs(FE_velocity);
    ndofs4cell_velocity::Int = FiniteElements.get_maxndofs4cell(FE_velocity);
    ndofs::Int = ndofs_velocity + FiniteElements.get_ndofs(FE_pressure);
    
    Grid.ensure_volume4cells!(grid);
    
    println("assembling Stokes matrix for FE pair " * FE_velocity.name * " x " * FE_pressure.name * "...");
    A = ExtendableSparseMatrix{Float64,Int64}(ndofs+1,ndofs+1) # +1 due to Lagrange multiplier for integral mean
    assemble_Stokes_Operator4FE!(A,nu,FE_velocity,FE_pressure);
    
    # compute right-hand side vector
    b = zeros(Float64,ndofs);
    FESolveCommon.assemble_rhsL2!(b, volume_data!, FE_velocity)
    
    return A,b
end


function solveStokesProblem!(val4dofs::Array,nu::Real,volume_data!::Function,boundary_data!,grid::Grid.Mesh,FE_velocity::FiniteElements.AbstractFiniteElement,FE_pressure::FiniteElements.AbstractFiniteElement,quadrature_order::Int, dirichlet_penalty = 1e60)
    # assemble system 
    A, b = assembleStokesSystem(nu, volume_data!,FE_velocity,FE_pressure,quadrature_order);
    
    ndofs_velocity = FiniteElements.get_ndofs(FE_velocity);
    ndofs_pressure = FiniteElements.get_ndofs(FE_pressure);
    ndofs = ndofs_velocity + ndofs_pressure;
    
     # add value for Lagrange multiplier for integral mean
    if length(val4dofs) == ndofs
        append!(val4dofs,0.0);
        append!(b,0.0); # add value for Lagrange multiplier for integral mean
    end
    
    # apply boundary data
    bdofs = FESolveCommon.computeDirichletBoundaryData!(val4dofs,FE_velocity,boundary_data!,true);
    for i = 1 : length(bdofs)
       A[bdofs[i],bdofs[i]] = dirichlet_penalty;
       b[bdofs[i]] = val4dofs[bdofs[i]]*dirichlet_penalty;
    end
    @assert maximum(bdofs) <= ndofs_velocity
    
    try
        @time val4dofs[:] = A\b;
    catch    
        println("Unsupported Number type for sparse lu detected: trying again with dense matrix");
        try
            val4dofs[:] = Array{typeof(grid.coords4nodes[1]),2}(A)\b;
        catch OverflowError
            println("OverflowError (Rationals?): trying again as Float64 sparse matrix");
            val4dofs[:] = Array{Float64,2}(A)\b;
        end
    end
    
    # compute residual (exclude bdofs)
    residual = A*val4dofs - b
    residual[bdofs] .= 0
    
    return norm(residual)
end


end
