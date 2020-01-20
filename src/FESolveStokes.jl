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

    
    # pre-allocate memory for gradients
    gradients4cell = Array{Array{T,1}}(undef,xdim*ndofs4cell_velocity);
    for j = 1 : xdim*ndofs4cell_velocity
        gradients4cell[j] = zeros(T,xdim);
    end
    dofs_velocity = zeros(Int64,ndofs4cell_velocity)
    
    # pre-allocate for derivatives of global2local trafo and basis function
    gradients_xref_cache = zeros(Float64,length(qf.w),xdim*ndofs4cell_velocity,celldim)
    #DRresult_grad = DiffResults.DiffResult(Vector{T}(undef, celldim), Matrix{T}(undef,xdim*ndofs4cell_velocity,celldim));
    trafo_jacobian = Matrix{T}(undef,xdim,xdim);
    
    dim = celldim - 1;
    if dim == 1
        loc2glob_trafo_tinv = FiniteElements.local2global_tinv_jacobian_line
    elseif dim == 2
        loc2glob_trafo_tinv = FiniteElements.local2global_tinv_jacobian_triangle
    end    
    
    # quadrature loop
    div_i::T = 0.0;
    div_j::T = 0.0;
    det::T = 0.0;
    offsets = [0,ndofs4cell_velocity];
    @time begin
    for cell = 1 : ncells
      # evaluate tinverted (=transposed + inverted) jacobian of element trafo
      loc2glob_trafo_tinv(trafo_jacobian,det,FE_velocity.grid,cell)
      
      # cache dofs
      for dof_i = 1 : ndofs4cell_velocity
          dofs_velocity[dof_i] = FiniteElements.get_globaldof4cell(FE_velocity, cell, dof_i);
      end 
      
      for i in eachindex(qf.w)
      
        # evaluate gradients of basis function
        #ForwardDiff.jacobian!(DRresult_grad,FiniteElements.get_all_basis_functions_on_cell(FE_velocity, cell),qf.xref[i]);
        if (cell == 1)
            gradients_xref_cache[i,:,:] = ForwardDiff.jacobian(FiniteElements.get_all_basis_functions_on_cell(FE_velocity, cell),qf.xref[i]);
        end    
        
        # multiply tinverted jacobian of element trafo with gradient of basis function
        # which yields (by chain rule) the gradient in x coordinates
        for dof_i = 1 : xdim*ndofs4cell_velocity
            for k = 1 : xdim
                gradients4cell[dof_i][k] = 0.0;
                for j = 1 : xdim
                    gradients4cell[dof_i][k] += trafo_jacobian[k,j]*gradients_xref_cache[i,dof_i,j]
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



function assemble_Stokes_Operator4FE!(A::ExtendableSparseMatrix, nu::Real, FE_velocity::FiniteElements.AbstractH1FiniteElement, FE_pressure::FiniteElements.AbstractH1FiniteElement, pressure_diagonal = 1e-12)

    grid = FE_velocity.grid;
    ncells::Int = size(grid.nodes4cells,1);
    xdim::Int = size(grid.coords4nodes,2);
    ndofs4cell_velocity::Int = FiniteElements.get_maxndofs4cell(FE_velocity);
    ndofs4cell_pressure::Int = FiniteElements.get_maxndofs4cell(FE_pressure);
    ncomponents::Int = FiniteElements.get_ncomponents(FE_velocity);
    ndofs_velocity = FiniteElements.get_ndofs(FE_velocity);
    ndofs4cell::Int = ndofs4cell_velocity+ndofs4cell_pressure;
    celldim::Int = size(grid.nodes4cells,2);
    
    @assert ncomponents == xdim
    
    T = eltype(grid.coords4nodes);
    quadorder = maximum([FiniteElements.get_polynomial_order(FE_pressure) + FiniteElements.get_polynomial_order(FE_velocity)-1, 2*(FiniteElements.get_polynomial_order(FE_velocity)-1)]);
    qf = QuadratureFormula{T}(quadorder, xdim);

    
    # pre-allocate memory for gradients
    gradients4cell = Array{Array{T,1}}(undef,xdim*ndofs4cell_velocity);
    for j = 1 : xdim*ndofs4cell_velocity
        gradients4cell[j] = zeros(T,xdim);
    end
    pressure_vals = Array{Array{T,1}}(undef,length(qf.w))
    for j = 1 : ndofs4cell_pressure
        pressure_vals[j] = zeros(T,ndofs4cell_pressure);
    end    
    dofs_velocity = zeros(Int64,ndofs4cell_velocity)
    dofs_pressure = zeros(Int64,ndofs4cell_pressure)
    
    # pre-allocate for derivatives of global2local trafo and basis function
    gradients_xref_cache = zeros(Float64,length(qf.w),xdim*ndofs4cell_velocity,celldim)
    #DRresult_grad = DiffResults.DiffResult(Vector{T}(undef, celldim), Matrix{T}(undef,xdim*ndofs4cell_velocity,celldim));
    trafo_jacobian = Matrix{T}(undef,xdim,xdim);
    
    dim = celldim - 1;
    if dim == 1
        loc2glob_trafo_tinv = FiniteElements.local2global_tinv_jacobian_line
    elseif dim == 2
        loc2glob_trafo_tinv = FiniteElements.local2global_tinv_jacobian_triangle
    end    
    
    # quadrature loop
    temp::T = 0.0;
    det::T = 0.0;
    offsets = [0,ndofs4cell_velocity];
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
      
      for i in eachindex(qf.w)
      
        # evaluate gradients of basis function
        #ForwardDiff.jacobian!(DRresult_grad,FiniteElements.get_all_basis_functions_on_cell(FE_velocity, cell),qf.xref[i]);
        if (cell == 1)
            gradients_xref_cache[i,:,:] = ForwardDiff.jacobian(FiniteElements.get_all_basis_functions_on_cell(FE_velocity, cell),qf.xref[i]);
            pressure_vals[i] = FiniteElements.get_all_basis_functions_on_cell(FE_pressure, cell)(qf.xref[i]);
        end    
        
        # multiply tinverted jacobian of element trafo with gradient of basis function
        # which yields (by chain rule) the gradient in x coordinates
        for dof_i = 1 : xdim*ndofs4cell_velocity
            for k = 1 : xdim
                gradients4cell[dof_i][k] = 0.0;
                for j = 1 : xdim
                    gradients4cell[dof_i][k] += trafo_jacobian[k,j]*gradients_xref_cache[i,dof_i,j]
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
                temp *= pressure_vals[i][dof_j] * qf.w[i] * grid.volume4cells[cell];
                A[dofs_velocity[dof_i],dofs_pressure[dof_j]] += temp;
                A[dofs_pressure[dof_j],dofs_velocity[dof_i]] += temp;
            end
        end    
        
        # pressure x pressure block (empty)
        for dof_i = 1 : ndofs4cell_pressure, dof_j = 1 : ndofs4cell_pressure
            if (dof_i == dof_j)
                A[dofs_pressure[dof_i],dofs_pressure[dof_j]] = pressure_diagonal;
            end    
        end
      end  
    end
  end  
end


# scalar functions times P1 basis functions
function rhs_integrand4Stokes!(f!::Function,FE::FiniteElements.AbstractFiniteElement,dim)
    cache = zeros(eltype(FE.grid.coords4nodes),dim)
    basisval = zeros(eltype(FE.grid.coords4nodes),dim)
    ndofs4cell::Int = FiniteElements.get_maxndofs4cell(FE);
    basisvals = zeros(eltype(FE.grid.coords4nodes),ndofs4cell,dim)
    function closure(result,x,xref,cellIndex::Int)
        f!(cache, x);
        basisvals = FiniteElements.get_all_basis_functions_on_cell(FE, cellIndex)(xref)
        for j=1:ndofs4cell
            result[j] = 0.0;
            for d=1:dim
                result[j] += cache[d] * basisvals[j,d];
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
    A = ExtendableSparseMatrix{Float64,Int64}(ndofs,ndofs)
    assemble_Stokes_Operator4FE!(A,nu,FE_velocity,FE_pressure);
    
    # compute right-hand side vector
    b = zeros(Float64,ndofs);
    FESolveCommon.assemble_rhsL2!(b, volume_data!, FE_velocity)
    
    return A,b
end


function solveStokesProblem!(val4dofs::Array,nu::Real,volume_data!::Function,boundary_data!,grid::Grid.Mesh,FE_velocity::FiniteElements.AbstractFiniteElement,FE_pressure::FiniteElements.AbstractFiniteElement,quadrature_order::Int, dirichlet_penalty = 1e60)
    # assemble system 
    A, b = assembleStokesSystem(nu, volume_data!,FE_velocity,FE_pressure,quadrature_order);
    
    # remove one pressure dof (todo: don't do this with Neumann boundary)
    ndofs_velocity = FiniteElements.get_ndofs(FE_velocity);
    ndofs_pressure = FiniteElements.get_ndofs(FE_pressure);
    ndofs = ndofs_velocity + ndofs_pressure;
    
    # apply boundary data
    bdofs = FESolveCommon.computeDirichletBoundaryData!(val4dofs,FE_velocity,boundary_data!,true);
    for i = 1 : length(bdofs)
       A[bdofs[i],bdofs[i]] = dirichlet_penalty;
       b[bdofs[i]] = val4dofs[bdofs[i]]*dirichlet_penalty;
    end
    @assert maximum(bdofs) <= ndofs_velocity
    
    #println("fixing one pressure dof with dofnr=",fixed_pressure_dof);
    fixed_pressure_dof = ndofs_velocity + 1;
    A[fixed_pressure_dof,fixed_pressure_dof] = 1e30;
    b[fixed_pressure_dof] = 0;
    val4dofs[fixed_pressure_dof] = 0;
    
    
    
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
    
    # move integral mean to zero
    integral_mean = integrate_xref(FESolveCommon.eval_FEfunction(val4dofs[ndofs_velocity+1:ndofs],FE_pressure),FE_pressure.grid, FiniteElements.get_polynomial_order(FE_pressure));
    integral_mean ./= sum(FE_pressure.grid.volume4cells)
    for i=1:ndofs_pressure
        val4dofs[ndofs_velocity+i] -= integral_mean[1] 
    end
    
    return norm(residual)
end


end
