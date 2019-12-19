module FESolveStokes

export solveStokesProblem!

using SparseArrays
using LinearAlgebra
using BenchmarkTools
using FiniteElements
using FESolveCommon
using Grid
using Quadrature


function StokesOperator4FE!(aa, ii, jj, nu::Real, FE_velocity::FiniteElements.FiniteElement, FE_pressure::FiniteElements.FiniteElement, pressure_diagonal = 1e-12)

    grid = FE_velocity.grid;
    ncells::Int = size(grid.nodes4cells,1);
    xdim::Int = size(grid.coords4nodes,2);
    ndofs4cell_velocity::Int = size(FE_velocity.dofs4cells,2);
    ndofs4cell_pressure::Int = size(FE_pressure.dofs4cells,2);
    ndofs_velocity = FE_velocity.ndofs;;
    ndofs4cell::Int = ndofs4cell_velocity+ndofs4cell_pressure;
    celldim::Int = size(grid.nodes4cells,2);
    
    @assert length(aa) == ncells*(ndofs4cell^2);
    @assert length(ii) == length(aa);
    @assert length(jj) == length(aa);
    
    T = eltype(grid.coords4nodes);
    quadorder = maximum([FE_pressure.polynomial_order + FE_velocity.polynomial_order-1, 2*(FE_velocity.polynomial_order-1)]);
    qf = QuadratureFormula{T}(quadorder, xdim);
    
    # compute local stiffness matrices
    curindex::Int = 0;
    x = zeros(T,xdim);
    
    # pre-allocate memory for gradients
    velogradients4cell = Array{Array{T,1}}(undef,ndofs4cell_velocity);
    pressure4cell = Array{T,1}(undef,ndofs4cell_pressure);
    for j = 1 : ndofs4cell_velocity
        velogradients4cell[j] = zeros(T,xdim*xdim);
    end
    for j = 1 : ndofs4cell_pressure
        pressure4cell[j] = 0.0;
    end
    
    # quadrature loop
    xref_mask = zeros(T,xdim)
    fill!(aa, 0.0);
    trace_indices = 1:(xdim+1):xdim^2
    for i in eachindex(qf.w)
      for j=1:xdim
        xref_mask[j] = qf.xref[i][j];
      end
      curindex = 0
      for cell = 1 : ncells
        # evaluate gradients at quadrature point
        for dof_i = 1 : ndofs4cell_velocity
            FE_velocity.bfun_grad![dof_i](velogradients4cell[dof_i],xref_mask,grid,cell);
        end    
        # evaluate pressures at quadrature point
        for dof_i = 1 : ndofs4cell_pressure
            pressure4cell[dof_i] = FE_pressure.bfun_ref[dof_i](xref_mask,grid,cell);
        end
        
        # fill fields aa,ii,jj
        for dof_i = 1 : ndofs4cell_velocity
            # stiffness matrix for velocity
            for dof_j = 1 : ndofs4cell_velocity
                curindex += 1;
                for k = 1 : xdim*xdim
                    aa[curindex] += nu*(velogradients4cell[dof_i][k] * velogradients4cell[dof_j][k] * qf.w[i] * grid.volume4cells[cell]);
                end
                if (i == 1)
                    ii[curindex] = FE_velocity.dofs4cells[cell,dof_i];
                    jj[curindex] = FE_velocity.dofs4cells[cell,dof_j];
                end    
            end
        end    
        # divvelo x pressure matrix
        for dof_i = 1 : ndofs4cell_velocity
            for dof_j = 1 : ndofs4cell_pressure
                curindex += 1;
                for k = 1 : length(trace_indices)
                    aa[curindex] -= (velogradients4cell[dof_i][trace_indices[k]] * pressure4cell[dof_j] * qf.w[i] * grid.volume4cells[cell]);
                end
                if (i == 1)
                    ii[curindex] = FE_velocity.dofs4cells[cell,dof_i];
                    jj[curindex] = ndofs_velocity + FE_pressure.dofs4cells[cell,dof_j];
                end  
                #copy transpose
                curindex += 1;
                aa[curindex] = aa[curindex-1]
                if (i == 1)
                    ii[curindex] = ndofs_velocity + FE_pressure.dofs4cells[cell,dof_j];
                    jj[curindex] = FE_velocity.dofs4cells[cell,dof_i];
                end 
            end
        end  
        # pressure x pressure block (empty)
        for dof_i = 1 : ndofs4cell_pressure, dof_j = 1 : ndofs4cell_pressure
            curindex +=1
            if (dof_i == dof_j)
                aa[curindex] = pressure_diagonal;
            end    
            if (i == 1)
                ii[curindex] = ndofs_velocity + FE_pressure.dofs4cells[cell,dof_i];
                jj[curindex] = ndofs_velocity + FE_pressure.dofs4cells[cell,dof_j];
            end 
        end
      end  
    end
end


# scalar functions times P1 basis functions
function rhs_integrand4Stokes!(f!::Function,FE::FiniteElements.FiniteElement,dim)
    cache = zeros(eltype(FE.grid.coords4nodes),dim)
    basisval = zeros(eltype(FE.grid.coords4nodes),dim)
    ndofcell::Int = size(FE.dofs4cells,2)
    function closure(result,x,xref,cellIndex::Int)
        f!(cache, x);
        for j=1:ndofcell
            basisval = FE.bfun_ref[j](xref,FE.grid,cellIndex);
            result[j] = 0.0;
            for d=1:dim
                result[j] += cache[d] * basisval[d];
            end    
        end
    end
end

function assembleStokesSystem(nu::Real, volume_data!::Function,FE_velocity::FiniteElements.FiniteElement,FE_pressure::FiniteElements.FiniteElement,quadrature_order::Int)

    grid = FE_velocity.grid;
    ncells::Int = size(grid.nodes4cells,1);
    nnodes::Int = size(grid.coords4nodes,1);
    celldim::Int = size(grid.nodes4cells,2);
    xdim::Int = size(grid.coords4nodes,2);
    ndofs_velocity::Int = FE_velocity.ndofs;
    ndofs::Int = ndofs_velocity + FE_pressure.ndofs;
    
    Grid.ensure_volume4cells!(grid);
    
    ndofs4cell_velocity = size(FE_velocity.dofs4cells,2)
    ndofs4cell = ndofs4cell_velocity + size(FE_pressure.dofs4cells,2)
    aa = Vector{typeof(grid.coords4nodes[1])}(undef, ndofs4cell^2*ncells);
    ii = Vector{Int64}(undef, ndofs4cell^2*ncells);
    jj = Vector{Int64}(undef, ndofs4cell^2*ncells);
    
    println("assembling Stokes matrix for FE pair " * FE_velocity.name * " x " * FE_pressure.name * "...");
    StokesOperator4FE!(aa,ii,jj,nu,FE_velocity,FE_pressure);
    A = sparse(ii,jj,aa,ndofs,ndofs);
    
    # compute right-hand side vector
    rhsintegral4cells = zeros(Base.eltype(grid.coords4nodes),ncells,ndofs4cell_velocity); # f x FEbasis
    println("integrate rhs for velocity");
    integrate!(rhsintegral4cells, rhs_integrand4Stokes!(volume_data!, FE_velocity,xdim), grid, quadrature_order, ndofs4cell_velocity);
         
    # accumulate right-hand side vector
    b = zeros(eltype(grid.coords4nodes),ndofs);
    FESolveCommon.accumarray!(b,FE_velocity.dofs4cells,rhsintegral4cells);
    
    return A,b
end


function solveStokesProblem!(val4dofs::Array,nu::Real,volume_data!::Function,boundary_data!,grid::Grid.Mesh,FE_velocity::FiniteElements.FiniteElement,FE_pressure::FiniteElements.FiniteElement,quadrature_order::Int, dirichlet_penalty = 1e60)
    # assemble system 
    A, b = assembleStokesSystem(nu, volume_data!,FE_velocity,FE_pressure,quadrature_order);
    
    # apply boundary data
    bdofs = FESolveCommon.computeDirichletBoundaryData!(val4dofs,FE_velocity,boundary_data!);
    for i = 1 : length(bdofs)
       A[bdofs[i],bdofs[i]] = dirichlet_penalty;
       b[bdofs[i]] = val4dofs[bdofs[i]]*dirichlet_penalty;
    end
    
    # remove one pressure dof (todo: don't do this with Neumann boundary)
    fixed_pressure_dof = FE_velocity.ndofs + 1;
    println("fixing one pressure dof with dofnr=",fixed_pressure_dof);
    A[fixed_pressure_dof,fixed_pressure_dof] = 1e30;
    b[fixed_pressure_dof] = 0;
    val4dofs[fixed_pressure_dof] = 0;
    
    println("solve");
    try
        val4dofs[:] = A\b;
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
    integral_mean = integrate_xref(FESolveCommon.eval_FEfunction(val4dofs[FE_velocity.ndofs+1:FE_velocity.ndofs+FE_pressure.ndofs],FE_pressure),FE_pressure.grid, FE_pressure.polynomial_order);
    integral_mean ./= sum(FE_pressure.grid.volume4cells)
    for i=1:FE_pressure.ndofs
        val4dofs[FE_velocity.ndofs+i] -= integral_mean[1] 
    end
    
    return norm(residual)
end

end
