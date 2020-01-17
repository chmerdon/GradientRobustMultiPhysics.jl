module FESolveCommon

export accumarray, computeBestApproximation!, computeFEInterpolation!, eval_interpolation_error!, eval_L2_interpolation_error!

using SparseArrays
using ExtendableSparse
using LinearAlgebra
using BenchmarkTools
using FiniteElements
using DiffResults
using ForwardDiff
using Grid
using Quadrature

function accumarray!(A,subs, val, sz=(maximum(subs),))
    for i = 1:length(val)
        A[subs[i]] += val[i]
    end
end



# matrix for L2 bestapproximation that writes into an ExtendableSparseMatrix
function assemble_mass_matrix4FE!(A::ExtendableSparseMatrix,FE::AbstractH1FiniteElement, talky = false)
    ncells::Int = size(FE.grid.nodes4cells,1);
    ndofs4cell::Int = FiniteElements.get_maxndofs4cell(FE);
    xdim::Int = size(FE.grid.coords4nodes,2);
    
    T = eltype(FE.grid.coords4nodes);
    qf = QuadratureFormula{T}(2*(FiniteElements.get_polynomial_order(FE)), xdim);
    
    if talky
        Quadrature.show(qf)
    end    
     
    # pre-allocate memory for basis functions
    ncomponents = FiniteElements.get_ncomponents(FE);
    basisvals = Array{Array{T,2}}(undef,length(qf.w));
    for i in eachindex(qf.w)
        basisvals[i] = zeros(T,ndofs4cell,ncomponents)
    end    
    dofs = zeros(Int64,ndofs4cell)
    
    # quadrature loop
    temp = 0.0;
    @time begin    
        for cell = 1 : ncells
        
            # get dofs
            for dof_i = 1 : ndofs4cell
                dofs[dof_i] = FiniteElements.get_globaldof4cell(FE, cell, dof_i);
            end
            
            for i in eachindex(qf.w)
                
                # evaluate basis functions at quadrature point
                if (cell == 1)
                    basisvals[i][:] = FiniteElements.get_all_basis_functions_on_cell(FE, cell)(qf.xref[i])
                end    
                
                
                for dof_i = 1 : ndofs4cell, dof_j = dof_i : ndofs4cell
                    # fill upper right part and diagonal of matrix
                    @inbounds begin
                      temp = 0.0
                      for k = 1 : ncomponents
                        temp += (basisvals[i][dof_i,k]*basisvals[i][dof_j,k] * qf.w[i] * FE.grid.volume4cells[cell]);
                      end
                      A[dofs[dof_i],dofs[dof_j]] += temp;
                      # fill lower left part of matrix
                      if dof_j > dof_i
                        A[dofs[dof_j],dofs[dof_i]] += temp;
                      end    
                    end
                end
            end
        end#
    end    
end


# stiffness matrix assembly that writes into an ExtendableSparseMatrix
function assemble_stiffness_matrix4FE!(A::ExtendableSparseMatrix,FE::AbstractH1FiniteElement,talky = false)
    ncells::Int = size(FE.grid.nodes4cells,1);
    ndofs4cell::Int = FiniteElements.get_maxndofs4cell(FE);
    xdim::Int = size(FE.grid.coords4nodes,2);
    celldim::Int = size(FE.grid.nodes4cells,2);
    
    T = eltype(FE.grid.coords4nodes);
    qf = QuadratureFormula{T}(2*(FiniteElements.get_polynomial_order(FE)-1), xdim);
    
    if talky
        Quadrature.show(qf)
    end    
    
    # pre-allocate memory for gradients
    gradients4cell = Array{Array{T,1}}(undef,ndofs4cell);
    for j = 1: ndofs4cell
        gradients4cell[j] = zeros(T,xdim);
    end
    
    # pre-allocate for derivatives of global2local trafo and basis function
    gradients_xref_cache = zeros(Float64,length(qf.w),ndofs4cell,celldim)
    #DRresult_grad = DiffResults.DiffResult(Vector{T}(undef, celldim), Matrix{T}(undef,ndofs4cell,celldim));
    trafo_jacobian = Matrix{T}(undef,xdim,xdim);
    dofs = zeros(Int64,ndofs4cell)
    
    dim = celldim - 1;
    if dim == 1
        loc2glob_trafo_tinv = FiniteElements.local2global_tinv_jacobian_line
    elseif dim == 2
        loc2glob_trafo_tinv = FiniteElements.local2global_tinv_jacobian_triangle
    end    
    
    # quadrature loop
    temp::T = 0.0;
    det::T = 0.0;
    @time begin
    for cell = 1 : ncells
      
      # evaluate tinverted (=transposed + inverted) jacobian of element trafo
      loc2glob_trafo_tinv(trafo_jacobian,det,FE.grid,cell)
      
      for dof_i = 1 : ndofs4cell
          dofs[dof_i] = FiniteElements.get_globaldof4cell(FE, cell, dof_i);
      end      
      
      for i in eachindex(qf.w)
      
        # evaluate gradients of basis function
        #ForwardDiff.jacobian!(DRresult_grad,FiniteElements.get_all_basis_functions_on_cell(FE, cell),qf.xref[i]);
        if (cell == 1)
            gradients_xref_cache[i,:,:] = ForwardDiff.jacobian(FiniteElements.get_all_basis_functions_on_cell(FE, cell),qf.xref[i]);
        end    
        
        
        # multiply tinverted jacobian of element trafo with gradient of basis function
        # which yields (by chain rule) the gradient in x coordinates
        for dof_i = 1 : ndofs4cell
            for k = 1 : xdim
                gradients4cell[dof_i][k] = 0.0;
                for j = 1 : xdim
                    gradients4cell[dof_i][k] += trafo_jacobian[k,j]*gradients_xref_cache[i,dof_i,j]
                end    
            end    
        end    
        
        # fill sparse array
        for dof_i = 1 : ndofs4cell, dof_j = dof_i : ndofs4cell
            # fill upper right part and diagonal of matrix
            temp = 0.0;
            for k = 1 : xdim
              temp += (gradients4cell[dof_i][k]*gradients4cell[dof_j][k] * qf.w[i] * FE.grid.volume4cells[cell]);
            end
            A[dofs[dof_i],dofs[dof_j]] += temp;
            # fill lower left part of matrix
            if dof_j > dof_i
              A[dofs[dof_j],dofs[dof_i]] += temp;
            end    
          end
      end  
    end
    end
end


function assemble_rhsL2!(b, f!::Function, FE::AbstractH1FiniteElement)
    ncells::Int = size(FE.grid.nodes4cells,1);
    ndofs4cell::Int = FiniteElements.get_maxndofs4cell(FE);
    xdim::Int = size(FE.grid.coords4nodes,2);
    
    T = eltype(FE.grid.coords4nodes);
    qf = QuadratureFormula{T}(2*(FiniteElements.get_polynomial_order(FE)), xdim);
     
    # pre-allocate memory for basis functions
    ncomponents = FiniteElements.get_ncomponents(FE);
    basisvals = Array{Array{T,2}}(undef,length(qf.w));
    for i in eachindex(qf.w)
        basisvals[i] = zeros(T,ndofs4cell,ncomponents)
    end    
    dofs = zeros(Int64,ndofs4cell)
    
    dim = size(FE.grid.nodes4cells,2) - 1;
    if dim == 1
        loc2glob_trafo = FiniteElements.local2global_line()
    elseif dim == 2
        loc2glob_trafo = FiniteElements.local2global_triangle()
    end    
    
    # quadrature loop
    temp = 0.0;
    fval = zeros(T,ncomponents)
    @time begin    
        for cell = 1 : ncells
            # get dofs
            for dof_i = 1 : ndofs4cell
                dofs[dof_i] = FiniteElements.get_globaldof4cell(FE, cell, dof_i);
            end
            for i in eachindex(qf.w)
                
                # evaluate basis functions at quadrature point
                if (cell == 1)
                    basisvals[i][:] = FiniteElements.get_all_basis_functions_on_cell(FE, cell)(qf.xref[i])
                end    
                
                # evaluate f
                x = loc2glob_trafo(FE.grid,cell)(qf.xref[i]);
                f!(fval, x)
                
                for dof_i = 1 : ndofs4cell
                    # fill vector
                    @inbounds begin
                      temp = 0.0
                      for k = 1 : ncomponents
                        temp += (fval[k]*basisvals[i][dof_i,k] * qf.w[i] * FE.grid.volume4cells[cell]);
                      end
                      b[dofs[dof_i]] += temp;
                    end
                end
            end
        end
    end    
end


function assemble_rhsH1!(b, f!::Function, FE::AbstractH1FiniteElement)
    ncells::Int = size(FE.grid.nodes4cells,1);
    ndofs4cell::Int = FiniteElements.get_maxndofs4cell(FE);
    xdim::Int = size(FE.grid.coords4nodes,2);
    celldim::Int = size(FE.grid.nodes4cells,2);
    
    T = eltype(FE.grid.coords4nodes);
    qf = QuadratureFormula{T}(2*(FiniteElements.get_polynomial_order(FE)), xdim);
     
    # pre-allocate memory for gradients
    gradients4cell = Array{Array{T,1}}(undef,ndofs4cell);
    for j = 1: ndofs4cell
        gradients4cell[j] = zeros(T,xdim);
    end
    
    # pre-allocate for derivatives of global2local trafo and basis function
    gradients_xref_cache = zeros(Float64,length(qf.w),ndofs4cell,celldim)
    #DRresult_grad = DiffResults.DiffResult(Vector{T}(undef, celldim), Matrix{T}(undef,ndofs4cell,celldim));
    trafo_jacobian = Matrix{T}(undef,xdim,xdim);
    dofs = zeros(Int64,ndofs4cell)
    
    dim = celldim - 1;
    if dim == 1
        loc2glob_trafo = FiniteElements.local2global_line()
        loc2glob_trafo_tinv = FiniteElements.local2global_tinv_jacobian_line
    elseif dim == 2
        loc2glob_trafo = FiniteElements.local2global_triangle()
        loc2glob_trafo_tinv = FiniteElements.local2global_tinv_jacobian_triangle
    end    
    
    
    # quadrature loop
    temp = 0.0;
    fval = zeros(T,xdim)
    @time begin    
        for cell = 1 : ncells
      
            # evaluate tinverted (=transposed + inverted) jacobian of element trafo
            loc2glob_trafo_tinv(trafo_jacobian,det,FE.grid,cell)
      
            for dof_i = 1 : ndofs4cell
                dofs[dof_i] = FiniteElements.get_globaldof4cell(FE, cell, dof_i);
            end      
      
            for i in eachindex(qf.w)
      
                # evaluate gradients of basis function
                #ForwardDiff.jacobian!(DRresult_grad,FiniteElements.get_all_basis_functions_on_cell(FE, cell),qf.xref[i]);
                if (cell == 1)
                    gradients_xref_cache[i,:,:] = ForwardDiff.jacobian(FiniteElements.get_all_basis_functions_on_cell(FE, cell),qf.xref[i]);
                end    
        
        
                # multiply tinverted jacobian of element trafo with gradient of basis function
                # which yields (by chain rule) the gradient in x coordinates
                for dof_i = 1 : ndofs4cell
                    for k = 1 : xdim
                        gradients4cell[dof_i][k] = 0.0;
                        for j = 1 : xdim
                            gradients4cell[dof_i][k] += trafo_jacobian[k,j]*gradients_xref_cache[i,dof_i,j]
                        end    
                    end    
                end 
                
                # evaluate f
                x = loc2glob_trafo(FE.grid,cell)(qf.xref[i]);
                f!(fval, x)
                
                for dof_i = 1 : ndofs4cell
                    # fill vector
                    @inbounds begin
                      temp = 0.0
                      for k = 1 : xdim
                        temp += (fval[k] * gradients4cell[dof_i][k] * qf.w[i] * FE.grid.volume4cells[cell]);
                      end
                      b[dofs[dof_i]] += temp;
                    end
                end
            end
        end
    end    
end


# vector functions times FE basis functions gradients
function rhs_integrandH1!(f!::Function,FE::AbstractH1FiniteElement,dim)
    cache = zeros(eltype(FE.grid.coords4nodes),dim)
    ndofs4cell::Int = FiniteElements.get_maxndofs4cell(FE);
    basisvals = zeros(eltype(FE.grid.coords4nodes),ndofs4cell,dim)
    T = eltype(FE.grid.coords4nodes);
    celldim::Int = size(FE.grid.nodes4cells,2);
    xdim::Int = size(FE.grid.coords4nodes,2);
    DRresult_grad = DiffResults.DiffResult(Vector{T}(undef, xdim), Matrix{T}(undef,ndofs4cell,xdim));
    trafo_jacobian = Matrix{T}(undef,xdim,xdim);
    dim = celldim - 1;
    if dim == 1
        loc2glob_trafo_tinv = FiniteElements.local2global_tinv_jacobian_line
    elseif dim == 2
        loc2glob_trafo_tinv = FiniteElements.local2global_tinv_jacobian_triangle
    end    
    function closure(result,x,xref,cell)
        f!(cache, x)
        # evaluate tinverted (=transposed + inverted) jacobian of element trafo
        loc2glob_trafo_tinv(trafo_jacobian,det,FE.grid,cell)
      
        # evaluate gradients of basis function
        ForwardDiff.jacobian!(DRresult_grad,FiniteElements.get_all_basis_functions_on_cell(FE, cell),xref);
        
        for j=1:ndofs4cell
            result[j] = 0.0
            for d=1:dim
                for k = 1 : xdim
                    result[j] += cache[d] * trafo_jacobian[d,k]*DiffResults.gradient(DRresult_grad)[j,k]
                end    
            end    
        end
    end
end


function assembleSystem(norm_lhs::String,norm_rhs::String,volume_data!::Function,grid::Grid.Mesh,FE::AbstractFiniteElement,quadrature_order::Int)

    ncells::Int = size(grid.nodes4cells,1);
    nnodes::Int = size(grid.coords4nodes,1);
    celldim::Int = size(grid.nodes4cells,2);
    xdim::Int = size(grid.coords4nodes,2);
    
    Grid.ensure_volume4cells!(grid);
    
    ndofs = FiniteElements.get_ndofs(FE);
    A = ExtendableSparseMatrix{Float64,Int64}(ndofs,ndofs);
    if norm_lhs == "L2"
        assemble_mass_matrix4FE!(A,FE);
    elseif norm_lhs == "H1"
        assemble_stiffness_matrix4FE!(A,FE);
    end 
    
    # compute right-hand side vector
    b = FiniteElements.createFEVector(FE);
    if norm_rhs == "L2"
        assemble_rhsL2!(b, volume_data!, FE)
    elseif norm_rhs == "H1"
        assemble_rhsH1!(b, volume_data!, FE)
    end
    
    
    return A,b
end


function computeDirichletBoundaryData!(val4dofs,FE,boundary_data!)
 # find boundary dofs
    xdim = FiniteElements.get_ncomponents(FE);
    ndofs::Int = FiniteElements.get_ndofs(FE);
    
    bdofs = [];
    if (boundary_data! == Nothing)
    else
        Grid.ensure_bfaces!(FE.grid);
        Grid.ensure_cells4faces!(FE.grid);
        xref = zeros(eltype(FE.xref4dofs4cell),size(FE.xref4dofs4cell,2));
        temp = zeros(eltype(FE.grid.coords4nodes),xdim);
        dim = size(FE.grid.nodes4cells,2) - 1;
        if dim == 1
            loc2glob_trafo = FiniteElements.local2global_line()
        elseif dim == 2
            loc2glob_trafo = FiniteElements.local2global_triangle()
        end   
        cell::Int = 0;
        j::Int = 1;
        ndofs4cell = FiniteElements.get_maxndofs4cell(FE);
        ndofs4face = FiniteElements.get_maxndofs4face(FE);
        basisvals = zeros(eltype(FE.grid.coords4nodes),ndofs4cell,dim)
        A4bface = Matrix{Float64}(undef,ndofs4face,ndofs4face)
        b4bface = Vector{Float64}(undef,ndofs4face)
        bdofs4bface = Vector{Int}(undef,ndofs4face)
        celldof2facedof = zeros(Int,ndofs4face)
        for i in eachindex(FE.grid.bfaces)
            cell = FE.grid.cells4faces[FE.grid.bfaces[i],1];
            for k = 1:ndofs4face
                bdof = FiniteElements.get_globaldof4face(FE, FE.grid.bfaces[i], k);
                bdofs4bface[k] = bdof;
            end    
            append!(bdofs,bdofs4bface);
            # setup local system of equations to determine piecewise interpolation of boundary data
            # find position of face dofs in cell dofs
            for j=1:ndofs4cell, k = 1 : ndofs4face
                    celldof = FiniteElements.get_globaldof4cell(FE, cell, j);
                    if celldof == bdofs4bface[k]
                        celldof2facedof[k] = j;
                    end   
            end
            # assemble matrix    
            for k = 1:ndofs4face
                for l = 1 : length(xref)
                    xref[l] = FE.xref4dofs4cell[celldof2facedof[k],l];
                end    
                basisvals = FiniteElements.get_all_basis_functions_on_cell(FE, cell)(xref)
                for l = 1:ndofs4face
                    A4bface[k,l] = dot(basisvals[celldof2facedof[k],:],basisvals[celldof2facedof[l],:]);
                end
                
                boundary_data!(temp,loc2glob_trafo(FE.grid,cell)(xref));
                b4bface[k] = dot(temp,basisvals[celldof2facedof[k],:]);
            end
            val4dofs[bdofs4bface] = A4bface\b4bface;
            if norm(A4bface*val4dofs[bdofs4bface]-b4bface) > eps(1e3)
                println("WARNING: large residual, boundary data may be inexact");
            end
        end    
    end
    return unique(bdofs)
end

# computes Bestapproximation in approx_norm="L2" or "H1"
# volume_data! for norm="H1" is expected to be the gradient of the function that is bestapproximated
function computeBestApproximation!(val4dofs::Array,approx_norm::String ,volume_data!::Function,boundary_data!,grid::Grid.Mesh,FE::AbstractFiniteElement,quadrature_order::Int, dirichlet_penalty = 1e60)
    # assemble system 
    A, b = assembleSystem(approx_norm,approx_norm,volume_data!,grid,FE,quadrature_order);
    
    # apply boundary data
    bdofs = computeDirichletBoundaryData!(val4dofs,FE,boundary_data!);
    for i = 1 : length(bdofs)
       A[bdofs[i],bdofs[i]] = dirichlet_penalty;
       b[bdofs[i]] = val4dofs[bdofs[i]]*dirichlet_penalty;
    end

    # solve
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


function computeFEInterpolation!(val4dofs::Array,source_function!::Function,grid::Grid.Mesh,FE::AbstractFiniteElement)
    dim = size(FE.grid.nodes4cells,2) - 1;
    temp = zeros(Float64,FiniteElements.get_ncomponents(FE));
    xref = zeros(eltype(FE.xref4dofs4cell),dim);
    ndofs4cell = FiniteElements.get_maxndofs4cell(FE);
    if dim == 1
        loc2glob_trafo = FiniteElements.local2global_line()
    elseif dim == 2
        loc2glob_trafo = FiniteElements.local2global_triangle()
    end    
    for j = 1 : size(FE.grid.nodes4cells,1)
        for k = 1 : ndofs4cell
            for l = 1 : length(xref)
                xref[l] = FE.xref4dofs4cell[k,l];
            end    
            x = loc2glob_trafo(grid,j)(xref);
            source_function!(temp,x);
            val4dofs[FiniteElements.get_globaldof4cell(FE,j,k),:] = temp;
        end    
    end
end


function eval_FEfunction(coeffs, FE::AbstractFiniteElement)
    temp = zeros(Float64,FiniteElements.get_ncomponents(FE));
    ndofs4cell = FiniteElements.get_maxndofs4cell(FE);
    basisvals = zeros(eltype(FE.grid.coords4nodes),ndofs4cell,FiniteElements.get_ncomponents(FE))
    function closure(result, x, xref, cellIndex)
        fill!(result,0.0)
        basisvals = FiniteElements.get_all_basis_functions_on_cell(FE, cellIndex)(xref)
        for j = 1 : ndofs4cell
            di = FiniteElements.get_globaldof4cell(FE, cellIndex, j);
            for k = 1 : length(temp);
                result[k] += basisvals[j,k] * coeffs[di];
            end    
        end    
    end
end

function eval_L2_interpolation_error!(exact_function!, coeffs_interpolation, FE::AbstractFiniteElement)
    temp = zeros(Float64,FiniteElements.get_ncomponents(FE));
    ndofs4cell = FiniteElements.get_maxndofs4cell(FE);
    ncomponents = FiniteElements.get_ncomponents(FE);
    basisvals = zeros(eltype(FE.grid.coords4nodes),ndofs4cell,ncomponents)
    function closure(result, x, xref, cellIndex)
        # evaluate exact function
        exact_function!(result, x);
        # subtract nodal interpolation
        basisvals = FiniteElements.get_all_basis_functions_on_cell(FE, cellIndex)(xref)
        for j = 1 : ndofs4cell
            di = FiniteElements.get_globaldof4cell(FE, cellIndex, j);
            for k = 1 : length(temp);
                result[k] -= basisvals[j,k] * coeffs_interpolation[di];
            end    
        end   
        # square for L2 norm
        for j = 1 : length(result)
            result[j] = result[j]^2
        end    
    end
end


function eval_at_nodes(val4dofs, FE::AbstractFiniteElement, offset::Int64 = 0)
    # evaluate at grid points
    ndofs4node = zeros(size(FE.grid.coords4nodes,1))
    dim = size(FE.grid.nodes4cells,2) - 1;
    if dim == 1
        xref4dofs4cell = Array{Float64,2}([0,1]')';
    elseif dim == 2
        xref4dofs4cell = [0 0; 1 0; 0 1];
    end    
    di = 0;
    temp = 0;
    ncomponents = FiniteElements.get_ncomponents(FE);
    ndofs4cell = FiniteElements.get_maxndofs4cell(FE);
    basisvals = zeros(eltype(FE.grid.coords4nodes),ndofs4cell,ncomponents)
    nodevals = zeros(size(FE.grid.coords4nodes,1),ncomponents)
    for cell = 1 : size(FE.grid.nodes4cells,1)
        for j = 1 : dim + 1
            basisvals = FiniteElements.get_all_basis_functions_on_cell(FE, cell)(xref4dofs4cell[j,:])
            for dof = 1 : ndofs4cell;
                for k = 1 : ncomponents
                    di = offset + FiniteElements.get_globaldof4cell(FE, cell, dof);
                    nodevals[FE.grid.nodes4cells[cell,j],k] += basisvals[dof,k]*val4dofs[di] 
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
