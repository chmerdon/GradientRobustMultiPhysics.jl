function assemble_stiffness_matrix4FE!(A::ExtendableSparseMatrix,nu::Real,FE::AbstractH1FiniteElement)
    ncells::Int = size(FE.grid.nodes4cells,1);
    T = eltype(FE.grid.coords4nodes);
    ET = FE.grid.elemtypes[1]
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    xdim::Int = size(FE.grid.coords4nodes,2);
    celldim::Int = size(FE.grid.nodes4cells,2);
    ncomponents::Int = FiniteElements.get_ncomponents(FE);
    
    qf = QuadratureFormula{T,typeof(ET)}(2*(FiniteElements.get_polynomial_order(FE)-1));
    
    # pre-allocate for derivatives of global2local trafo and basis function
    gradients_xref_cache = zeros(Float64,length(qf.w),ncomponents*ndofs4cell,celldim)
    #DRresult_grad = DiffResults.DiffResult(Vector{T}(undef, celldim), Matrix{T}(undef,ndofs4cell,celldim));
    trafo_jacobian = Matrix{T}(undef,xdim,xdim);
    dofs = zeros(Int64,ndofs4cell)
    gradients4cell = Array{Array{T,1}}(undef,ncomponents*ndofs4cell);
    coefficients = zeros(Float64,ndofs4cell,xdim)
    for j = 1 : ncomponents*ndofs4cell
        gradients4cell[j] = zeros(T,xdim);
    end
    for i in eachindex(qf.w)
        # evaluate gradients of basis function
        gradients_xref_cache[i,:,:] = ForwardDiff.jacobian(FiniteElements.get_basis_on_elemtype(FE, ET),qf.xref[i]);
    end    

    loc2glob_trafo_tinv = Grid.local2global_tinv_jacobian(FE.grid,ET)
          
    # quadrature loop
    temp::T = 0.0;
    offsets = [0,ndofs4cell];
    #@time begin
    for cell = 1 : ncells
      
      # evaluate tinverted (=transposed + inverted) jacobian of element trafo
      loc2glob_trafo_tinv(trafo_jacobian,cell)

      # get dofs
      FiniteElements.get_dofs_on_cell!(dofs, FE, cell, ET);

      # get coefficients
      FiniteElements.get_basis_coefficients_on_cell!(coefficients, FE, cell, ET);
      
      for i in eachindex(qf.w)
        
        # multiply tinverted jacobian of element trafo with gradient of basis function
        # which yields (by chain rule) the gradient in x coordinates
        for dof_i = 1 : ndofs4cell
            for c=1 : ncomponents, k = 1 : xdim
                gradients4cell[dof_i + offsets[c]][k] = 0.0;
                for j = 1 : xdim
                    gradients4cell[dof_i + offsets[c]][k] += trafo_jacobian[k,j]*gradients_xref_cache[i,dof_i + offsets[c],j] * coefficients[dof_i,c]
                end    
            end    
        end    
        
        # fill sparse array
        for dof_i = 1 : ndofs4cell, dof_j = dof_i : ndofs4cell
            # fill upper right part and diagonal of matrix
            temp = 0.0;
            for k = 1 : xdim
                for c = 1 : ncomponents
                    temp += gradients4cell[offsets[c]+dof_i][k]*gradients4cell[offsets[c]+dof_j][k];
                end
            end
            temp *= nu * qf.w[i] * FE.grid.volume4cells[cell]
            A[dofs[dof_i],dofs[dof_j]] += temp;
            # fill lower left part of matrix
            if dof_j > dof_i
              A[dofs[dof_j],dofs[dof_i]] += temp;
            end    
          end
      end  
    end
    #end
end