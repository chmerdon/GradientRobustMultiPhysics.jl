function assemble_ugradu_matrix4FE!(A::ExtendableSparseMatrix, dofs4a, FE::AbstractH1FiniteElement)
    ncells::Int = size(FE.grid.nodes4cells,1);
    T = eltype(FE.grid.coords4nodes);
    ET = FE.grid.elemtypes[1]
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    xdim::Int = size(FE.grid.coords4nodes,2);
    celldim::Int = size(FE.grid.nodes4cells,2);
    ncomponents::Int = FiniteElements.get_ncomponents(FE);

    @assert xdim == ncomponents
    quadorder = 3*FiniteElements.get_polynomial_order(FE) - 2
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
    
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
    if ncomponents > 1
        basisvals = Array{Array{T,2}}(undef,length(qf.w));
    else
        basisvals = Array{Array{T,1}}(undef,length(qf.w));
    end    
    for i in eachindex(qf.w)
        # evaluate basis functions at quadrature point
        basisvals[i] = FiniteElements.get_basis_on_elemtype(FE, ET)(qf.xref[i])
    end    

    loc2glob_trafo_tinv = Grid.local2global_tinv_jacobian(FE.grid,ET)
          
    # quadrature loop
    temp::T = 0.0;
    a4qp = zeros(Float64,xdim)
    agradu = zeros(Float64,xdim)
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

        # compute a in quadrature point
        fill!(a4qp,0.0)
        for k = 1 : xdim
            for dof_i = 1 : ndofs4cell
                a4qp[k] += dofs4a[dofs[dof_i]] * basisvals[i][dof_i,k] * coefficients[dof_i,k]
            end
        end
        
        # fill sparse array
        for dof_i = 1 : ndofs4cell
            # compute agradu
            fill!(agradu,0.0)
            for k = 1 : xdim
                for c = 1 : ncomponents
                    agradu[k] += a4qp[c]*gradients4cell[offsets[k]+dof_i][c] * coefficients[dof_i,k]
                end
            end    
            for dof_j = 1 : ndofs4cell
                temp = 0.0;
                for k = 1 : xdim
                    temp += agradu[k] * basisvals[i][dof_j,k] * coefficients[dof_j,k];
                end
                temp *= qf.w[i] * FE.grid.volume4cells[cell]
                A[dofs[dof_j],dofs[dof_i]] += temp;   
            end
        end 
      end  
    end
    #end
end