function assemble_divdiv_Matrix!(A::ExtendableSparseMatrix, FE_velocity::FiniteElements.AbstractH1FiniteElement)

    T = eltype(FE_velocity.grid.coords4nodes);
    ET = FE_velocity.grid.elemtypes[1]
    ndofs4cell_velocity::Int = FiniteElements.get_ndofs4elemtype(FE_velocity, ET);
    ncomponents::Int = FiniteElements.get_ncomponents(FE_velocity);
    ndofs_velocity = FiniteElements.get_ndofs(FE_velocity);
    ncells::Int = size(FE_velocity.grid.nodes4cells,1);
    xdim::Int = size(FE_velocity.grid.coords4nodes,2);
    
    @assert ncomponents == xdim
    
    quadorder = 2*(FiniteElements.get_polynomial_order(FE_velocity)-1);
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);

    # pre-allocate gradients of basis function in reference coordinates
    celldim::Int = size(FE_velocity.grid.nodes4cells,2);
    gradients_xref_cache = zeros(Float64,length(qf.w),xdim*ndofs4cell_velocity,celldim)
    for i in eachindex(qf.w)
        # evaluate gradients of basis function
        gradients_xref_cache[i,:,:] = ForwardDiff.jacobian(FiniteElements.get_basis_on_elemtype(FE_velocity, ET),qf.xref[i]);
    end    
    
    # determine needed trafo
    loc2glob_trafo_tinv = Grid.local2global_tinv_jacobian(FE_velocity.grid,ET)
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
    offsets = [0,ndofs4cell_velocity];
    
    # audrature loop
    #@time begin
    for cell = 1 : ncells
      # evaluate tinverted (=transposed + inverted) jacobian of element trafo
      loc2glob_trafo_tinv(trafo_jacobian,cell)
      
      # get dofs
      FiniteElements.get_dofs_on_cell!(dofs_velocity,FE_velocity, cell, ET);
      
      # get coefficients
      FiniteElements.get_basis_coefficients_on_cell!(coefficients_velocity,FE_velocity,cell, ET);
      
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
            div_i *= qf.w[i] * FE_velocity.grid.volume4cells[cell];
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
    #end  
end