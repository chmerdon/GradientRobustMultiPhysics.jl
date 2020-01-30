function assemble_rhsH1!(b, f!::Function, FE::AbstractH1FiniteElement, quadrature_order::Int)
    ncells::Int = size(FE.grid.nodes4cells,1);
    ET = FE.grid.elemtypes[1];
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    xdim::Int = size(FE.grid.coords4nodes,2);
    celldim::Int = size(FE.grid.nodes4cells,2);
    
    T = eltype(FE.grid.coords4nodes);
    qf = QuadratureFormula{T,typeof(ET)}(quadrature_order + min(0,FiniteElements.get_polynomial_order(FE) - 1));
     
    # pre-allocate memory for gradients
    gradients4cell = Array{Array{T,1}}(undef,ndofs4cell);
    for j = 1: ndofs4cell
        gradients4cell[j] = zeros(T,xdim);
    end
    gradients_xref_cache = zeros(Float64,length(qf.w),ndofs4cell,celldim)
    for i in eachindex(qf.w)
        # evaluate gradients of basis function
        gradients_xref_cache[i,:,:] = ForwardDiff.jacobian(FiniteElements.get_basis_on_elemtype(FE, ET),qf.xref[i]);
    end    
    #DRresult_grad = DiffResults.DiffResult(Vector{T}(undef, celldim), Matrix{T}(undef,ndofs4cell,celldim));
    trafo_jacobian = Matrix{T}(undef,xdim,xdim);
    dofs = zeros(Int64,ndofs4cell)
    
    dim = celldim - 1;
    loc2glob_trafo = Grid.local2global(FE.grid,ET)
    loc2glob_trafo_tinv = Grid.local2global_tinv_jacobian(FE.grid,ET)
    
    
    # quadrature loop
    temp = 0.0;
    fval = zeros(T,xdim)
    #@time begin    
        for cell = 1 : ncells
      
            # evaluate tinverted (=transposed + inverted) jacobian of element trafo
            loc2glob_trafo_tinv(trafo_jacobian,cell)
      
            # get dofs
            FiniteElements.get_dofs_on_cell!(dofs, FE, cell, ET);      

            # get trafo
            cell_trafo = loc2glob_trafo(cell)
      
            for i in eachindex(qf.w)
        
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
                x = cell_trafo(qf.xref[i]);
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
    #end    
end