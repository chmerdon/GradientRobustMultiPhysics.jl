function assemble_stiffness_matrix4FE!(A::ExtendableSparseMatrix,nu::Real,FE::AbstractFiniteElement)
    
    # get quadrature formula
    T = eltype(FE.grid.coords4nodes);
    ET = FE.grid.elemtypes[1]
    quadorder = 2*(FiniteElements.get_polynomial_order(FE)-1);
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
    
    # generate caller for FE basis functions
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    ncomponents::Int = FiniteElements.get_ncomponents(FE);
    xdim = size(FE.grid.coords4nodes,2)
    FEbasis = FiniteElements.FEbasis_caller(FE, qf, true);
    gradients = zeros(Float64,ndofs4cell,ncomponents*xdim);
    dofs = zeros(Int64,ndofs4cell)
          
    # quadrature loop
    temp = 0.0;
    #@time begin
    for cell = 1 : size(FE.grid.nodes4cells,1)

        # get dofs
        FiniteElements.get_dofs_on_cell!(dofs, FE, cell, ET);

        # update FEbasis on cell
        FiniteElements.updateFEbasis!(FEbasis, cell)
      
        for i in eachindex(qf.w)
        
            # get FE basis gradients at quadrature point
            FiniteElements.getFEbasisgradients4qp!(gradients, FEbasis, i)

            # fill sparse array
            for dof_i = 1 : ndofs4cell, dof_j = dof_i : ndofs4cell
                # fill upper right part and diagonal of matrix
                temp = 0.0;
                for k = 1 : size(gradients,2)
                    temp += gradients[dof_i,k]*gradients[dof_j,k];
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