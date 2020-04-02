struct CELL_EPSUdotEPSV <: FiniteElements.AbstractFEOperator end

function assemble_operator!(A::ExtendableSparseMatrix, ::Type{CELL_EPSUdotEPSV}, FE::AbstractFiniteElement, nu::Real = 1.0)
    
    # get quadrature formula
    T = eltype(FE.grid.coords4nodes);
    ET = FE.grid.elemtypes[1]
    quadorder = 2*(FiniteElements.get_polynomial_order(FE)-1);
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
    
    # generate caller for FE basis functions
    FEbasis = FiniteElements.FEbasis_caller(FE, qf, true);
    symgradients = zeros(Float64,FEbasis.ndofs4item,(FEbasis.xdim == 2) ? 3 : 6);
    voigtfactors = (FEbasis.xdim == 2) ? [1.0 1.0 0.5] : [1.0 1.0 1.0 0.5 0.5 0.5];
          
    # quadrature loop
    temp = 0.0;
    #@time begin
    for cell = 1 : size(FE.grid.nodes4cells,1)

        # update FEbasis on cell
        FiniteElements.updateFEbasis!(FEbasis, cell)
      
        for i in eachindex(qf.w)
        
            # get FE basis gradients at quadrature point
            FiniteElements.getFEbasissymgradients4qp!(symgradients, FEbasis, i)

            # fill sparse array
            for dof_i = 1 : FEbasis.ndofs4item, dof_j = dof_i : FEbasis.ndofs4item
                # fill upper right part and diagonal of matrix
                temp = 0.0;
                for k = 1 : size(symgradients,2)
                    temp += voigtfactors[k] * symgradients[dof_i,k] * symgradients[dof_j,k];
                end
                temp *= nu * qf.w[i] * FE.grid.volume4cells[cell]
                A[FEbasis.current_dofs[dof_i],FEbasis.current_dofs[dof_j]] += temp;
                # fill lower left part of matrix
                if dof_j > dof_i
                    A[FEbasis.current_dofs[dof_j],FEbasis.current_dofs[dof_i]] += temp;
                end    
            end
        end  
    end
    #end
end