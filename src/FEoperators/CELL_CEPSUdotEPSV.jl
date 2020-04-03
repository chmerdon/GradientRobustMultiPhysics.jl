struct CELL_CEPSUdotEPSV <: FiniteElements.AbstractFEOperator end

function assemble_operator!(A::ExtendableSparseMatrix, ::Type{CELL_CEPSUdotEPSV}, FE::AbstractFiniteElement, mu::Real = 1.0, lambda::Real = 1.0)
    
    # get quadrature formula
    T = eltype(FE.grid.coords4nodes);
    ET = FE.grid.elemtypes[1]
    quadorder = 2*(FiniteElements.get_polynomial_order(FE)-1);
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
    
    # generate caller for FE basis functions
    FEbasis = FiniteElements.FEbasis_caller(FE, qf, true);
    lsym = (FEbasis.xdim == 2) ? 3 : 6;
    symgradients = zeros(Float64,FEbasis.ndofs4item,lsym);

    # compute Cauchy stress tensor
    C = zeros(Float64,lsym,lsym)
    if FEbasis.xdim == 2
        C[1,1] = lambda + 2*mu
        C[2,2] = C[1,1]
        C[2,1] = lambda
        C[1,2] = lambda
        C[3,3] = mu
    elseif FEbasis.xdim == 3
        C[1,1] = lambda + 2*mu
        C[2,2] = C[1,1]
        C[3,3] = C[1,1]
        C[2,1] = lambda
        C[1,2] = lambda
        C[3,1] = lambda
        C[1,3] = lambda
        C[2,3] = lambda
        C[3,2] = lambda
        C[4,4] = mu
        C[5,5] = mu
        C[6,6] = mu
    end

          
    # quadrature loop
    temp = 0.0;
    temp2 = 0.0;
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
                # compute dot(C eps(u), eps(v))
                temp = 0.0;
                for k = 1 : lsym
                    temp2 = 0.0
                    for l = 1 : lsym
                        temp2 += C[k,l] * symgradients[dof_i,l]
                    end
                    temp += temp2 * symgradients[dof_j,k];
                end
                temp *= qf.w[i] * FE.grid.volume4cells[cell]
                A[FEbasis.current_dofs[dof_i],FEbasis.current_dofs[dof_j]] += temp;
                # fill lower left part of matrix (C is symmetric!!!)
                if dof_j > dof_i
                    A[FEbasis.current_dofs[dof_j],FEbasis.current_dofs[dof_i]] += temp;
                end    
            end
        end  
    end
    #end
end