struct CELL_MdotDUdotDV <: FiniteElements.AbstractFEOperator end

function assemble_operator!(A::ExtendableSparseMatrix, ::Type{CELL_MdotDUdotDV}, FE::AbstractFiniteElement, M!::Function, M_quadorder::Int)
    
    # get quadrature formula
    T = eltype(FE.grid.coords4nodes);
    ET = FE.grid.elemtypes[1]
    quadorder = M_quadorder + 2*(FiniteElements.get_polynomial_order(FE)-1);
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
    
    # generate caller for FE basis functions
    FEbasis = FiniteElements.FEbasis_caller(FE, qf, true);
    gradients = zeros(Float64,FEbasis.ndofs4item,FEbasis.ncomponents*FEbasis.xdim);
          
    loc2glob_trafo = Grid.local2global(FE.grid,ET)

    # quadrature loop
    temp = 0.0;
    Mval = zeros(T,FEbasis.xdim,FEbasis.ncomponents*FEbasis.xdim)
    MDui = zeros(T,FEbasis.ncomponents*FEbasis.xdim)
    x = zeros(T,FEbasis.xdim)
    #@time begin
    for cell = 1 : size(FE.grid.nodes4cells,1)

        # update FEbasis on cell
        FiniteElements.updateFEbasis!(FEbasis, cell)
      
        # get trafo
        cell_trafo = loc2glob_trafo(cell)

        for i in eachindex(qf.w)
        
            # get FE basis gradients at quadrature point
            FiniteElements.getFEbasisgradients4qp!(gradients, FEbasis, i)

            # evaluate (matrix-valued) M
            x[:] = cell_trafo(qf.xref[i]);
            M!(Mval, x)

            # fill sparse array
            for dof_i = 1 : FEbasis.ndofs4item
                # compute M times gradient of i-th basis function
                fill!(MDui,0.0)
                for k = 1 : FEbasis.xdim, l = 1 : size(gradients,2)
                    MDui[k] += Mval[k,l]*gradients[dof_i,l]
                end    

                for dof_j = dof_i : FEbasis.ndofs4item
                    # fill upper right part and diagonal of matrix
                    temp = 0.0;
                    for k = 1 : size(gradients,2)
                        temp += MDui[k]*gradients[dof_j,k];
                    end
                    temp *= qf.w[i] * FE.grid.volume4cells[cell]
                    A[FEbasis.current_dofs[dof_i],FEbasis.current_dofs[dof_j]] += temp;
                    # fill lower left part of matrix
                    if dof_j > dof_i
                        A[FEbasis.current_dofs[dof_j],FEbasis.current_dofs[dof_i]] += temp;
                    end    
                end
            end
        end  
    end
    #end
end