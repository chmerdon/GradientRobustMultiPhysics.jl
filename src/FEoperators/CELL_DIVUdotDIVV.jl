struct CELL_DIVUdotDIVV <: FiniteElements.AbstractFEOperator end

function assemble_operator!(A::ExtendableSparseMatrix, ::Type{CELL_DIVUdotDIVV}, FE::FiniteElements.AbstractFiniteElement, factor::Real = 1.0)

    # get quadrature formula
    T = eltype(FE.grid.coords4nodes);
    ET = FE.grid.elemtypes[1]
    quadorder = 2*(FiniteElements.get_polynomial_order(FE)-1);
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
    
    # generate caller for FE basis functions
    FEbasis = FiniteElements.FEbasis_caller(FE, qf, true);
    divergences = zeros(Float64,FEbasis.ndofs4item,1);
          
    # quadrature loop
    #@time begin
    for cell = 1 : size(FE.grid.nodes4cells,1)
      
      # update FEbasis on cell
      FiniteElements.updateFEbasis!(FEbasis, cell)
      
      for i in eachindex(qf.w)
        
        # get FE basis gradients at quadrature point
        FiniteElements.getFEbasisdivergence4qp!(divergences, FEbasis, i)
        
        # div x div
        for dof_i = 1 : FEbasis.ndofs4item, dof_j = 1 : FEbasis.ndofs4item
            A[FEbasis.current_dofs[dof_i],FEbasis.current_dofs[dof_j]] += divergences[dof_i] * divergences[dof_j] * factor * qf.w[i] * FE.grid.volume4cells[cell];
        end    
      end  
    end
    #end  
end