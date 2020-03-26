struct CELL_DIVUdotDIVV <: FiniteElements.AbstractFEOperator end

function assemble_operator!(A::ExtendableSparseMatrix, ::Type{CELL_DIVUdotDIVV}, FE::FiniteElements.AbstractFiniteElement, factor::Real = 1.0)

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
    divergences = zeros(Float64,ndofs4cell,1);
    dofs = zeros(Int64,ndofs4cell)
          
    # quadrature loop
    #@time begin
    for cell = 1 : size(FE.grid.nodes4cells,1)
      # get dofs
      FiniteElements.get_dofs_on_cell!(dofs,FE, cell, ET);
      
      # update FEbasis on cell
      FiniteElements.updateFEbasis!(FEbasis, cell)
      
      for i in eachindex(qf.w)
        
        # get FE basis gradients at quadrature point
        FiniteElements.getFEbasisdivergence4qp!(divergences, FEbasis, i)
        
        # div x div
        for dof_i = 1 : ndofs4cell, dof_j = 1 : ndofs4cell
            A[dofs[dof_i],dofs[dof_j]] += divergences[dof_i] * divergences[dof_j] * factor * qf.w[i] * FE.grid.volume4cells[cell];
        end    
      end  
    end
    #end  
end