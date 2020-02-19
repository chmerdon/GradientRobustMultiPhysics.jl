struct CELL_DIVUdotDIVV <: FiniteElements.AbstractFEOperator end

function assemble_operator!(A::ExtendableSparseMatrix, ::Type{CELL_DIVUdotDIVV}, FE::FiniteElements.AbstractFiniteElement)

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
    diagonal_entries = [1, 4]
          
    # quadrature loop
    temp = 0.0;
    div_i::T = 0.0;
    div_j::T = 0.0;
    #@time begin
    for cell = 1 : size(FE.grid.nodes4cells,1)
      # get dofs
      FiniteElements.get_dofs_on_cell!(dofs,FE, cell, ET);
      
      # update FEbasis on cell
      FiniteElements.updateFEbasis!(FEbasis, cell)
      
      for i in eachindex(qf.w)
        
        # get FE basis gradients at quadrature point
        FiniteElements.getFEbasisgradients4qp!(gradients, FEbasis, i)
        
        # div x div
        for dof_i = 1 : ndofs4cell
            div_i = 0.0;
            for c = 1 : length(diagonal_entries)
                div_i += gradients[dof_i,diagonal_entries[c]];
            end    
            div_i *= qf.w[i] * FE.grid.volume4cells[cell];
            for dof_j = 1 : ndofs4cell
                div_j = 0.0;
                for c = 1 : xdim
                    div_j += gradients[dof_j,diagonal_entries[c]];
                end    
                A[dofs[dof_i],dofs[dof_j]] += div_j*div_i;
            end
        end    
      end  
    end
    #end  
end