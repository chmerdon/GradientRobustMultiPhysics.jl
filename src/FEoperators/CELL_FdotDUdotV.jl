struct CELL_FdotDUdotV <: FiniteElements.AbstractFEOperator end

function assemble_operator!(A::ExtendableSparseMatrix, ::Type{CELL_FdotDUdotV}, FE::AbstractFiniteElement, f!::Function, quadrature_order::Int)
    
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
    basisvals = zeros(Float64,ndofs4cell,ncomponents);
    dofs = zeros(Int64,ndofs4cell)

    loc2glob_trafo = Grid.local2global(FE.grid,ET)
          
    # quadrature loop
    temp = 0.0;
    fval = zeros(Float64,ncomponents*xdim)
    x = zeros(Float64,xdim)
    #@time begin
    for cell = 1 : size(FE.grid.nodes4cells,1)

        # get dofs
        FiniteElements.get_dofs_on_cell!(dofs, FE, cell, ET);

        # update FEbasis on cell
        FiniteElements.updateFEbasis!(FEbasis, cell)

        # get trafo
        cell_trafo = loc2glob_trafo(cell)
      
        for i in eachindex(qf.w)
        
            # get FE basisvals and basis gradients at quadrature point
            FiniteElements.getFEbasisgradients4qp!(gradients, FEbasis, i)
            FiniteElements.getFEbasis4qp!(basisvals, FEbasis, i)

            # evaluate f
            x[:] = cell_trafo(qf.xref[i]);
            f!(fval, x)

            # fill sparse array
            for dof_i = 1 : ndofs4cell
                temp = 0.0;
                for k = 1 : size(gradients,2)
                    temp += gradients[dof_i,k]*fval[k];
                end
                for dof_j = 1 : ndofs4cell
                    A[dofs[dof_i],dofs[dof_j]] += temp * basisvals[dof_j] * qf.w[i] * FE.grid.volume4cells[cell];
                end    
            end
        end  
    end
    #end
end