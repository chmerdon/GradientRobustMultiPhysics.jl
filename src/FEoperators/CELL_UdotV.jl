struct CELL_UdotV <: FiniteElements.AbstractFEOperator end

# matrix for L2 bestapproximation that writes into an ExtendableSparseMatrix
function assemble_operator!(A::ExtendableSparseMatrix,::Type{CELL_UdotV},FE::AbstractFiniteElement, factor::Real = 1.0)
    
    # get quadrature formula
    T = eltype(FE.grid.coords4nodes);
    ET = FE.grid.elemtypes[1]
    quadorder = 2*FiniteElements.get_polynomial_order(FE);
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
     
    # generate caller for FE basis functions
    ncells::Int = size(FE.grid.nodes4cells,1);
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    ncomponents::Int = FiniteElements.get_ncomponents(FE);
    FEbasis = FiniteElements.FEbasis_caller(FE, qf, false);
    basisvals = zeros(Float64,ndofs4cell,ncomponents)
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
            # get FE basis at quadrature point
            FiniteElements.getFEbasis4qp!(basisvals, FEbasis, i)

            for dof_i = 1 : ndofs4cell, dof_j = dof_i : ndofs4cell
                # fill upper right part and diagonal of matrix
                @inbounds begin
                    temp = 0.0
                    for k = 1 : ncomponents
                        temp += basisvals[dof_i,k]*basisvals[dof_j,k];
                    end
                    temp *= factor * qf.w[i] * FE.grid.volume4cells[cell]
                    A[dofs[dof_i],dofs[dof_j]] += temp;
                    # fill lower left part of matrix
                    if dof_j > dof_i
                        A[dofs[dof_j],dofs[dof_i]] += temp;
                    end    
                end
            end
        end
    end
    #end    
end
