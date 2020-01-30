# matrix for L2 bestapproximation that writes into an ExtendableSparseMatrix
function assemble_mass_matrix4FE!(A::ExtendableSparseMatrix,FE::AbstractH1FiniteElement)
    ncells::Int = size(FE.grid.nodes4cells,1);
    ET = FE.grid.elemtypes[1]
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    xdim::Int = size(FE.grid.coords4nodes,2);
    
    T = eltype(FE.grid.coords4nodes);
    qf = QuadratureFormula{T,typeof(FE.grid.elemtypes[1])}(2*(FiniteElements.get_polynomial_order(FE)));
     
    # pre-allocate memory for basis functions
    ncomponents = FiniteElements.get_ncomponents(FE);
    if ncomponents > 1
        basisvals = Array{Array{T,2}}(undef,length(qf.w));
    else
        basisvals = Array{Array{T,1}}(undef,length(qf.w));
    end    
    for i in eachindex(qf.w)
        # evaluate basis functions at quadrature point
        basisvals[i] = FiniteElements.get_basis_on_elemtype(FE, ET)(qf.xref[i])
    end    
                    
                
    dofs = zeros(Int64,ndofs4cell)
    coefficients = zeros(Float64,ndofs4cell,xdim)
    
    # quadrature loop
    temp = 0.0;
    #@time begin    
    for cell = 1 : ncells
        
        # get dofs
        FiniteElements.get_dofs_on_cell!(dofs, FE, cell, ET);

        # get coefficients
        FiniteElements.get_basis_coefficients_on_cell!(coefficients, FE, cell, ET);
            
        for i in eachindex(qf.w)
            for dof_i = 1 : ndofs4cell, dof_j = dof_i : ndofs4cell
                # fill upper right part and diagonal of matrix
                @inbounds begin
                    temp = 0.0
                    for k = 1 : ncomponents
                        temp += (basisvals[i][dof_i,k]*basisvals[i][dof_j,k] * qf.w[i] * FE.grid.volume4cells[cell]) * coefficients[dof_i,k] * coefficients[dof_j,k];
                    end
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
