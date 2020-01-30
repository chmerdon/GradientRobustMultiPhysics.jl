function assemble_rhsL2!(b, f!::Function, FE::AbstractHdivFiniteElement, quadrature_order::Int64)
    ncells::Int = size(FE.grid.nodes4cells,1);
    ET = FE.grid.elemtypes[1]
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    xdim::Int = size(FE.grid.coords4nodes,2);
    
    T = eltype(FE.grid.coords4nodes);
    qf = QuadratureFormula{T,typeof(ET)}(quadrature_order + FiniteElements.get_polynomial_order(FE));
     
    # pre-allocate memory for basis functions
    ncomponents = FiniteElements.get_ncomponents(FE);
    if ncomponents == 1
        basisvals = Array{Array{T,1}}(undef,length(qf.w));
    else
        basisvals = Array{Array{T,2}}(undef,length(qf.w));
    end
    for i in eachindex(qf.w)
        basisvals[i] = FiniteElements.get_basis_on_elemtype(FE, ET)(qf.xref[i])
    end    
    transformed_basisvals = zeros(ndofs4cell,ncomponents);
    dofs = zeros(Int64,ndofs4cell)
    coefficients = zeros(Float64,ndofs4cell,xdim)
    
    loc2glob_trafo = Grid.local2global(FE.grid,ET)
    
    AT = zeros(Float64,2,2)
    get_Piola_trafo_on_cell! = Grid.local2global_Piola(FE.grid, FE.grid.elemtypes[1])
    
    # quadrature loop
    temp = 0.0;
    det = 0.0;
    fval = zeros(T,ncomponents)
   # @time begin    
        for cell = 1 : ncells
            # get dofs
            FiniteElements.get_dofs_on_cell!(dofs, FE, cell, ET);
            
            # get coefficients
            FiniteElements.get_basis_coefficients_on_cell!(coefficients,FE,cell,ET);
            
            # get trafo and Piola trafo
            cell_trafo = loc2glob_trafo(cell)
            det = get_Piola_trafo_on_cell!(AT,cell);
            
            for i in eachindex(qf.w)
                
                # use Piola transformation on basisvals
                for dof_i = 1 : ndofs4cell
                    for k = 1 : ncomponents
                        transformed_basisvals[dof_i,k] = 0.0;
                        for l = 1 : ncomponents
                            transformed_basisvals[dof_i,k] += AT[k,l]*basisvals[i][dof_i,l];
                        end    
                        transformed_basisvals[dof_i,k] /= det;
                    end    
                end    

                # evaluate f
                x = cell_trafo(qf.xref[i]);
                f!(fval, x)
                
                for dof_i = 1 : ndofs4cell
                    # fill vector
                    @inbounds begin
                      temp = 0.0
                      for k = 1 : ncomponents
                        temp += (fval[k]*transformed_basisvals[dof_i,k]*coefficients[dof_i,k] * qf.w[i] * FE.grid.volume4cells[cell]);
                      end
                      b[dofs[dof_i]] += temp;
                    end
                end
            end
        end
   # end    
end