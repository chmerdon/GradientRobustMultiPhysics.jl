function assemble_bface_mass_matrix4FE!(A::ExtendableSparseMatrix,FE::AbstractHdivFiniteElement)
    ensure_bfaces!(FE.grid);
    ensure_length4faces!(FE.grid);
    nbfaces::Int = size(FE.grid.bfaces,1);
    ET = FE.grid.elemtypes[1]
    ETF = Grid.get_face_elemtype(ET)
    ndofs4face::Int = FiniteElements.get_ndofs4elemtype(FE, ETF);
    xdim::Int = size(FE.grid.coords4nodes,2);
    
    T = eltype(FE.grid.coords4nodes);
    qf = QuadratureFormula{T,typeof(ETF)}(2*(FiniteElements.get_polynomial_order(FE)));
     
    # pre-allocate memory for basis functions
    ncomponents = FiniteElements.get_ncomponents(FE);
    basisvals = Array{Array{T,1}}(undef,length(qf.w));
        
    for i in eachindex(qf.w)
        # evaluate basis functions at quadrature point
        basisvals[i] = FiniteElements.get_basis_fluxes_on_elemtype(FE, ETF)(qf.xref[i])
    end    
    dofs = zeros(Int64,ndofs4face)
    coefficients = zeros(Float64,ndofs4face)
    
    # quadrature loop
    det = 0.0;
    temp = 0.0;
    face = 0;
    #@time begin    
        for j in eachindex(FE.grid.bfaces)
            face = FE.grid.bfaces[j];

            # get dofs
            FiniteElements.get_dofs_on_face!(dofs, FE, face, ETF);

            # get coefficients
            FiniteElements.get_basis_coefficients_on_face!(coefficients,FE,face,ETF);

            # get Piola trafo (todo: use elemtype-steered trafo)
            det = FE.grid.length4faces[face]; # determinant of transformation on face
            
            for i in eachindex(qf.w)
                
                for dof_i = 1 : ndofs4face, dof_j = dof_i : ndofs4face
                    # fill upper right part and diagonal of matrix
                    @inbounds begin
                      temp = (basisvals[i][dof_i]*basisvals[i][dof_j] * qf.w[i] * FE.grid.length4faces[face]) * coefficients[dof_i] * coefficients[dof_j] / det / det;
                      A[dofs[dof_i],dofs[dof_j]] += temp;
                      # fill lower left part of matrix
                      if dof_j > dof_i
                        A[dofs[dof_j],dofs[dof_i]] += temp;
                      end 
                    end
                end
            end
        end#
    #end    
end