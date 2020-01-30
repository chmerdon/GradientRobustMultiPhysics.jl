function assemble_rhsL2_on_bface!(b, f!::Function, FE::AbstractH1FiniteElement)
    ensure_bfaces!(FE.grid);
    ensure_length4faces!(FE.grid);
    ET = FE.grid.elemtypes[1]
    ETF = Grid.get_face_elemtype(ET);
    nbfaces::Int = size(FE.grid.bfaces,1);
    ndofs4face::Int = FiniteElements.get_ndofs4elemtype(FE, ETF);
    xdim::Int = size(FE.grid.coords4nodes,2);
    
    T = eltype(FE.grid.coords4nodes);
    qf = QuadratureFormula{T,Grid.ElemType1DInterval}(2*(FiniteElements.get_polynomial_order(FE)));
     
    # pre-allocate memory for basis functions
    ncomponents = FiniteElements.get_ncomponents(FE);
    if ncomponents == 1
        basisvals = Array{Array{T,1}}(undef,length(qf.w));
    else
        basisvals = Array{Array{T,2}}(undef,length(qf.w));
    end
    for i in eachindex(qf.w)
        basisvals[i] = FiniteElements.get_basis_on_elemtype(FE, ETF)(qf.xref[i])
    end    
    dofs = zeros(Int64,ndofs4face)
    coefficients = zeros(Float64,ndofs4face,xdim)
    
    loc2glob_trafo = Grid.local2global_face(FE.grid, ET)

    temp = 0.0
    fval = zeros(T,ncomponents)
    x = zeros(T,xdim);
    face = 0
    @time begin    
        for j in eachindex(FE.grid.bfaces)

            face = FE.grid.bfaces[j];
            
            # get dofs
            FiniteElements.get_dofs_on_face!(dofs,FE,face,ETF);

            # get coefficients
            FiniteElements.get_basis_coefficients_on_face!(coefficients,FE,face,ETF);

            # get face trafo
            face_trafo = loc2glob_trafo(face)
            
            for i in eachindex(qf.w)
                # evaluate f
                x = face_trafo(qf.xref[i])
                f!(fval, x)
                
                for dof_i = 1 : ndofs4face
                    # fill vector
                    @inbounds begin
                      temp = 0.0
                      for k = 1 : ncomponents
                        temp += (fval[k]*basisvals[i][dof_i,k] * qf.w[i] * FE.grid.length4faces[face] * coefficients[dof_i,k]);
                      end
                      b[dofs[dof_i]] += temp;
                    end
                end
            end
        end
    end    
end