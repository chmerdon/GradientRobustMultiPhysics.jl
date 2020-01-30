function assemble_rhsL2_on_bface!(b, f!::Function, FE::AbstractHdivFiniteElement)
    ensure_bfaces!(FE.grid);
    ensure_length4faces!(FE.grid);
    nbfaces::Int = size(FE.grid.bfaces,1);
    ET = FE.grid.elemtypes[1]
    ETF = Grid.get_face_elemtype(ET)
    ndofs4face::Int = FiniteElements.get_ndofs4elemtype(FE, ETF);
    celldim::Int = size(FE.grid.nodes4cells,2) - 1;
    xdim::Int = size(FE.grid.coords4nodes,2);
    
    T = eltype(FE.grid.coords4nodes);
    qf = QuadratureFormula{T,typeof(ETF)}(2*(FiniteElements.get_polynomial_order(FE)));
     
    # pre-allocate memory for basis functions
    ncomponents = FiniteElements.get_ncomponents(FE);
    basisvals = Array{Array{T,1}}(undef,length(qf.w));
    for i in eachindex(qf.w)
        basisvals[i] = FiniteElements.get_basis_fluxes_on_elemtype(FE, ETF)(qf.xref[i])
    end    
    dofs = zeros(Int64,ndofs4face)
    coefficients = zeros(Float64,ndofs4face)
    
    loc2glob_trafo = Grid.local2global_face(FE.grid, ET)
    
    # quadrature loop
    det = 0.0
    fval = zeros(T,ncomponents)
    x = zeros(T,xdim);
    face = 0
    #@time begin    
        for j in eachindex(FE.grid.bfaces)
            face = FE.grid.bfaces[j];

            # get dofs
            FiniteElements.get_dofs_on_face!(dofs, FE, face, ETF);
            
            # get coefficients
            FiniteElements.get_basis_coefficients_on_face!(coefficients,FE,face,ETF);

            # get face trafo
            face_trafo = loc2glob_trafo(face)

            # get Piola trafo (todo: use elemtype-steered trafo)
            det = FE.grid.length4faces[face]; # determinant of transformation on face
            
            for i in eachindex(qf.w)
                # evaluate f
                x = face_trafo(qf.xref[i])
                f!(fval, x)

                # multiply with normal and save in fval[1]
                fval[1] = fval[1] * FE.grid.normal4faces[face,1] + fval[2] * FE.grid.normal4faces[face,2];
                fval[2] = 0.0;

                for dof_i = 1 : ndofs4face
                    # fill vector
                    # note: basisvals contain normalfluxes that have to be scaled with 1/det (Piola)
                    @inbounds begin
                      b[dofs[dof_i]] += (fval[1]*basisvals[i][dof_i,1]/det * qf.w[i] * FE.grid.length4faces[face] * coefficients[dof_i]);;
                    end
                end
            end
        end
    #end    
end