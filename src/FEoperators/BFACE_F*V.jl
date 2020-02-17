struct BFACE_FdotV <: FiniteElements.AbstractFEOperator end


function assemble_operator!(b, ::Type{BFACE_FdotV}, FE::AbstractH1FiniteElement, f!::Function, quadorder_f::Int)
    ensure_bfaces!(FE.grid);
    ensure_length4faces!(FE.grid);
  
    # get quadrature formula
    T = eltype(FE.grid.coords4nodes);
    ET = FE.grid.elemtypes[1]
    ETF = Grid.get_face_elemtype(ET);
    quadorder = quadorder_f + FiniteElements.get_polynomial_order(FE);
    qf = QuadratureFormula{T,typeof(ETF)}(quadorder);
     
    # generate caller for FE basis functions
    ndofs4face::Int = FiniteElements.get_ndofs4elemtype(FE, ETF);
    ncomponents::Int = FiniteElements.get_ncomponents(FE);
    FEbasis = FiniteElements.FEbasis_caller_face(FE, qf, false);
    basisvals = zeros(Float64,ndofs4face,ncomponents)
    dofs = zeros(Int64,ndofs4face)
    
    loc2glob_trafo = Grid.local2global_face(FE.grid, ET)

    # quadrature loop
    temp = 0.0;
    fval = zeros(T,ncomponents)
    face = 0;
    #@time begin    
    for bface = 1 : size(FE.grid.bfaces,1)

        face = FE.grid.bfaces[bface];
         
        # get dofs
        FiniteElements.get_dofs_on_face!(dofs,FE,face,ETF);
 
        # update FEbasis on face
        FiniteElements.updateFEbasis!(FEbasis, face)

        # get face trafo
        face_trafo = loc2glob_trafo(face)
             
        for i in eachindex(qf.w)
            # get FE basis at quadrature point
            FiniteElements.getFEbasis4qp!(basisvals, FEbasis, i)

            # evaluate f
            x = face_trafo(qf.xref[i])
            f!(fval, x)
             
            for dof_i = 1 : ndofs4face
                # fill vector
                @inbounds begin
                    temp = 0.0
                    for k = 1 : ncomponents
                        temp += fval[k]*basisvals[dof_i,k];
                    end
                    b[dofs[dof_i]] += temp * qf.w[i] * FE.grid.length4faces[face];
                end
            end
        end    
    end
    #end    
end

function assemble_operator!(b, ::Type{BFACE_FdotV}, FE::AbstractHdivFiniteElement, f!::Function, quadorder_f::Int)
    ensure_bfaces!(FE.grid);
    ensure_length4faces!(FE.grid);
  
    # get quadrature formula
    T = eltype(FE.grid.coords4nodes);
    ET = FE.grid.elemtypes[1]
    ETF = Grid.get_face_elemtype(ET);
    quadorder = quadorder_f + FiniteElements.get_polynomial_order(FE);
    qf = QuadratureFormula{T,typeof(ETF)}(quadorder);
     
    # generate caller for FE basis functions
    ndofs4face::Int = FiniteElements.get_ndofs4elemtype(FE, ETF);
    ncomponents::Int = FiniteElements.get_ncomponents(FE);
    FEbasis = FiniteElements.FEbasis_caller_face(FE, qf, false);
    basisvals = zeros(Float64,ndofs4face,ncomponents)
    dofs = zeros(Int64,ndofs4face)
    
    loc2glob_trafo = Grid.local2global_face(FE.grid, ET)

    # quadrature loop
    temp = 0.0;
    fval = zeros(T,ncomponents)
    face = 0;
    #@time begin    
    for bface = 1 : size(FE.grid.bfaces,1)

        face = FE.grid.bfaces[bface];
         
        # get dofs
        FiniteElements.get_dofs_on_face!(dofs,FE,face,ETF);
 
        # update FEbasis on face
        FiniteElements.updateFEbasis!(FEbasis, face)

        # get face trafo
        face_trafo = loc2glob_trafo(face)
             
        for i in eachindex(qf.w)
            # get FE basis at quadrature point
            FiniteElements.getFEbasis4qp!(basisvals, FEbasis, i)

            # evaluate f
            x = face_trafo(qf.xref[i])
            f!(fval, x)

            # multiply with normal and save in fval[1]
            fval[1] = fval[1] * FE.grid.normal4faces[face,1] + fval[2] * FE.grid.normal4faces[face,2];
            fval[2] = 0.0;
             
            for dof_i = 1 : ndofs4face
                # fill vector
                @inbounds begin
                    temp = 0.0
                    for k = 1 : ncomponents
                        temp += fval[k]*basisvals[dof_i,k];
                    end
                    b[dofs[dof_i]] += temp * qf.w[i] * FE.grid.length4faces[face];
                end
            end
        end    
    end
    #end    
end