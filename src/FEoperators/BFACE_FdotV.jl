struct BFACE_FdotV <: FiniteElements.AbstractFEOperator end


function assemble_operator!(b, ::Type{BFACE_FdotV}, FE::AbstractH1FiniteElement, Dbid::Int64, f!::Function, quadorder_f::Int)
    ensure_bfaces!(FE.grid);
    ensure_length4faces!(FE.grid);
  
    # get quadrature formula
    T = eltype(FE.grid.coords4nodes);
    ET = FE.grid.elemtypes[1]
    ETF = Grid.get_face_elemtype(ET);
    quadorder = quadorder_f + FiniteElements.get_polynomial_order(FE);
    qf = QuadratureFormula{T,typeof(ETF)}(quadorder);
     
    # generate caller for FE basis functions
    FEbasis = FiniteElements.FEbasis_caller_face(FE, qf);
    basisvals = zeros(Float64,FEbasis.ndofs4item,FEbasis.ncomponents)
    
    # trafo for evaluation of f
    loc2glob_trafo = Grid.local2global_face(FE.grid, ET)

    # quadrature loop
    temp = 0.0;
    fval = zeros(T,FEbasis.ncomponents)
    x = zeros(T,FEbasis.xdim)
    face = 0;
    #@time begin    
    for bface = 1 : size(FE.grid.bfaces,1)
        if FE.grid.bregions[bface] == Dbid

            face = FE.grid.bfaces[bface];
 
            # update FEbasis on face
            FiniteElements.updateFEbasis!(FEbasis, face)

            # get face trafo
            face_trafo = loc2glob_trafo(face)
             
            for i in eachindex(qf.w)
                # get FE basis at quadrature point
                FiniteElements.getFEbasis4qp!(basisvals, FEbasis, i)

                # evaluate f
                x[:] = face_trafo(qf.xref[i])
                f!(fval, x)
             
                for dof_i = 1 : FEbasis.ndofs4item
                    # fill vector
                    @inbounds begin
                        temp = 0.0
                        for k = 1 : FEbasis.ncomponents
                            temp += fval[k]*basisvals[dof_i,k];
                        end
                        b[FEbasis.current_dofs[dof_i]] += temp * qf.w[i] * FE.grid.length4faces[face];
                    end
                end
            end    
        end
    end
end