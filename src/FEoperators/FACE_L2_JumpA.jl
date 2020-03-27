struct FACE_L2_JumpA <: FiniteElements.AbstractFEOperator end

function assemble_operator!(b,::Type{FACE_L2_JumpA},FE::AbstractFiniteElement, dofs4A)
    ensure_length4faces!(FE.grid);
    ensure_faces4cells!(FE.grid);
    ensure_cells4faces!(FE.grid);
  
    # get quadrature formula for face
    T = eltype(FE.grid.coords4nodes);
    ET = FE.grid.elemtypes[1]
    ETF = Grid.get_face_elemtype(ET);
    quadorder = 2*FiniteElements.get_polynomial_order(FE);
    qf = QuadratureFormula{T,typeof(ETF)}(quadorder);

    # evil hack? : we take a quadrature formula on the face and add
    # one zero coordinates to xref to have quadrature points on the cell
    # depending on the position of this additional zero coordinate
    # we are on a different face of the cell
    for i in eachindex(qf.w)
        qf.xref[i] = [0.0; qf.xref[i][1:end]]
    end

    if typeof(ET) <: Grid.Abstract2DElemType
        zeropos4faces = [1 2 3 1];
        invert = [false,true,false];
    else
        zeropos4faces = [1 1 2];
        invert = [false,false]
    end    

    # generate caller for FE basis functions on cell faces
    nfaces4cell = size(FE.grid.faces4cells,2);
    FEbasis = Array{FiniteElements.FEbasis_caller}(undef,nfaces4cell)
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    ncomponents::Int = FiniteElements.get_ncomponents(FE);
    xdim = size(FE.grid.coords4nodes,2)
    basisvals = zeros(Float64,ndofs4cell,ncomponents);
    dofs = zeros(Int64,ndofs4cell)

    for j = 1 : nfaces4cell
        # move zero in qf.xref one position backward
        for i in eachindex(qf.w)
            qf.xref[i][[zeropos4faces[j+1],zeropos4faces[j]]] = qf.xref[i][[zeropos4faces[j],zeropos4faces[j+1]]]
            if invert[j] == true
                qf.xref[:] = qf.xref[end:-1:1]
            end
        end

        # generate FEbasis
        FEbasis[j] = FiniteElements.FEbasis_caller(FE, qf, false);

        if invert[j] == true
            qf.xref[:] = qf.xref[end:-1:1]
        end
    end
          
    # quadrature loop
    jump4face4qp = zeros(Float64,length(qf.xref),ncomponents);
    cell = 0
    nac = 0
    pos = 
    #@time begin
    for face = 1 : size(FE.grid.nodes4faces,1)

        nac = (FE.grid.cells4faces[face,2] == 0) ? 1 : 2

        if nac == 2
            fill!(jump4face4qp,0.0)

            for ac = 1 : nac

                cell = FE.grid.cells4faces[face,ac]
                # find position of face in adjacentcell
                # (to choose correct quadrature formula/FEbasis_caller)
                pos = findall(x->x == face, FE.grid.faces4cells[cell,:])
                
                # get dofs on adjacent cell
                FiniteElements.get_dofs_on_cell!(dofs, FE, cell, ET);

                FiniteElements.updateFEbasis!(FEbasis[pos[1]], cell)
        
                for i in eachindex(qf.w) # loop over all quadrature points on face (masked as cell points)
                
                    # get FE basis at quadrature point
                    FiniteElements.getFEbasis4qp!(basisvals, FEbasis[pos[1]], i)

                    # compute jump
                    for dof_i = 1 : ndofs4cell
                        # fill upper right part and diagonal of matrix
                        for k = 1 : ncomponents
                            if (ac == 1)
                                jump4face4qp[i,k] += basisvals[dof_i,k] * dofs4A[dofs[dof_i]];
                            else
                                jump4face4qp[length(qf.xref)+1-i,k] -= basisvals[dof_i,k] * dofs4A[dofs[dof_i]];
                            end    
                        end
                    end
                end  
            end
    
            for i in eachindex(qf.w)
                for k = 1 : ncomponents
                    b[face,k] += jump4face4qp[i,k]^2 * qf.w[i] * FE.grid.length4faces[face];
                end    
            end
        end
    end

    #end
end