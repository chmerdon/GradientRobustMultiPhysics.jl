struct CELL_NAVIERSTOKES_AdotDUdotDV <: FiniteElements.AbstractFEOperator end

function assemble_operator!(A::ExtendableSparseMatrix,::Type{CELL_NAVIERSTOKES_AdotDUdotDV}, FEU::AbstractH1FiniteElement, FEAV::AbstractFiniteElement, dofs4a)
    # get quadrature formula
    T = eltype(FEU.grid.coords4nodes);
    ET = FEU.grid.elemtypes[1]
    quadorder = 2*(FiniteElements.get_polynomial_order(FEU)-1);
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
    
    # generate caller for FE basis functions
    ndofs4cellU::Int = FiniteElements.get_ndofs4elemtype(FEU, ET);
    ndofs4cellAV::Int = FiniteElements.get_ndofs4elemtype(FEAV, ET);
    ncomponents::Int = FiniteElements.get_ncomponents(FEU);
    xdim = size(FEU.grid.coords4nodes,2)
    FEbasisU = FiniteElements.FEbasis_caller(FEU, qf, true);
    if FEU == FEAV
        FEbasisAV = FEbasisU    
    else
        FEbasisAV = FiniteElements.FEbasis_caller(FEAV, qf, true);
    end    
    gradientsU = zeros(Float64,ndofs4cellU,ncomponents*xdim);
    basisvalsAV = zeros(Float64,ndofs4cellAV,ncomponents)
    dofsU = zeros(Int64,ndofs4cellU)
    dofsAV = zeros(Int64,ndofs4cellAV)
    
    # quadrature loop
    temp::T = 0.0
    det = 0.0;
    a4qp = zeros(Float64,xdim)
    agradu = zeros(Float64,xdim)
    #@time begin
    for cell = 1 : size(FEU.grid.nodes4cells,1)
      
        # get dofs
        FiniteElements.get_dofs_on_cell!(dofsU, FEU, cell, ET);
        FiniteElements.get_dofs_on_cell!(dofsAV, FEAV, cell, ET);

        # update FEbasis on cell
        FiniteElements.updateFEbasis!(FEbasisU, cell)
        if FEU != FEAV
            FiniteElements.updateFEbasis!(FEbasisAV, cell)
        end    

        for i in eachindex(qf.w)

            # get FE basis gradients at quadrature point
            FiniteElements.getFEbasisgradients4qp!(gradientsU, FEbasisU, i)

            # get FE basis at quadrature point
            FiniteElements.getFEbasis4qp!(basisvalsAV, FEbasisAV, i)

            # compute a in quadrature point
            fill!(a4qp,0.0)
            for k = 1 : xdim
                for dof_i = 1 : ndofs4cellAV
                    a4qp[k] += dofs4a[dofsAV[dof_i]] * basisvalsAV[dof_i,k];
                end
            end
        
            # fill sparse array
            for dof_i = 1 : ndofs4cellU
                # compute agradu
                fill!(agradu,0.0)
                for k = 1 : xdim
                    for c = 1 : ncomponents
                        agradu[k] += a4qp[c]*gradientsU[dof_i,k+FEbasisU.offsets[c]]
                    end
                end    
                
                for dof_j = 1 : ndofs4cellAV
                    temp = 0.0;
                    for k = 1 : xdim
                        temp += agradu[k] * basisvalsAV[dof_j,k];
                    end
                    temp *= qf.w[i] * FEU.grid.volume4cells[cell]
                    A[dofsAV[dof_j],dofsU[dof_i]] += temp;   
                end
            end 
        end  
    end
    #end
end