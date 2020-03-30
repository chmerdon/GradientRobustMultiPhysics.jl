struct CELL_NAVIERSTOKES_RHOdotAdotDAdotDV <: FiniteElements.AbstractFEOperator end

function assemble_operator!(b,::Type{CELL_NAVIERSTOKES_RHOdotAdotDAdotDV}, FEU::AbstractH1FiniteElement, FEAV::AbstractFiniteElement, FERHO::AbstractFiniteElement, dofs4aAV, dofs4aU, dofs4RHO)
    # get quadrature formula
    T = eltype(FEU.grid.coords4nodes);
    ET = FEU.grid.elemtypes[1]
    quadorder = 3*FiniteElements.get_polynomial_order(FEU);
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
    
    # generate caller for FE basis functions
    FEbasisU = FiniteElements.FEbasis_caller(FEU, qf, true);
    FEbasisRHO = FiniteElements.FEbasis_caller(FERHO, qf, true);
    if FEU == FEAV
        FEbasisAV = FEbasisU    
    else
        FEbasisAV = FiniteElements.FEbasis_caller(FEAV, qf, true);
    end    
    gradientsU = zeros(Float64,FEbasisU.ndofs4item,FEbasisU.ncomponents*FEbasisU.xdim);
    basisvalsAV = zeros(Float64,FEbasisAV.ndofs4item,FEbasisAV.ncomponents)
    basisvalsRHO = zeros(Float64,FEbasisRHO.ndofs4item,FEbasisRHO.ncomponents)
    
    # quadrature loop
    temp::T = 0.0
    det = 0.0;
    rho = 0.0
    a4qp = zeros(Float64,FEbasisAV.xdim)
    agrada = zeros(Float64,FEbasisAV.xdim)
    #@time begin
    for cell = 1 : size(FEU.grid.nodes4cells,1)
      
        # update FEbasis on cell
        FiniteElements.updateFEbasis!(FEbasisU, cell)
        FiniteElements.updateFEbasis!(FEbasisRHO, cell)
        if FEU != FEAV
            FiniteElements.updateFEbasis!(FEbasisAV, cell)
        end    

        for i in eachindex(qf.w)

            # get FE basis gradients at quadrature point
            FiniteElements.getFEbasisgradients4qp!(gradientsU, FEbasisU, i)

            # get FE basis at quadrature point
            FiniteElements.getFEbasis4qp!(basisvalsAV, FEbasisAV, i)
            FiniteElements.getFEbasis4qp!(basisvalsRHO, FEbasisRHO, i)

            # compute a in quadrature point
            fill!(a4qp,0.0)
            for k = 1 : FEbasisAV.xdim
                for dof_i = 1 : FEbasisAV.ndofs4item
                    a4qp[k] += dofs4aAV[FEbasisAV.current_dofs[dof_i]] * basisvalsAV[dof_i,k];
                end
            end

            # compute rho in quadrature point
            rho = 0.0
            for dof_i = 1 : FEbasisRHO.ndofs4item
                rho += dofs4RHO[FEbasisRHO.current_dofs[dof_i]] * basisvalsRHO[dof_i,1];
            end
        
            # compute agrada
            fill!(agrada,0.0)
            for dof_i = 1 : FEbasisU.ndofs4item
                for k = 1 : FEbasisU.xdim
                    for c = 1 : FEbasisU.ncomponents
                        agrada[k] += a4qp[c] * dofs4aU[FEbasisU.current_dofs[dof_i]] *gradientsU[dof_i,c+FEbasisU.offsets[k]]
                    end
                end    
            end    
                
            for dof_j = 1 : FEbasisAV.ndofs4item
                temp = 0.0;
                for k = 1 : FEbasisAV.xdim
                    temp += agrada[k] * basisvalsAV[dof_j,k];
                end
                temp *= rho * qf.w[i] * FEU.grid.volume4cells[cell]
                b[FEbasisAV.current_dofs[dof_j]] -= temp;   
            end 
        end  
    end
    #end
end