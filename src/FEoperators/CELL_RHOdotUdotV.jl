struct CELL_RHOdotUdotV <: FiniteElements.AbstractFEOperator end

# elementwise matrix for L2 bestapproximation that writes into an ExtendableSparseMatrix
function assemble_operator!(A::ExtendableSparseMatrix,::Type{CELL_RHOdotUdotV},FEU::AbstractFiniteElement,FEV::AbstractFiniteElement, FERHO::AbstractFiniteElement, dofs4RHO)
    
    # get quadrature formula
    T = eltype(FEU.grid.coords4nodes);
    ET = FEU.grid.elemtypes[1]
    quadorder = FiniteElements.get_polynomial_order(FEU) + FiniteElements.get_polynomial_order(FEV) + FiniteElements.get_polynomial_order(FERHO);
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
     
    # generate caller for FE basis functions
    FEbasisU = FiniteElements.FEbasis_caller(FEU, qf, false);
    FEbasisV = FiniteElements.FEbasis_caller(FEV, qf, false);
    FEbasisRHO = FiniteElements.FEbasis_caller(FERHO, qf, false);
    basisvalsU = zeros(Float64,FEbasisU.ndofs4item,FEbasisU.ncomponents)
    basisvalsV = zeros(Float64,FEbasisV.ndofs4item,FEbasisV.ncomponents)
    basisvalsRHO = zeros(Float64,FEbasisRHO.ndofs4item,1)
    
    # quadrature loop
    temp = 0.0;
    rho = 0.0
    #@time begin    
    for cell = 1 : size(FEU.grid.nodes4cells,1);
        
        # update FEbasis on cell
        FiniteElements.updateFEbasis!(FEbasisU, cell)
        FiniteElements.updateFEbasis!(FEbasisV, cell)
        FiniteElements.updateFEbasis!(FEbasisRHO, cell)
            
        for i in eachindex(qf.w)
            # get FE basis at quadrature point
            FiniteElements.getFEbasis4qp!(basisvalsU, FEbasisU, i)
            FiniteElements.getFEbasis4qp!(basisvalsV, FEbasisV, i)

            # compute rho in quadrature point
            rho = 0.0
            for dof_i = 1 : FEbasisRHO.ndofs4item
                rho += dofs4RHO[FEbasisRHO.current_dofs[dof_i]] * basisvalsRHO[dof_i,1];
            end

            for dof_i = 1 : FEbasisU.ndofs4item, dof_j = 1 : FEbasisV.ndofs4item
                @inbounds begin
                    temp = 0.0
                    for k = 1 : FEbasisU.ncomponents
                        temp += basisvalsU[dof_i,k]*basisvalsV[dof_j,k];
                    end
                    temp *= rho * qf.w[i] * FEU.grid.volume4cells[cell]
                    A[FEbasisU.current_dofs[dof_i],FEbasisV.current_dofs[dof_j]] += temp;
                end
            end
        end
    end
    #end    
end
