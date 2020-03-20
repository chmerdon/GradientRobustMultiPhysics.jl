struct CELL_RHOdotUdotV <: FiniteElements.AbstractFEOperator end

# elementwise matrix for L2 bestapproximation that writes into an ExtendableSparseMatrix
function assemble_operator!(A::ExtendableSparseMatrix,::Type{CELL_RHOdotUdotV},FEU::AbstractFiniteElement,FEV::AbstractFiniteElement, FERHO::AbstractFiniteElement, dofs4RHO)
    
    # get quadrature formula
    T = eltype(FEU.grid.coords4nodes);
    ET = FEU.grid.elemtypes[1]
    quadorder = FiniteElements.get_polynomial_order(FEU) + FiniteElements.get_polynomial_order(FEV) + FiniteElements.get_polynomial_order(FERHO);
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
     
    # generate caller for FE basis functions
    ncells::Int = size(FEU.grid.nodes4cells,1);
    ndofs4cellU::Int = FiniteElements.get_ndofs4elemtype(FEU, ET);
    ndofs4cellV::Int = FiniteElements.get_ndofs4elemtype(FEV, ET);
    ndofs4cellRHO::Int = FiniteElements.get_ndofs4elemtype(FERHO, ET);
    ncomponents::Int = FiniteElements.get_ncomponents(FEU);
    xdim = size(FEU.grid.coords4nodes,2)
    FEbasisU = FiniteElements.FEbasis_caller(FEU, qf, false);
    FEbasisV = FiniteElements.FEbasis_caller(FEV, qf, false);
    FEbasisRHO = FiniteElements.FEbasis_caller(FERHO, qf, false);
    basisvalsU = zeros(Float64,ndofs4cellU,ncomponents)
    basisvalsV = zeros(Float64,ndofs4cellV,ncomponents)
    basisvalsRHO = zeros(Float64,ndofs4cellRHO,1)
    dofsU = zeros(Int64,ndofs4cellU)
    dofsV = zeros(Int64,ndofs4cellV)
    dofsRHO = zeros(Int64,ndofs4cellRHO)
    
    # quadrature loop
    temp = 0.0;
    rho = 0.0
    #@time begin    
    for cell = 1 : ncells
        
        # get dofs
        FiniteElements.get_dofs_on_cell!(dofsU, FEU, cell, ET);
        FiniteElements.get_dofs_on_cell!(dofsV, FEV, cell, ET);
        FiniteElements.get_dofs_on_cell!(dofsRHO, FERHO, cell, ET);

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
            for dof_i = 1 : ndofs4cellRHO
                rho += dofs4RHO[dofsRHO[dof_i]] * basisvalsRHO[dof_i,1];
            end

            for dof_i = 1 : ndofs4cellU, dof_j = 1 : ndofs4cellV
                @inbounds begin
                    temp = 0.0
                    for k = 1 : ncomponents
                        temp += basisvalsU[dof_i,k]*basisvalsV[dof_j,k];
                    end
                    temp *= rho * qf.w[i] * FEU.grid.volume4cells[cell]
                    A[dofsU[dof_i],dofsV[dof_j]] += temp;
                end
            end
        end
    end
    #end    
end
