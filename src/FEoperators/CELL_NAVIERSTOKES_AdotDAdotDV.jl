struct CELL_NAVIERSTOKES_AdotDAdotDV <: FiniteElements.AbstractFEOperator end

function assemble_operator!(b,::Type{CELL_NAVIERSTOKES_AdotDAdotDV}, FEA::AbstractFiniteElement, FEU::AbstractH1FiniteElement, FEV::AbstractFiniteElement, dofs4aA, dofs4aU)
    # get quadrature formula
    T = eltype(FEU.grid.coords4nodes);
    ET = FEU.grid.elemtypes[1]
    quadorder = 3*FiniteElements.get_polynomial_order(FEU);
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
    
    # generate caller for FE basis functions
    FEbasisA = FiniteElements.FEbasis_caller(FEA, qf, false);
    FEbasisU = FiniteElements.FEbasis_caller(FEU, qf, true);
    FEbasisV = FiniteElements.FEbasis_caller(FEV, qf, false);
    gradientsU = zeros(Float64,FEbasisU.ndofs4item,FEbasisU.ncomponents*FEbasisU.xdim);
    basisvalsA = zeros(Float64,FEbasisA.ndofs4item,FEbasisA.xdim)
    basisvalsV = zeros(Float64,FEbasisV.ndofs4item,FEbasisV.xdim)

    # quadrature loop
    temp::T = 0.0
    det = 0.0;
    a4qp = zeros(Float64,FEbasisA.xdim)
    agrada = zeros(Float64,FEbasisA.xdim)
    #@time begin
    for cell = 1 : size(FEU.grid.nodes4cells,1)

        # update FEbasis on cell
        FiniteElements.updateFEbasis!(FEbasisU, cell)
        FiniteElements.updateFEbasis!(FEbasisV, cell)
        FiniteElements.updateFEbasis!(FEbasisA, cell)

        for i in eachindex(qf.w)

            # get FE basis gradients at quadrature point
            FiniteElements.getFEbasisgradients4qp!(gradientsU, FEbasisU, i)

            # get FE basis at quadrature point
            FiniteElements.getFEbasis4qp!(basisvalsA, FEbasisA, i)
            FiniteElements.getFEbasis4qp!(basisvalsV, FEbasisV, i)

            # compute a in quadrature point
            fill!(a4qp,0.0)
            for k = 1 : FEbasis.xdim
                for dof_i = 1 : FEbasisA.ndofs4item
                    a4qp[k] += dofs4aA[FEbasisA.current_dofs[dof_i]] * basisvalsA[dof_i,k];
                end
            end
        
            # compute agrada
            fill!(agrada,0.0)

            for dof_i = 1 : FEbasisU.ndofs4item
                for k = 1 : FEbasis.xdim
                    for c = 1 : ncomponents
                        agrada[k] += a4qp[c] * dofs4aU[FEbasisU.current_dofs[dof_i]] * gradientsU[dof_i,c+FEbasisU.offsets[k]]
                    end
                end    
            end
                
            for dof_j = 1 : FEbasisV.ndofs4item
                temp = 0.0;
                for k = 1 : FEbasis.xdim
                    temp += agrada[k] * basisvalsV[dof_j,k];
                end
                temp *= qf.w[i] * FEU.grid.volume4cells[cell]
                b[FEbasisV.current_dofs[dof_j]] -= temp;   
            end 
        end  
    end
    #end
end