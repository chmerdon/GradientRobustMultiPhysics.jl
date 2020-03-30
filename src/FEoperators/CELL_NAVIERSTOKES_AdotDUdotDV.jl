struct CELL_NAVIERSTOKES_AdotDUdotDV <: FiniteElements.AbstractFEOperator end

function assemble_operator!(A::ExtendableSparseMatrix,::Type{CELL_NAVIERSTOKES_AdotDUdotDV}, FEA::AbstractFiniteElement, FEU::AbstractH1FiniteElement, FEV::AbstractFiniteElement, dofs4a)
    # get quadrature formula
    T = eltype(FEU.grid.coords4nodes);
    ET = FEU.grid.elemtypes[1]
    quadorder = FiniteElements.get_polynomial_order(FEA) + FiniteElements.get_polynomial_order(FEU)-1 + FiniteElements.get_polynomial_order(FEV);
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
    agradu = zeros(Float64,FEbasisA.xdim)
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
            for k = 1 : FEbasisA.xdim
                for dof_i = 1 : FEbasisA.ndofs4item
                    a4qp[k] += dofs4a[FEbasisA.current_dofs[dof_i]] * basisvalsA[dof_i,k];
                end
            end
        
            # fill sparse array
            for dof_i = 1 : FEbasisU.ndofs4item
                # compute agradu
                fill!(agradu,0.0)
                for k = 1 : FEbasisA.xdim
                    for c = 1 : FEbasisA.ncomponents
                        agradu[k] += a4qp[c]*gradientsU[dof_i,c+FEbasisU.offsets[k]]
                    end
                end    
                
                for dof_j = 1 : FEbasisV.ndofs4item
                    temp = 0.0;
                    for k = 1 : FEbasisA.xdim
                        temp += agradu[k] * basisvalsV[dof_j,k];
                    end
                    temp *= qf.w[i] * FEU.grid.volume4cells[cell]
                    A[FEbasisV.current_dofs[dof_j],FEbasisU.current_dofs[dof_i]] += temp;   
                end
            end 
        end  
    end
    #end
end