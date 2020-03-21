struct CELL_NAVIERSTOKES_AdotDAdotDV <: FiniteElements.AbstractFEOperator end

function assemble_operator!(b,::Type{CELL_NAVIERSTOKES_AdotDAdotDV}, FEA::AbstractFiniteElement, FEU::AbstractH1FiniteElement, FEV::AbstractFiniteElement, dofs4aA, dofs4aU)
    # get quadrature formula
    T = eltype(FEU.grid.coords4nodes);
    ET = FEU.grid.elemtypes[1]
    quadorder = 3*FiniteElements.get_polynomial_order(FEU);
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
    
    # generate caller for FE basis functions
    ndofs4cellU::Int = FiniteElements.get_ndofs4elemtype(FEU, ET);
    ndofs4cellA::Int = FiniteElements.get_ndofs4elemtype(FEA, ET);
    ndofs4cellV::Int = FiniteElements.get_ndofs4elemtype(FEV, ET);
    ncomponents::Int = FiniteElements.get_ncomponents(FEU);
    xdim = size(FEU.grid.coords4nodes,2)
    FEbasisA = FiniteElements.FEbasis_caller(FEA, qf, false);
    FEbasisU = FiniteElements.FEbasis_caller(FEU, qf, true);
    FEbasisV = FiniteElements.FEbasis_caller(FEV, qf, false);
    gradientsU = zeros(Float64,ndofs4cellU,ncomponents*xdim);
    basisvalsA = zeros(Float64,ndofs4cellA,xdim)
    basisvalsV = zeros(Float64,ndofs4cellV,xdim)
    dofsU = zeros(Int64,ndofs4cellU)
    dofsA = zeros(Int64,ndofs4cellA)
    dofsV = zeros(Int64,ndofs4cellV)
    
    # quadrature loop
    temp::T = 0.0
    det = 0.0;
    a4qp = zeros(Float64,xdim)
    agrada = zeros(Float64,xdim)
    #@time begin
    for cell = 1 : size(FEU.grid.nodes4cells,1)
      
        # get dofs
        FiniteElements.get_dofs_on_cell!(dofsU, FEU, cell, ET);
        FiniteElements.get_dofs_on_cell!(dofsA, FEA, cell, ET);
        FiniteElements.get_dofs_on_cell!(dofsV, FEV, cell, ET);

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
            for k = 1 : xdim
                for dof_i = 1 : ndofs4cellA
                    a4qp[k] += dofs4aA[dofsA[dof_i]] * basisvalsA[dof_i,k];
                end
            end
        
            # compute agrada
            fill!(agrada,0.0)
            for dof_i = 1 : ndofs4cellU
                for k = 1 : xdim
                    for c = 1 : ncomponents
                        agrada[k] += a4qp[c] * dofs4aU[dofsU[dof_i]] *gradientsU[dof_i,k+FEbasisU.offsets[c]]
                    end
                end    
            end    
                
            for dof_j = 1 : ndofs4cellV
                temp = 0.0;
                for k = 1 : xdim
                    temp += agrada[k] * basisvalsV[dof_j,k];
                end
                temp *= qf.w[i] * FEU.grid.volume4cells[cell]
                b[dofsV[dof_j]] -= temp;   
            end 
        end  
    end
    #end
end