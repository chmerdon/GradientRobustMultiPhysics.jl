struct CELL_FdotRHOdotV <: FiniteElements.AbstractFEOperator end

function assemble_operator!(A::ExtendableSparseMatrix, ::Type{CELL_FdotRHOdotV}, FEV::AbstractFiniteElement, FERHO::AbstractFiniteElement, f!::Function, quadrature_order::Int64)
    # get quadrature formula
    T = eltype(FEV.grid.coords4nodes);
    ET = FEV.grid.elemtypes[1]
    quadorder = quadrature_order + FiniteElements.get_polynomial_order(FEV) + FiniteElements.get_polynomial_order(FERHO);
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
     
    # generate caller for FE basis functions
    FEbasisV = FiniteElements.FEbasis_caller(FEV, qf, false);
    FEbasisRHO = FiniteElements.FEbasis_caller(FERHO, qf, false);
    basisvalsV = zeros(Float64,FEbasisV.ndofs4item,FEbasisV.ncomponents)
    basisvalsRHO = zeros(Float64,FEbasisRHO.ndofs4item,1)
    
    # call trafo (needed to evaluate f)
    loc2glob_trafo = Grid.local2global(FEV.grid,ET)
    
    # quadrature loop
    temp = 0.0;
    fval = zeros(T,FEbasisV.ncomponents)
    x = zeros(T,FEbasisV.xdim)
    rho = 0.0;
    #@time begin    
    for cell = 1 : size(FEV.grid.nodes4cells,1)
            
        # update FEbasis on cell
        FiniteElements.updateFEbasis!(FEbasisV, cell)
        FiniteElements.updateFEbasis!(FEbasisRHO, cell)

        # setup trafo on cell
        cell_trafo = loc2glob_trafo(cell)
            
        for i in eachindex(qf.w)
                
            # get FE basis at quadrature point
            FiniteElements.getFEbasis4qp!(basisvalsV, FEbasisV, i)
            FiniteElements.getFEbasis4qp!(basisvalsRHO, FEbasisRHO, i)

            # evaluate f
            x[:] = cell_trafo(qf.xref[i]);
            f!(fval, x)

            for dof_i = 1 : FEbasisV.ndofs4item, dof_j = 1 : FEbasisRHO.ndofs4item
                # fill vector
                @inbounds begin
                    temp = 0.0
                    for k = 1 : FEbasisV.ncomponents
                        temp += fval[k]*basisvalsV[dof_i,k];
                    end
                    A[FEbasisV.current_dofs[dof_i],FEbasisRHO.current_dofs[dof_j]] += temp * qf.w[i] * FEV.grid.volume4cells[cell] * basisvalsRHO[dof_j,1];
                end
            end
        end
    end
    #end    
end

function assemble_operator!(b, ::Type{CELL_FdotRHOdotV}, FEV::AbstractFiniteElement, FERHO::AbstractFiniteElement, f!::Function, quadrature_order::Int64, vals4RHO::Array)
    # get quadrature formula
    T = eltype(FEV.grid.coords4nodes);
    ET = FEV.grid.elemtypes[1]
    quadorder = quadrature_order + FiniteElements.get_polynomial_order(FEV) + FiniteElements.get_polynomial_order(FERHO);
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
     
    # generate caller for FE basis functions
    FEbasisV = FiniteElements.FEbasis_caller(FEV, qf, false);
    FEbasisRHO = FiniteElements.FEbasis_caller(FERHO, qf, false);
    basisvalsV = zeros(Float64,FEbasisV.ndofs4item,FEbasisV.ncomponents)
    basisvalsRHO = zeros(Float64,FEbasisRHO.ndofs4item,1)
    
    # call trafo (needed to evaluate f)
    loc2glob_trafo = Grid.local2global(FEV.grid,ET)
    
    # quadrature loop
    temp = 0.0;
    fval = zeros(T,FEbasisV.ncomponents)
    x = zeros(T,FEbasisV.xdim)
    rho = 0.0;
    #@time begin    
    for cell = 1 : size(FEV.grid.nodes4cells,1)
            
        # update FEbasis on cell
        FiniteElements.updateFEbasis!(FEbasisV, cell)
        FiniteElements.updateFEbasis!(FEbasisRHO, cell)

        # setup trafo on cell
        cell_trafo = loc2glob_trafo(cell)
            
        for i in eachindex(qf.w)
                
            # get FE basis at quadrature point
            FiniteElements.getFEbasis4qp!(basisvalsV, FEbasisV, i)
            FiniteElements.getFEbasis4qp!(basisvalsRHO, FEbasisRHO, i)

            # evaluate f
            x[:] = cell_trafo(qf.xref[i]);
            f!(fval, x)

            # evaluate rho
            rho = 0.0
            for dof_j = 1 : FEbasisRHO.ndofs4item
                rho += vals4RHO[FEbasisRHO.current_dofs[dof_j]]*basisvalsRHO[dof_j,1]
            end

            # fill vector
            for dof_i = 1 : FEbasisV.ndofs4item
                @inbounds begin
                    temp = 0.0
                    for k = 1 : FEbasisV.ncomponents
                        temp += fval[k]*basisvalsV[dof_i,k];
                    end
                    b[FEbasisV.current_dofs[dof_i]] += temp * qf.w[i] * FEV.grid.volume4cells[cell] * rho;
                end
            end
        end
    end
    #end    
end