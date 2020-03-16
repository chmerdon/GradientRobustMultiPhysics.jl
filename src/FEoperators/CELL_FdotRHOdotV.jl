struct CELL_FdotRHOdotV <: FiniteElements.AbstractFEOperator end

function assemble_operator!(A::ExtendableSparseMatrix, ::Type{CELL_FdotRHOdotV}, FEV::AbstractFiniteElement, FERHO::AbstractFiniteElement, f!::Function, quadrature_order::Int64)
    # get quadrature formula
    T = eltype(FEV.grid.coords4nodes);
    ET = FEV.grid.elemtypes[1]
    quadorder = quadrature_order + FiniteElements.get_polynomial_order(FEV) + FiniteElements.get_polynomial_order(FERHO);
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
     
    # generate caller for FE basis functions
    ncells::Int = size(FEV.grid.nodes4cells,1);
    ndofs4cellV::Int = FiniteElements.get_ndofs4elemtype(FEV, ET);
    ndofs4cellRHO::Int = FiniteElements.get_ndofs4elemtype(FERHO, ET);
    ncomponents::Int = FiniteElements.get_ncomponents(FEV);
    FEbasisV = FiniteElements.FEbasis_caller(FEV, qf, false);
    FEbasisRHO = FiniteElements.FEbasis_caller(FERHO, qf, false);
    basisvalsV = zeros(Float64,ndofs4cellV,ncomponents)
    basisvalsRHO = zeros(Float64,ndofs4cellRHO,1)
    dofsV = zeros(Int64,ndofs4cellV)
    dofsRHO = zeros(Int64,ndofs4cellRHO)
    
    # call trafo (needed to evaluate f)
    loc2glob_trafo = Grid.local2global(FEV.grid,ET)
    
    # quadrature loop
    temp = 0.0;
    fval = zeros(T,ncomponents)
    rho = 0.0;
    #@time begin    
    for cell = 1 : ncells

        # get dofs
        FiniteElements.get_dofs_on_cell!(dofsV, FEV, cell, ET);
        FiniteElements.get_dofs_on_cell!(dofsRHO, FERHO, cell, ET);
            
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
            x = cell_trafo(qf.xref[i]);
            f!(fval, x)
                
            for dof_i = 1 : ndofs4cellV, dof_j = 1 : ndofs4cellRHO
                # fill vector
                @inbounds begin
                    temp = 0.0
                    for k = 1 : ncomponents
                        temp += fval[k]*basisvalsV[dof_i,k];
                    end
                    A[dofsV[dof_i],dofsRHO[dof_j]] += temp * qf.w[i] * FEV.grid.volume4cells[cell] * basisvalsRHO[dof_j,1];
                end
            end
        end
    end
    #end    
end