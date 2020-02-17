struct CELL_FdotV <: FiniteElements.AbstractFEOperator end

function assemble_operator!(b, ::Type{CELL_FdotV}, FE::AbstractFiniteElement, f!::Function, quadrature_order::Int64)
    # get quadrature formula
    T = eltype(FE.grid.coords4nodes);
    ET = FE.grid.elemtypes[1]
    quadorder = quadrature_order + FiniteElements.get_polynomial_order(FE);
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
     
    # generate caller for FE basis functions
    ncells::Int = size(FE.grid.nodes4cells,1);
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    ncomponents::Int = FiniteElements.get_ncomponents(FE);
    FEbasis = FiniteElements.FEbasis_caller(FE, qf, false);
    basisvals = zeros(Float64,ndofs4cell,ncomponents)
    dofs = zeros(Int64,ndofs4cell)
    
    # call trafo (needed to evaluate f)
    loc2glob_trafo = Grid.local2global(FE.grid,FE.grid.elemtypes[1])
    
    # quadrature loop
    temp = 0.0;
    fval = zeros(T,ncomponents)
    #@time begin    
    for cell = 1 : ncells

        # get dofs
        FiniteElements.get_dofs_on_cell!(dofs, FE, cell, ET);
            
        # update FEbasis on cell
        FiniteElements.updateFEbasis!(FEbasis, cell)

        # setup trafo on cell
        cell_trafo = loc2glob_trafo(cell)
            
        for i in eachindex(qf.w)
                
            # get FE basis at quadrature point
            FiniteElements.getFEbasis4qp!(basisvals, FEbasis, i)

            # evaluate f
            x = cell_trafo(qf.xref[i]);
            f!(fval, x)
                
            for dof_i = 1 : ndofs4cell
                # fill vector
                @inbounds begin
                    temp = 0.0
                    for k = 1 : ncomponents
                        temp += fval[k]*basisvals[dof_i,k];
                    end
                    b[dofs[dof_i]] += temp * qf.w[i] * FE.grid.volume4cells[cell];
                end
            end
        end
    end
    #end    
end