struct CELL_FdotDIVV <: FiniteElements.AbstractFEOperator end

function assemble_operator!(b, ::Type{CELL_FdotDIVV}, FE::AbstractFiniteElement, f!::Function, quadrature_order::Int)
    
    # get quadrature formula
    T = eltype(FE.grid.coords4nodes);
    ET = FE.grid.elemtypes[1]
    quadorder = quadrature_order + FiniteElements.get_polynomial_order(FE) - 1;
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
    
    # generate caller for FE basis functions
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    ncomponents::Int = FiniteElements.get_ncomponents(FE);
    xdim::Int = size(FE.grid.coords4nodes,2)

    @assert ncomponents == xdim

    FEbasis = FiniteElements.FEbasis_caller(FE, qf, true);
    divergences = zeros(Float64,ndofs4cell,1);
    dofs = zeros(Int64,ndofs4cell)

    loc2glob_trafo = Grid.local2global(FE.grid,ET)
    
    # quadrature loop
    fval = zeros(T,xdim)
    x = zeros(T,xdim)
    #@time begin    
    for cell = 1 : size(FE.grid.nodes4cells,1)
      
        # update FEbasis on cell
        FiniteElements.updateFEbasis!(FEbasis, cell)
      
        # get dofs
        FiniteElements.get_dofs_on_cell!(dofs, FE, cell, ET);      

        # get trafo
        cell_trafo = loc2glob_trafo(cell)
      
        for i in eachindex(qf.w)
        
            # get FE basis gradients at quadrature point
            FiniteElements.getFEbasisdivergence4qp!(divergences, FEbasis, i)
                
            # evaluate f
            x[:] = cell_trafo(qf.xref[i]);
            f!(fval, x)
                
            for dof_i = 1 : ndofs4cell
                # fill vector
                @inbounds begin
                    b[dofs[dof_i]] += divergences[dof_i] * fval[1] * qf.w[i] * FE.grid.volume4cells[cell];
                end    
            end
        end
    end
    #end    
end