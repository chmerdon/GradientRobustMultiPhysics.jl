struct CELL_L2_FplusLA <: FiniteElements.AbstractFEOperator end

function assemble_operator!(b, ::Type{CELL_L2_FplusLA}, FE::AbstractH1FiniteElement, f!::Function, quadrature_order::Int, val4dofsA, nu::Real)
    
    # get quadrature formula
    T = eltype(FE.grid.coords4nodes);
    ET = FE.grid.elemtypes[1]
    quadorder = 2*(quadrature_order + FiniteElements.get_polynomial_order(FE) - 2);
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
    
    # generate caller for FE basis functions
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    ncomponents::Int = FiniteElements.get_ncomponents(FE);
    xdim::Int = size(FE.grid.coords4nodes,2)
    FEbasis = FiniteElements.FEbasis_caller(FE, qf, false, true); # <-- second bool is for 2nd derviatives
    laplacian = zeros(Float64,ndofs4cell,ncomponents);
    dofs = zeros(Int64,ndofs4cell)

    loc2glob_trafo = Grid.local2global(FE.grid,ET)
    
    # quadrature loop
    temp = 0.0;
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
            FiniteElements.getFEbasislaplacians4qp!(laplacian, FEbasis, i)
                
            # evaluate f
            x[:] = cell_trafo(qf.xref[i]);
            f!(fval, x)
                
            @inbounds begin
                for k = 1 : ncomponents
                    temp = 0.0
                    for dof_i = 1 : ndofs4cell
                        temp += laplacian[dof_i,k] * val4dofsA[dofs[dof_i]];
                    end
                    temp *= nu
                    temp += fval[k];
                    b[cell] += temp^2 * qf.w[i] * FE.grid.volume4cells[cell];
                end    
            end
        end
    end
    #end   
end