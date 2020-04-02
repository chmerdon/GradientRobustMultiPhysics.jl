struct DOMAIN_L2_FplusA <: FiniteElements.AbstractFEOperator end

function assemble_operator!(::Type{DOMAIN_L2_FplusA}, f!::Function, FE::AbstractFiniteElement, val4dofsA; degreeF::Int = 2, factorA::Real = 1.0)
    
    # get quadrature formula
    T = eltype(FE.grid.coords4nodes);
    ET = FE.grid.elemtypes[1]
    quadorder = 2*max(degreeF,FiniteElements.get_polynomial_order(FE));
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
    
    # generate caller for FE basis functions
    FEbasis = FiniteElements.FEbasis_caller(FE, qf, false);
    basisvals = zeros(Float64,FEbasis.ndofs4item,FEbasis.ncomponents);

    loc2glob_trafo = Grid.local2global(FE.grid,ET)
    
    # quadrature loop
    temp = 0.0;
    result = 0.0
    fval = zeros(T,FEbasis.ncomponents)
    x = zeros(T,FEbasis.xdim)
    #@time begin    
    for cell = 1 : size(FE.grid.nodes4cells,1)
      
        # update FEbasis on cell
        FiniteElements.updateFEbasis!(FEbasis, cell)    

        # get trafo
        cell_trafo = loc2glob_trafo(cell)
      
        for i in eachindex(qf.w)

            # get div(diffusion_matrix*DV) of FE basis functions at quadrature point
            FiniteElements.getFEbasis4qp!(basisvals, FEbasis, i)
                
            # evaluate f
            x[:] = cell_trafo(qf.xref[i]);
            f!(fval, x)
                
            @inbounds begin
                for k = 1 : FEbasis.ncomponents
                    temp = 0.0
                    for dof_i = 1 : FEbasis.ndofs4item
                        temp += basisvals[dof_i,k] * val4dofsA[FEbasis.current_dofs[dof_i]];
                    end
                    temp *= factorA
                    temp += fval[k];
                    result += temp^2 * qf.w[i] * FE.grid.volume4cells[cell];
                end    
            end
        end
    end
    #end   
    return result
end