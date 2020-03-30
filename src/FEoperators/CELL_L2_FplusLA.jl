struct CELL_L2_FplusLA <: FiniteElements.AbstractFEOperator end

function assemble_operator!(b, ::Type{CELL_L2_FplusLA}, FE::AbstractH1FiniteElement, f!::Function, quadrature_order::Int, val4dofsA, diffusion!, diffusion_quadorder::Int = 0)
    
    # get quadrature formula
    T = eltype(FE.grid.coords4nodes);
    ET = FE.grid.elemtypes[1]
    quadorder = 2*max(quadrature_order,FiniteElements.get_polynomial_order(FE) - 2 + diffusion_quadorder);
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
    
    # generate caller for FE basis functions
    FEbasis = FiniteElements.FEbasis_caller(FE, qf, false, true); # <-- second bool is for 2nd derviatives
    laplacian = zeros(Float64,FEbasis.ndofs4item,FEbasis.ncomponents);

    loc2glob_trafo = Grid.local2global(FE.grid,ET)

    diffusion_matrix = zeros(Float64,FEbasis.xdim,FEbasis.xdim)
    constant_diffusion = false
    if typeof(diffusion!) <: Real
        constant_diffusion = true
        for j= 1 : xdim
            diffusion_matrix[j,j] = diffusion!
        end      
    end
    
    # quadrature loop
    temp = 0.0;
    fval = zeros(T,FEbasis.xdim)
    x = zeros(T,FEbasis.xdim)
    #@time begin    
    for cell = 1 : size(FE.grid.nodes4cells,1)
      
        # update FEbasis on cell
        FiniteElements.updateFEbasis!(FEbasis, cell)    

        # get trafo
        cell_trafo = loc2glob_trafo(cell)
      
        for i in eachindex(qf.w)
        
            # evaluate diffusion matrix
            if constant_diffusion == false
                diffusion!(diffusion_matrix, x)
            end    

            # get div(diffusion_matrix*DV) of FE basis functions at quadrature point
            FiniteElements.getFEbasislaplacians4qp!(laplacian, FEbasis, i, diffusion_matrix)
                
            # evaluate f
            x[:] = cell_trafo(qf.xref[i]);
            f!(fval, x)
                
            @inbounds begin
                for k = 1 : FEbasis.ncomponents
                    temp = 0.0
                    for dof_i = 1 : FEbasis.ndofs4item
                        temp += laplacian[dof_i,k] * val4dofsA[FEbasis.current_dofs[dof_i]];
                    end
                    temp += fval[k];
                    b[cell] += temp^2 * qf.w[i] * FE.grid.volume4cells[cell];
                end    
            end
        end
    end
    #end   
end