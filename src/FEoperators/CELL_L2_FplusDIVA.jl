struct CELL_L2_FplusDIVA <: FiniteElements.AbstractFEOperator end

function assemble_operator!(b, ::Type{CELL_L2_FplusDIVA}, FE::AbstractFiniteElement, f!::Function, quadrature_order::Int, val4dofsA);#, diffusion!, diffusion_quadorder::Int = 0)
    
    # get quadrature formula
    T = eltype(FE.grid.coords4nodes);
    ET = FE.grid.elemtypes[1]
    quadorder = 2*max(quadrature_order,FiniteElements.get_polynomial_order(FE) - 1);# + diffusion_quadorder);
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
    
    # generate caller for FE basis functions
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    xdim::Int = size(FE.grid.coords4nodes,2)
    FEbasis = FiniteElements.FEbasis_caller(FE, qf, true);
    divergences = zeros(Float64,ndofs4cell,1);
    dofs = zeros(Int64,ndofs4cell)

    loc2glob_trafo = Grid.local2global(FE.grid,ET)

    #diffusion_matrix = zeros(Float64,xdim,xdim)
    #constant_diffusion = false
    #if typeof(diffusion!) <: Real
    #    constant_diffusion = true
    #    for j= 1 : xdim
    #        diffusion_matrix[j,j] = diffusion!
    #    end      
    #end
    
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
        
            # evaluate diffusion matrix
            #if constant_diffusion == false
            #    diffusion!(diffusion_matrix, x)
            #end    

            # get div(diffusion_matrix*DV) of FE basis functions at quadrature point
            FiniteElements.getFEbasisdivergence4qp!(divergences, FEbasis, i)
                
            # evaluate f
            x[:] = cell_trafo(qf.xref[i]);
            f!(fval, x)
                
            @inbounds begin
                temp = 0.0
                for dof_i = 1 : ndofs4cell
                    temp += divergences[dof_i,1] * val4dofsA[dofs[dof_i]];
                end
                temp += fval[1];
                b[cell] += temp^2 * qf.w[i] * FE.grid.volume4cells[cell];
            end
        end
    end
    #end   
end