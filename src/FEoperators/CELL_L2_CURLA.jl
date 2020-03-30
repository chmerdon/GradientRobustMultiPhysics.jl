struct CELL_L2_CURLA <: FiniteElements.AbstractFEOperator end

function assemble_operator!(b, ::Type{CELL_L2_CURLA}, FE::AbstractFiniteElement, val4dofsA) #, diffusion!, diffusion_quadorder::Int = 0)
    
    # get quadrature formula
    T = eltype(FE.grid.coords4nodes);
    ET = FE.grid.elemtypes[1]
    quadorder = 2*FiniteElements.get_polynomial_order(FE) - 2;
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
    
    # generate caller for FE basis functions
    if typeof(FE) <: AbstractH1FiniteElement
        FEbasis = FiniteElements.FEbasis_caller(FE, qf, true);
    elseif typeof(FE) <: AbstractHdivFiniteElement
        FEbasis = FiniteElements.FEbasis_caller(FE, qf, false, true); # <-- second bool needed for for curl derviatives
    end    
    curls = zeros(Float64,FEbasis.ndofs4item,1);

    # diffusion_matrix = zeros(Float64,xdim,xdim)
    # constant_diffusion = false
    # if typeof(diffusion!) <: Real
    #     constant_diffusion = true
    #     for j= 1 : xdim
    #         diffusion_matrix[j,j] = diffusion!
    #     end      
    # end
    
    # quadrature loop
    temp = 0.0
    #@time begin    
    for cell = 1 : size(FE.grid.nodes4cells,1)
      
        # update FEbasis on cell
        FiniteElements.updateFEbasis!(FEbasis, cell)
      
        for i in eachindex(qf.w)
        
            # evaluate diffusion matrix
            #if constant_diffusion == false
            #    diffusion!(diffusion_matrix, x)
            #end    

            # get div(diffusion_matrix*DV) of FE basis functions at quadrature point
            FiniteElements.getFEbasiscurls4qp!(curls, FEbasis, i)
                
            @inbounds begin
                temp = 0.0
                for dof_i = 1 : FEbasis.ndofs4item
                    temp += curls[dof_i,1] * val4dofsA[FEbasis.current_dofs[dof_i]];
                end
                b[cell] += temp^2 * qf.w[i] * FE.grid.volume4cells[cell];
            end
        end
    end
    #end   
end