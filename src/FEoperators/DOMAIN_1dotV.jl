struct DOMAIN_1dotV <: FiniteElements.AbstractFEOperator end

function assemble_operator!(b, ::Type{DOMAIN_1dotV}, FE::AbstractFiniteElement)
    # get quadrature formula
    T = eltype(FE.grid.coords4nodes);
    ET = FE.grid.elemtypes[1]
    quadorder = FiniteElements.get_polynomial_order(FE);
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
     
    # generate caller for FE basis functions
    FEbasis = FiniteElements.FEbasis_caller(FE, qf, false);
    basisvals = zeros(Float64,FEbasis.ndofs4item,FEbasis.ncomponents)
    
    # quadrature loop
    #@time begin    
    for cell = 1 : size(FE.grid.nodes4cells,1);
            
        # update FEbasis on cell
        FiniteElements.updateFEbasis!(FEbasis, cell)
            
        for i in eachindex(qf.w)
                
            # get FE basis at quadrature point
            FiniteElements.getFEbasis4qp!(basisvals, FEbasis, i)
                
            for dof_i = 1 : FEbasis.ndofs4item
                # fill vector
                for k = 1 : FEbasis.ncomponents
                    @inbounds b[FEbasis.current_dofs[dof_i],k] += basisvals[dof_i,k] * qf.w[i] * FE.grid.volume4cells[cell];
                end    
            end
        end
    end
    #end    
end