struct DOMAIN_1dotV <: FiniteElements.AbstractFEOperator end

function assemble_operator!(b, ::Type{DOMAIN_1dotV}, FE::AbstractFiniteElement)
    # get quadrature formula
    T = eltype(FE.grid.coords4nodes);
    ET = FE.grid.elemtypes[1]
    quadorder = FiniteElements.get_polynomial_order(FE);
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
     
    # generate caller for FE basis functions
    ncells::Int = size(FE.grid.nodes4cells,1);
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    ncomponents::Int = FiniteElements.get_ncomponents(FE);
    FEbasis = FiniteElements.FEbasis_caller(FE, qf, false);
    basisvals = zeros(Float64,ndofs4cell,ncomponents)
    dofs = zeros(Int64,ndofs4cell)
    
    # quadrature loop
    #@time begin    
    for cell = 1 : ncells

        # get dofs
        FiniteElements.get_dofs_on_cell!(dofs, FE, cell, ET);
            
        # update FEbasis on cell
        FiniteElements.updateFEbasis!(FEbasis, cell)
            
        for i in eachindex(qf.w)
                
            # get FE basis at quadrature point
            FiniteElements.getFEbasis4qp!(basisvals, FEbasis, i)
                
            for dof_i = 1 : ndofs4cell
                # fill vector
                for k = 1 : ncomponents
                    @inbounds b[dofs[dof_i],k] += basisvals[dof_i,k] * qf.w[i] * FE.grid.volume4cells[cell];
                end    
            end
        end
    end
    #end    
end