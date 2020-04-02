struct CELL_UdotDIVV <: FiniteElements.AbstractFEOperator end

function assemble_operator!(A::ExtendableSparseMatrix, ::Type{CELL_UdotDIVV}, FEU::AbstractFiniteElement, FEV::Union{AbstractHdivFiniteElement, AbstractH1FiniteElement})
    
    # get quadrature formula
    T = eltype(FEU.grid.coords4nodes);
    ET = FEU.grid.elemtypes[1]
    quadorder = FiniteElements.get_polynomial_order(FEU) + FiniteElements.get_polynomial_order(FEV) - 1;
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
    
    # generate caller for FE basis functions
    FEbasisU = FiniteElements.FEbasis_caller(FEU, qf, false);
    FEbasisV = FiniteElements.FEbasis_caller(FEV, qf, true);
    basisvalsU = zeros(Float64,FEbasisU.ndofs4item,1)
    divergenceV = zeros(Float64,FEbasisV.ndofs4item,1);
    @assert FEbasisV.ncomponents == FEbasisV.xdim
    @assert FEbasisU.ncomponents == 1

    # quadrature loop
    temp = 0.0;
    x = zeros(T,FEbasisV.xdim)
    #@time begin    
    for cell = 1 : size(FEU.grid.nodes4cells,1)
      
        # update FEbasis on cell
        FiniteElements.updateFEbasis!(FEbasisU, cell)
        FiniteElements.updateFEbasis!(FEbasisV, cell)
      
        for i in eachindex(qf.w)
        
            # get FE basis gradients at quadrature point
            FiniteElements.getFEbasis4qp!(basisvalsU, FEbasisU, i)
            FiniteElements.getFEbasisdivergence4qp!(divergenceV, FEbasisV, i)
                
            for dof_i = 1 : FEbasisU.ndofs4item, dof_j = 1 : FEbasisV.ndofs4item
                # fill vector
                @inbounds begin
                    A[FEbasisV.current_dofs[dof_j],FEbasisU.current_dofs[dof_i]] += basisvalsU[dof_i,1] * divergenceV[dof_j,1] * qf.w[i] * FEU.grid.volume4cells[cell];
                end    
            end
        end
    end
    #end    
end