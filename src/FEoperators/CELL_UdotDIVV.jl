struct CELL_UdotDIVV <: FiniteElements.AbstractFEOperator end

function assemble_operator!(A::ExtendableSparseMatrix, ::Type{CELL_UdotDIVV}, FEU::AbstractFiniteElement, FEV::AbstractHdivFiniteElement)
    
    # get quadrature formula
    T = eltype(FEU.grid.coords4nodes);
    ET = FEU.grid.elemtypes[1]
    quadorder = FiniteElements.get_polynomial_order(FEU) + FiniteElements.get_polynomial_order(FEV) - 1;
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
    
    # generate caller for FE basis functions
    ndofs4cellU::Int = FiniteElements.get_ndofs4elemtype(FEU, ET);
    ndofs4cellV::Int = FiniteElements.get_ndofs4elemtype(FEV, ET);
    ncomponentsU::Int = FiniteElements.get_ncomponents(FEU);
    ncomponentsV::Int = FiniteElements.get_ncomponents(FEV);
    xdim::Int = size(FEU.grid.coords4nodes,2)

    @assert ncomponentsV == xdim
    @assert ncomponentsU == 1

    FEbasisU = FiniteElements.FEbasis_caller(FEU, qf, false);
    FEbasisV = FiniteElements.FEbasis_caller(FEV, qf, true);
    basisvalsU = zeros(Float64,ndofs4cellU,1)
    divergenceV = zeros(Float64,ndofs4cellV,1);
    dofsU = zeros(Int64,ndofs4cellU)
    dofsV = zeros(Int64,ndofs4cellV)

    
    # quadrature loop
    temp = 0.0;
    x = zeros(T,xdim)
    diagonal_entries = [1, 4]
    #@time begin    
    for cell = 1 : size(FEU.grid.nodes4cells,1)
      
        # update FEbasis on cell
        FiniteElements.updateFEbasis!(FEbasisU, cell)
        FiniteElements.updateFEbasis!(FEbasisV, cell)
      
        # get dofs
        FiniteElements.get_dofs_on_cell!(dofsU, FEU, cell, ET);  
        FiniteElements.get_dofs_on_cell!(dofsV, FEV, cell, ET);  
      
        for i in eachindex(qf.w)
        
            # get FE basis gradients at quadrature point
            FiniteElements.getFEbasis4qp!(basisvalsU, FEbasisU, i)
            FiniteElements.getFEbasisdivergence4qp!(divergenceV, FEbasisV, i)
                
            for dof_i = 1 : ndofs4cellU, dof_j = 1 : ndofs4cellV
                # fill vector
                @inbounds begin
                    A[dofsU[dof_i],dofsV[dof_j]] += basisvalsU[dof_i,1] * divergenceV[dof_j,1] * qf.w[i] * FEU.grid.volume4cells[cell];
                end    
            end
        end
    end
    #end    
end