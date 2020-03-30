struct BFACE_UdotV <: FiniteElements.AbstractFEOperator end

# matrix for L2 bestapproximation on boundary faces that writes into an ExtendableSparseMatrix
function assemble_operator!(A::ExtendableSparseMatrix,::Type{BFACE_UdotV},FE::AbstractFiniteElement, Dbids::Vector{Int64})
    ensure_bfaces!(FE.grid);
    ensure_length4faces!(FE.grid);
  
    # get quadrature formula
    T = eltype(FE.grid.coords4nodes);
    ET = FE.grid.elemtypes[1]
    ETF = Grid.get_face_elemtype(ET);
    quadorder = 2*FiniteElements.get_polynomial_order(FE);
    qf = QuadratureFormula{T,typeof(ETF)}(quadorder);
     
    # generate caller for FE basis functions
    FEbasis = FiniteElements.FEbasis_caller_face(FE, qf);
    basisvals = zeros(Float64,FEbasis.ndofs4item,FEbasis.ncomponents)
   
    # quadrature loop
    temp = 0.0;
    face = 0;
    #@time begin      
    for r = 1 : length(Dbids),  bface = 1 : size(FE.grid.bfaces,1)
        if FE.grid.bregions[bface] == Dbids[r]

            face = FE.grid.bfaces[bface];

            # update FEbasis on face
            FiniteElements.updateFEbasis!(FEbasis, face)
             
            for i in eachindex(qf.w)
                # get FE basis at quadrature point
                FiniteElements.getFEbasis4qp!(basisvals, FEbasis, i)
            
                for dof_i = 1 : FEbasis.ndofs4item, dof_j = dof_i : FEbasis.ndofs4item
                    # fill upper right part and diagonal of matrix
                    @inbounds begin
                        temp = 0.0
                        for k = 1 : FEbasis.ncomponents
                            temp += basisvals[dof_i,k]*basisvals[dof_j,k];
                        end
                        temp *= qf.w[i] * FE.grid.length4faces[face];
                        A[FEbasis.current_dofs[dof_i],FEbasis.current_dofs[dof_j]] += temp;
                        # fill lower left part of matrix
                        if dof_j > dof_i
                            A[FEbasis.current_dofs[dof_j],FEbasis.current_dofs[dof_i]] += temp;
                        end 
                    end
                end
            end
      end
    end    
    #end      
end