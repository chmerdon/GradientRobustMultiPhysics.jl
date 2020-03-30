struct FACE_1dotVn <: FiniteElements.AbstractFEOperator end

function assemble_operator!(A::ExtendableSparseMatrix,::Type{FACE_1dotVn},FE::Union{AbstractH1FiniteElement, AbstractHdivFiniteElement}, factor::Float64 = 1.0)
    ensure_length4faces!(FE.grid);
    ensure_normal4faces!(FE.grid);
  
    # get quadrature formula
    T = eltype(FE.grid.coords4nodes);
    ET = FE.grid.elemtypes[1]
    ETF = Grid.get_face_elemtype(ET);
    quadorder = FiniteElements.get_polynomial_order(FE);
    qf = QuadratureFormula{T,typeof(ETF)}(quadorder);
     
    # generate caller for FE basis functions
    FEbasis = FiniteElements.FEbasis_caller_face(FE, qf);
    basisvals = zeros(Float64,FEbasis.ndofs4item,FEbasis.ncomponents)

    @assert FEbasis.ncomponents == size(FE.grid.normal4faces,2)
    
    # quadrature loop
    temp = 0.0;
    #@time begin      
    for face = 1 : size(FE.grid.nodes4faces,1)
         
        # update FEbasis on face
        FiniteElements.updateFEbasis!(FEbasis, face)
             
        for i in eachindex(qf.w)
            # get FE basis at quadrature point
            FiniteElements.getFEbasis4qp!(basisvals, FEbasis, i)
            
            for dof_i = 1 : FEbasis.ndofs4item
                @inbounds begin
                    temp = 0.0
                    for k = 1 : FEbasis.ncomponents
                        temp += basisvals[dof_i,k]*FE.grid.normal4faces[face,k];
                    end
                    temp *= factor * qf.w[i] * FE.grid.length4faces[face];
                    A[face,FEbasis.current_dofs[dof_i]] += temp; 
                end
            end
        end
    end    
    #end      
end