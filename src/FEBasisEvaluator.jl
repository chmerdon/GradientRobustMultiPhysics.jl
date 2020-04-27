struct FEBasisEvaluator{T <: Real, FE <: AbstractFiniteElement, EGEG <: AbstractElementGeometry, FEOP <: AbstractFEFunctionOperator}
    ItemDofs::VariableTargetAdjacency
    local2global::L2GTransformer
    refbasisvals::Array{Array{Float64,2},1} # basis evaluation on EG reference cell
    refoperatorvals::Array{Float64,3} # additional values to evaluate operator
    ncomponents::Int
    ndofs4item::Int
    citem::Int
    cvals::Array{Float64,2}
end

function FEBasisEvaluator{T,FEType,EG,FEOP}(FE::AbstractFiniteElement, qf::QuadratureRule, AT::Type{<:AbstractAssemblyType}) where {T <: Real, FEType <: AbstractFiniteElement, EG <: AbstractElementGeometry, FEOP <: AbstractFEFunctionOperator}
    ItemDofs = FEPropertyDofs4AssemblyType(FE,AT)
    local2global = L2GTransformer{T, EG, FE.xgrid[CoordinateSystem]}(FE.xgrid,AT)

    refbasis = FiniteElements.get_basis_on_cell(FEType, EG)
    if AT <: AbstractAssemblyTypeFACE
        refbasis = FiniteElements.get_basis_on_face(FEType, EG)
    end    
    
    # pre-allocate memory for basis functions
    ncomponents = FiniteElements.get_ncomponents(FEType)
    refbasisvals = Array{Array{T,2},1}(undef,length(qf.w));
    for i in eachindex(qf.w)
        # evaluate basis functions at quadrature point
        if ncomponents == 1
            refbasisvals[i] = reshape(refbasis(qf.xref[i]),:,1)
        else
            refbasisvals[i] = refbasis(qf.xref[i])
        end    
    end    
    ndofs4item = size(refbasisvals[1],1)
    refoperatorvals = zeros(T,0,0,0)
    current_eval = copy(refbasisvals[1])
    #if with_standard_derivs == false
    #    refgradients = zeros(Float64,0,0,0)
    #else
    #    refgradients = zeros(Float64,length(qf.w),ncomponents*ndofs4cell,length(qf.xref[1]))
    #    for i in eachindex(qf.w)
    #        # evaluate gradients of basis function
    #        refgradients[i,:,:] = ForwardDiff.jacobian(FiniteElements.get_basis_on_cell(FE, ET),qf.xref[i]);
    #    end 
    #end

    #if with_higher_derivs == false
    #    refgradients_2ndorder = zeros(Float64,0,0,0)
    #else
    #    refgradients_2ndorder = zeros(Float64,length(qf.w),ncomponents*ndofs4cell*length(qf.xref[1]),length(qf.xref[1]))
    #    for i in eachindex(qf.w)
    #        # evaluate gradients of basis function
    #        #refgradients_2ndorder[i,:,:]
    #        refgradients_2ndorder[i,:,:] = vector_hessian(FiniteElements.get_basis_on_cell(FE, ET),qf.xref[i])
    #    end    
    #end
    return FEBasisEvaluator{T,FEType,EG,FEOP}(ItemDofs,local2global,refbasisvals,refoperatorvals,ncomponents,ndofs4item,0,current_eval)
end    



function update!(FEBE::FEBasisEvaluator{T,FE,EG,FEOP}, item::Int) where {T <: Real, FE <: AbstractFiniteElement, EG <: AbstractElementGeometry, FEOP <: AbstractFEFunctionOperator}
    # update transformation
    FEXGrid.update!(FEBE.local2global, item)

    # update coefficients

    # update further stuff depending on FEtype and FEoperator
end

function eval!(FEBE::FEBasisEvaluator{T,FE,EG,Identity}, qp::Int) where {T <: Real, FE <: AbstractH1FiniteElement, EG <: AbstractElementGeometry}
    # note : identity operator of H1 operator looks the same on each item
    for j = 1 : FEBE.ndofs4item, k = 1 : FEBE.ncomponents
        FEBE.cvals[j,k] = FEBE.refbasisvals[qp][j,k];
    end    
end