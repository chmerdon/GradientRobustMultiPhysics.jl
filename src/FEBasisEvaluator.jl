mutable struct FEBasisEvaluator{T <: Real, FE <: AbstractFiniteElement, EGEG <: AbstractElementGeometry, FEOP <: AbstractFEFunctionOperator, AT <: AbstractAssemblyType}
    ItemDofs::VariableTargetAdjacency    # link to ItemDofs
    L2G::L2GTransformer                  # local2global mapper
    L2GM::Array{T,2}                     # heap for transformation matrix (possibly tinverted)
    force_updateL2G::Bool                # update L2G on cell also if not needed?
    xref::Array{Array{T,1},1}            # xref of quadrature formula
    refbasisvals::Array{Array{T,2},1}    # basis evaluation on EG reference cell 
    refoperatorvals::Array{Array{T,2},1} # additional values to evaluate operator
    ncomponents::Int                     # number of FE components
    offsets::Array{Int32,1}              # offsets for gradient entries of each dof
    offsets2::Array{Int32,1}             # offsets for dof entries of each gradient (on ref)
    citem::Int                           # current item
    cvals::Array{Array{T,2},1}           # current operator vals on item
end

function vector_hessian(f, x)
    n = length(x)
    return ForwardDiff.jacobian(x -> ForwardDiff.jacobian(f, x), x)
end

function FEBasisEvaluator{T,FEType,EG,FEOP,AT}(FE::AbstractFiniteElement, qf::QuadratureRule; force_updateL2G = false) where {T <: Real, FEType <: AbstractFiniteElement, EG <: AbstractElementGeometry, FEOP <: AbstractFEFunctionOperator, AT <: AbstractAssemblyType}
    ItemDofs = FEPropertyDofs4AssemblyType(FE,AT)
    L2G = L2GTransformer{T, EG, FE.xgrid[CoordinateSystem]}(FE.xgrid,AT)
    L2GM = copy(L2G.A)

    refbasis = FiniteElements.get_basis_on_cell(FEType, EG)
    if AT <: Union{AbstractAssemblyTypeBFACE,AbstractAssemblyTypeFACE}
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

    derivorder = NeededDerivative4Operator(FEType,FEOP)
    edim = dim_element(EG)
    xdim = size(FE.xgrid[Coordinates],1)
    if derivorder > 0
        refoperatorvals = Array{Array{T,2},1}(undef,length(qf.w));
        current_eval = Array{Array{T,2},1}(undef,length(qf.w));
        for i in eachindex(qf.w)
            # evaluate gradients of basis function
            refoperatorvals[i] = ForwardDiff.jacobian(refbasis,qf.xref[i]);
            current_eval[i] = zeros(T,ndofs4item,ncomponents*edim)
        end 
    end
    if derivorder > 1
        refoperatorvals = Array{Array{T,2},1}(undef,length(qf.w));
        current_eval = Array{Array{T,2},1}(undef,length(qf.w));
        for i in eachindex(qf.w)
            # evaluate gradients of basis function
            refoperatorvals[i] = vector_hessian(refbasis,qf.xref[i])
            current_eval = zeros(T,ndofs4item,ncomponents*edim*edim);
        end   
    end
    if derivorder == 0
        refoperatorvals = [zeros(T,0,0)]
    end
    offsets = 0:edim:(ncomponents*edim);
    offsets2 = 0:ndofs4item:ncomponents*ndofs4item;

    if FEOP == Identity
        current_eval = deepcopy(refbasisvals)   
    end      
    
    return FEBasisEvaluator{T,FEType,EG,FEOP,AT}(ItemDofs,L2G,L2GM,force_updateL2G,qf.xref,refbasisvals,refoperatorvals,ncomponents,offsets,offsets2,0,current_eval)
end    


# IDENTITY OPERATOR
# H1 ELEMENTS (nothing has to be done)
function update!(FEBE::FEBasisEvaluator{T,FE,EG,FEOP,AT}, item::Int) where {T <: Real, FE <: AbstractH1FiniteElement, EG <: AbstractElementGeometry, FEOP <: Identity, AT <:AbstractAssemblyType}
    FEBE.citem = item
    
    if FEBE.force_updateL2G
        FEXGrid.update!(FEBE.L2G, item)
    end    
end


# IDENTITY OPERATOR
# H1 ELEMENTS WITH COEFFICIENTS (no transformation needed, just multiply coefficients)
function update!(FEBE::FEBasisEvaluator{T,FE,EG,FEOP}, item::Int) where {T <: Real, FE <: FiniteElements.AbstractH1FiniteElementWithCoefficients, EG <: AbstractElementGeometry, FEOP <: Identity, AT <:AbstractAssemblyType}
    FEBE.citem = item

    # only update if enforced by "user" (e.g. when a function is evaluated in operator)
    if FEBE.force_updateL2G
        FEXGrid.update!(FEBE.L2G, item)
    end    
end


# GRADIENT OPERATOR
# H1 ELEMENTS
# multiply tinverted jacobian of element trafo with gradient of basis function
# which yields (by chain rule) the gradient in x coordinates
#
# Note: for e.g. EDGE1D/CARTESIAN2D the tangentialderivative is produced,
#       i.e. the surface derivative in general
function update!(FEBE::FEBasisEvaluator{T,FE,EG,FEOP}, item::Int) where {T <: Real, FE <: AbstractH1FiniteElement, EG <: AbstractElementGeometry, FEOP <: Gradient, AT <:AbstractAssemblyType}
    FEBE.citem = item

    # update L2G (we need the matrix)
    FEXGrid.update!(FEBE.L2G, item)

    for i = 1 : length(FEBE.cvals)
        if FEBE.L2G.nonlinear || i == 1
            mapderiv!(FEBE.L2GM,FEBE.L2G,FEBE.xref[i])
        end
        for dof_i = 1 : FEBE.offsets2[2] # ndofs4item
            for c = 1 : FEBE.ncomponents, k = 1 : FEBE.offsets[2] # xdim
                FEBE.cvals[i][dof_i,k + FEBE.offsets[c]] = 0.0;
                for j = 1 : FEBE.offsets[2] # xdim
                    # compute duc/dxk
                    FEBE.cvals[i][dof_i,k + FEBE.offsets[c]] += FEBE.L2GM[k,j]*FEBE.refoperatorvals[i][dof_i + FEBE.offsets2[c],j]
                end    
                #cvals[i][dof_i,k + FEBE.offsets[c]] *= FEBC.coefficients[dof_i,c]
            end    
        end    
    end  

end

