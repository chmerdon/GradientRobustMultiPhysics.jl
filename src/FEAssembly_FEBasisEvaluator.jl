####################
# FEBasisEvaluator #
####################
#
# steers the evaluation of finite element basis functions within an assembly pattern
#
# during construction it gets assigned a
#   AbstractFunctionOperator: what is evaluted (e.g. Identity, Gradietn, Divergence...)
#   AbstractAssemblyType : cells, faces, bfaces? (to decide which dof field to use) 
#   ElementGeometry : triangles, quads? 
#   QuadratureRule : where to evaluate



abstract type AbstractFunctionOperator end # to dispatch which evaluator of the FE_basis_caller is used
abstract type Identity <: AbstractFunctionOperator end # 1*v_h
abstract type Gradient <: AbstractFunctionOperator end # D_geom(v_h)
abstract type SymmetricGradient <: AbstractFunctionOperator end # eps_geom(v_h)
abstract type Laplacian <: AbstractFunctionOperator end # L_geom(v_h)
abstract type Hessian <: AbstractFunctionOperator end # D^2(v_h)
abstract type Curl <: AbstractFunctionOperator end # only 2D: Curl(v_h) = D(v_h)^\perp
abstract type Rotation <: AbstractFunctionOperator end # only 3D: Rot(v_h) = D \times v_h
abstract type Divergence <: AbstractFunctionOperator end # div(v_h)
abstract type Trace <: AbstractFunctionOperator end # tr(v_h)
abstract type Deviator <: AbstractFunctionOperator end # dev(v_h)

NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Identity}) = 0
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Gradient}) = 1
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{SymmetricGradient}) = 1
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Laplacian}) = 2
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Hessian}) = 2
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Curl}) = 1
NeededDerivative4Operator(::Type{<:AbstractHcurlFiniteElement},::Type{Curl}) = 0
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Rotation}) = 1
NeededDerivative4Operator(::Type{<:AbstractHcurlFiniteElement},::Type{Rotation}) = 0
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Divergence}) = 1
NeededDerivative4Operator(::Type{<:AbstractHdivFiniteElement},::Type{Divergence}) = 0
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Trace}) = 0
NeededDerivative4Operator(::Type{<:AbstractFiniteElement},::Type{Deviator}) = 0

Length4Operator(::Type{Identity}, xdim::Int) = 1
Length4Operator(::Type{Divergence}, xdim::Int) = 1
Length4Operator(::Type{Trace}, xdim::Int) = 1
Length4Operator(::Type{Curl}, xdim::Int) = (xdim == 2) ? 1 : xdim
Length4Operator(::Type{Gradient}, xdim::Int) = xdim
Length4Operator(::Type{SymmetricGradient}, xdim::Int) = (xdim == 2) ? 3 : 6
Length4Operator(::Type{Hessian}, xdim::Int) = xdim*xdim

QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{Identity}) = 0
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{Gradient}) = -1
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{Divergence}) = -1
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{SymmetricGradient}) = -1
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{Laplacian}) = -2
QuadratureOrderShift4Operator(::Type{<:AbstractFiniteElement},::Type{Hessian}) = -2


mutable struct FEBasisEvaluator{T <: Real, FEType <: AbstractFiniteElement, EGEG <: AbstractElementGeometry, FEOP <: AbstractFunctionOperator, AT <: AbstractAssemblyType}
    FE::AbstractFiniteElement            # linke to full FE (e.g. for coefficients)
    ItemDofs::VariableTargetAdjacency    # link to ItemDofs
    L2G::L2GTransformer                  # local2global mapper
    L2GM::Array{T,2}                     # heap for transformation matrix (possibly tinverted)
    force_updateL2G::Bool                # update L2G on cell also if not needed?
    xref::Array{Array{T,1},1}            # xref of quadrature formula
    refbasisvals::Array{Array{T,2},1}    # basis evaluation on EG reference cell 
    refoperatorvals::Array{Array{T,2},1} # additional values to evaluate operator
    ncomponents::Int                     # number of FE components
    offsets::Array{Int,1}              # offsets for gradient entries of each dof
    offsets2::Array{Int,1}             # offsets for dof entries of each gradient (on ref)
    citem::Int                           # current item
    cvals::Array{Array{T,2},1}           # current operator vals on item
    coefficients::Array{T,2}             # coefficients
end

function vector_hessian(f, x)
    n = length(x)
    return ForwardDiff.jacobian(x -> ForwardDiff.jacobian(f, x), x)
end

function FEBasisEvaluator{T,FEType,EG,FEOP,AT}(FE::AbstractFiniteElement, qf::QuadratureRule; force_updateL2G = false) where {T <: Real, FEType <: AbstractFiniteElement, EG <: AbstractElementGeometry, FEOP <: AbstractFunctionOperator, AT <: AbstractAssemblyType}
    ItemDofs = FEPropertyDofs4AssemblyType(FE,AT)
    L2G = L2GTransformer{T, EG, FE.xgrid[CoordinateSystem]}(FE.xgrid,AT)
    L2GM = copy(L2G.A)

    if AT <: Union{AbstractAssemblyTypeBFACE,AbstractAssemblyTypeFACE}
        refbasis = FiniteElements.get_basis_on_face(FEType, EG)
    else
        refbasis = FiniteElements.get_basis_on_cell(FEType, EG)
    end    
    
    # pre-allocate memory for basis functions
    ncomponents = FiniteElements.get_ncomponents(FEType)
    refbasisvals = Array{Array{T,2},1}(undef,length(qf.w));
    for i in eachindex(qf.w)
        # evaluate basis functions at quadrature point
        if ncomponents == 1
            refbasisvals[i] = reshape(refbasis(qf.xref[i]),1,:)
        else
            refbasisvals[i] = refbasis(qf.xref[i]')'
        end    
    end    
    ndofs4item = size(refbasisvals[1],2)

    derivorder = NeededDerivative4Operator(FEType,FEOP)
    edim = dim_element(EG)
    xdim = size(FE.xgrid[Coordinates],1)
    if derivorder > 0
        refoperatorvals = Array{Array{T,2},1}(undef,length(qf.w));
        current_eval = Array{Array{T,2},1}(undef,length(qf.w));
        for i in eachindex(qf.w)
            # evaluate gradients of basis function
            # = list of vectors [du_k/dx_1; du_k,dx_2]
            refoperatorvals[i] = ForwardDiff.jacobian(refbasis,qf.xref[i]);
            current_eval[i] = zeros(T,ncomponents*edim,ndofs4item)
        end 
    end
    if derivorder > 1
        refoperatorvals = Array{Array{T,2},1}(undef,length(qf.w));
        current_eval = Array{Array{T,2},1}(undef,length(qf.w));
        for i in eachindex(qf.w)
            # evaluate gradients of basis function
            refoperatorvals[i] = vector_hessian(refbasis,qf.xref[i])
            current_eval = zeros(T,ncomponents*edim*edim,ndofs4item);
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
    
    xref = copy(qf.xref)
    if FEType <: Union{AbstractH1FiniteElementWithCoefficients, AbstractHdivFiniteElement}
        coefficients = zeros(T,ncomponents,ndofs4item)
    else
        coefficients = zeros(T,0,0)
    end    

    return FEBasisEvaluator{T,FEType,EG,FEOP,AT}(FE,ItemDofs,L2G,L2GM,force_updateL2G,xref,refbasisvals,refoperatorvals,ncomponents,offsets,offsets2,0,current_eval,coefficients)
end    


# IDENTITY OPERATOR
# H1 ELEMENTS (nothing has to be done)
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,FEOP,AT}, item::Int) where {T <: Real, FEType <: AbstractH1FiniteElement, EG <: AbstractElementGeometry, FEOP <: Identity, AT <:AbstractAssemblyType}
    FEBE.citem = item
    
    if FEBE.force_updateL2G
        FEXGrid.update!(FEBE.L2G, item)
    end    
end


# IDENTITY OPERATOR
# H1 ELEMENTS WITH COEFFICIENTS
# (no transformation needed, just multiply coefficients)
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,FEOP,AT}, item::Int) where {T <: Real, FEType <: FiniteElements.AbstractH1FiniteElementWithCoefficients, EG <: AbstractElementGeometry, FEOP <: Identity, AT <:AbstractAssemblyType}
    FEBE.citem = item

    if AT <: Union{AbstractAssemblyTypeBFACE,AbstractAssemblyTypeFACE}
        FiniteElements.get_coefficients_on_face!(FEBE.coefficients, FEBE.FE, EG, item)
    else
        FiniteElements.get_coefficients_on_cell!(FEBE.coefficients, FEBE.FE, EG, item)
    end    


    for i = 1 : length(FEBE.cvals)
        for dof_i = 1 : FEBE.offsets2[2] # ndofs4item
            for k = 1 : FEBE.offsets[2] # ncomponents
                FEBE.cvals[i][k,dof_i] = FEBE.refbasisvals[i][k,dof_i] * FEBE.coefficients[k,dof_i];
            end    
        end
    end
    
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
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,FEOP}, item::Int) where {T <: Real, FEType <: AbstractH1FiniteElement, EG <: AbstractElementGeometry, FEOP <: Gradient, AT <:AbstractAssemblyType}
    if FEBE.citem != item
        FEBE.citem = item

        # update L2G (we need the matrix)
        FEXGrid.update!(FEBE.L2G, item)

        for i = 1 : length(FEBE.cvals)
            if FEBE.L2G.nonlinear || i == 1
                mapderiv!(FEBE.L2GM,FEBE.L2G,FEBE.xref[i])
            end
            for dof_i = 1 : FEBE.offsets2[2] # ndofs4item
                for c = 1 : FEBE.ncomponents, k = 1 : FEBE.offsets[2] # xdim
                    FEBE.cvals[i][k + FEBE.offsets[c],dof_i] = 0.0;
                    for j = 1 : FEBE.offsets[2] # xdim
                        # compute duc/dxk
                        FEBE.cvals[i][k + FEBE.offsets[c],dof_i] += FEBE.L2GM[k,j]*FEBE.refoperatorvals[i][dof_i + FEBE.offsets2[c],j]
                    end    
                    #cvals[i][dof_i,k + FEBE.offsets[c]] *= FEBC.coefficients[dof_i,c]
                end    
            end    
        end  
    end    
end

# GRADIENT OPERATOR
# H1 ELEMENTS WITH COEFFICIENTS
# multiply tinverted jacobian of element trafo with gradient of basis function
# which yields (by chain rule) the gradient in x coordinates
#
# Note: for e.g. EDGE1D/CARTESIAN2D the tangentialderivative is produced,
#       i.e. the surface derivative in general
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,FEOP,AT}, item::Int) where {T <: Real, FEType <: AbstractH1FiniteElementWithCoefficients, EG <: AbstractElementGeometry, FEOP <: Gradient, AT <:AbstractAssemblyType}
    if FEBE.citem != item
        FEBE.citem = item

        # update L2G (we need the matrix)
        FEXGrid.update!(FEBE.L2G, item)

        # get coefficients
        if AT <: Union{AbstractAssemblyTypeBFACE,AbstractAssemblyTypeFACE}
            FiniteElements.get_coefficients_on_face!(FEBE.coefficients, FEBE.FE, EG, item)
        else
            FiniteElements.get_coefficients_on_cell!(FEBE.coefficients, FEBE.FE, EG, item)
        end    

        for i = 1 : length(FEBE.cvals)
            if FEBE.L2G.nonlinear || i == 1
                mapderiv!(FEBE.L2GM,FEBE.L2G,FEBE.xref[i])
            end
            for dof_i = 1 : FEBE.offsets2[2] # ndofs4item
                for c = 1 : FEBE.ncomponents, k = 1 : FEBE.offsets[2] # xdim
                    FEBE.cvals[i][k + FEBE.offsets[c],dof_i] = 0.0;
                    for j = 1 : FEBE.offsets[2] # xdim
                        # compute duc/dxk
                        FEBE.cvals[i][k + FEBE.offsets[c],dof_i] += FEBE.L2GM[k,j]*FEBE.refoperatorvals[i][dof_i + FEBE.offsets2[c],j]
                    end    
                    FEBE.cvals[i][k + FEBE.offsets[c], dof_i] *= FEBE.coefficients[c, dof_i]
                end    
            end    
        end  
    end    
end

# DIVERGENCE OPERATOR
# H1 ELEMENTS
# multiply tinverted jacobian of element trafo with gradient of basis function
# which yields (by chain rule) the gradient in x coordinates
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,FEOP}, item::Int) where {T <: Real, FEType <: AbstractH1FiniteElement, EG <: AbstractElementGeometry, FEOP <: Divergence, AT <:AbstractAssemblyType}
    if FEBE.citem != item
        FEBE.citem = item

        # update L2G (we need the matrix)
        FEXGrid.update!(FEBE.L2G, item)

        for i = 1 : length(FEBE.cvals)
            if FEBE.L2G.nonlinear || i == 1
                mapderiv!(FEBE.L2GM,FEBE.L2G,FEBE.xref[i])
            end
            for dof_i = 1 : FEBE.offsets2[2] # ndofs4item
                FEBE.cvals[i][1,dof_i] = 0.0;
                for k = 1 : FEBE.offsets[2] # xdim
                    for j = 1 : FEBE.offsets[2] # xdim
                        # compute duk/dxk
                        FEBE.cvals[i][1,dof_i] += FEBE.L2GM[k,j]*FEBE.refoperatorvals[i][dof_i + FEBE.offsets2[k],j]
                    end    
                end    
            end    
        end  
    end    
end

# DIVERGENCE OPERATOR
# H1 ELEMENTS WITH COEFFICIENTS
# multiply tinverted jacobian of element trafo with gradient of basis function
# which yields (by chain rule) the gradient in x coordinates
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,FEOP,AT}, item::Int) where {T <: Real, FEType <: AbstractH1FiniteElementWithCoefficients, EG <: AbstractElementGeometry, FEOP <: Divergence, AT <:AbstractAssemblyType}
    if FEBE.citem != item
        FEBE.citem = item

        # update L2G (we need the matrix)
        FEXGrid.update!(FEBE.L2G, item)

        # get coefficients
        if AT <: Union{AbstractAssemblyTypeBFACE,AbstractAssemblyTypeFACE}
            FiniteElements.get_coefficients_on_face!(FEBE.coefficients, FEBE.FE, EG, item)
        else
            FiniteElements.get_coefficients_on_cell!(FEBE.coefficients, FEBE.FE, EG, item)
        end    

        for i = 1 : length(FEBE.cvals)
            if FEBE.L2G.nonlinear || i == 1
                mapderiv!(FEBE.L2GM,FEBE.L2G,FEBE.xref[i])
            end
            for dof_i = 1 : FEBE.offsets2[2] # ndofs4item
                FEBE.cvals[i][1,dof_i] = 0.0;
                for k = 1 : FEBE.offsets[2] # xdim
                    for j = 1 : FEBE.offsets[2] # xdim
                        # compute duk/dxk
                        FEBE.cvals[i][1,dof_i] += FEBE.L2GM[k,j]*FEBE.refoperatorvals[i][dof_i + FEBE.offsets2[k],j] * FEBE.coefficients[k, dof_i]
                    end    
                end    
            end    
        end  
    end    
end

