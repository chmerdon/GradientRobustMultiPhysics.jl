####################
# FEBasisEvaluator #
####################
#
# steers the evaluation of finite element basis functions within an assembly pattern
#
# during construction it gets assigned a
#   AbstractFunctionOperator: what is evaluted (e.g. Identity, Gradient, Divergence...)
#   AbstractAssemblyType : cells, faces, bfaces? (to decide which dof field to use) 
#   ElementGeometry : triangles, quads? 
#   QuadratureRule : where to evaluate

mutable struct FEBasisEvaluator{T <: Real, FEType <: AbstractFiniteElement, EG <: AbstractElementGeometry, FEOP <: AbstractFunctionOperator, AT <: AbstractAssemblyType}
    FE::FESpace                          # link to full FE (e.g. for coefficients)
    FE2::FESpace                         # link to reconstruction FE
    ItemDofs::Union{VariableTargetAdjacency,SerialVariableTargetAdjacency,Array{Int32,2}}    # link to ItemDofs
    L2G::L2GTransformer                  # local2global mapper
    L2GM::Array{T,2}                     # heap for transformation matrix (possibly tinverted)
    L2GM2::Array{T,2}                    # 2nd heap for transformation matrix (e.g. Piola + mapderiv)
    iteminfo::Array{T,1}                 # (e.g. current determinant for Hdiv, current tangent)
    xref::Array{Array{T,1},1}            # xref of quadrature formula
    refbasisvals::Array{Array{T,2},1}    # basis evaluation on EG reference cell 
    refbasisderivvals::Array{T,3}        # additional values to evaluate operator
    derivorder::Int                      # order of derivatives that are needed
    Dresult                              # DiffResults for ForwardDiff handling
    Dcfg                                 # config for ForwardDiff handling
    ncomponents::Int                     # number of FE components
    offsets::Array{Int,1}                # offsets for gradient entries of each dof
    offsets2::Array{Int,1}               # offsets for dof entries of each gradient (on ref)
    citem::Int                           # current item
    cvals::Array{T,3}                    # current operator vals on item
    coefficients::Array{T,2}             # coefficients for finite element
    coefficients2::Array{T,2}            # coefficients for reconstruction
    coefficients3::Array{T,2}            # coefficients for operator (e.g. TangentialGradient)
    coeffs_handler                       # function to call to get coefficients for finite element
    reconstcoeffs_handler                # function to call to get reconstruction coefficients
    compressiontargets::Array{Int,1}     # some operators allow for compressed storage (e.g. SymmetricGradient)
end

function vector_hessian(f, x)
    n = length(x)
    return ForwardDiff.jacobian(x -> ForwardDiff.jacobian(f, x), x)
end



function FEBasisEvaluator{T,FEType,EG,FEOP,AT}(FE::FESpace, qf::QuadratureRule; verbosity::Int = 0) where {T <: Real, FEType <: AbstractFiniteElement, EG <: AbstractElementGeometry, FEOP <: AbstractFunctionOperator, AT <: AbstractAssemblyType}
    ItemDofs = Dofmap4AssemblyType(FE, DofitemAT4Operator(AT, FEOP))
    L2G = L2GTransformer{T, EG, FE.xgrid[CoordinateSystem]}(FE.xgrid,AT)
    L2GM = copy(L2G.A)
    L2GM2 = copy(L2G.A)


    if verbosity > 0
        println("  ...constructing FEBasisEvaluator for $FEType, EG = $EG, operator = $FEOP")
    end

    # collect basis function information
    ncomponents::Int = get_ncomponents(FEType)
    ndofs4item::Int = 0
    if AT <: Union{ON_BFACES,<:ON_FACES,<:ON_EDGES,ON_BEDGES}
        if FEType <: AbstractHdivFiniteElement
            refbasis = get_basis_normalflux_on_face(FEType, EG)
            ndofs4item = get_ndofs_on_face(FEType, EG)
            ncomponents = 1
        elseif FEType <: AbstractHcurlFiniteElement
                refbasis = get_basis_tangentflux_on_edge(FEType, EG)
                ndofs4item = get_ndofs_on_edge(FEType, EG)
                ncomponents = 1
        else
            refbasis = get_basis_on_face(FEType, EG)
            ndofs4item = get_ndofs_on_face(FEType, EG)
        end
    else
        refbasis = get_basis_on_cell(FEType, EG)
        ndofs4item = get_ndofs_on_cell(FEType, EG)
    end    

    # evaluate basis on reference domain
    refbasisvals = Array{Array{T,2},1}(undef,length(qf.w));
    for i in eachindex(qf.w)
        # evaluate basis functions at quadrature point
        refbasisvals[i] = zeros(T,ndofs4item,ncomponents)
        refbasis(refbasisvals[i], qf.xref[i])
    end    

    # set coefficient handlers needed for basis evaluation
    coefficients = zeros(T,0,0)
    coeff_handler = nothing
    if FEType <: Union{AbstractH1FiniteElementWithCoefficients, AbstractHdivFiniteElement, AbstractHcurlFiniteElement}
        coefficients = zeros(T,ncomponents,ndofs4item)
        if AT == ON_CELLS
            coeff_handler = get_coefficients_on_cell!(FE, EG)
        elseif AT <: Union{ON_BFACES,<:ON_FACES}
            coeff_handler = get_coefficients_on_face!(FE, EG)
        elseif AT <: Union{ON_BEDGES,<:ON_EDGES}
            coeff_handler = get_coefficients_on_edge!(FE, EG)
        end
    end    

    # compute refbasisderivvals and further coefficients needed for operator eval
    derivorder = NeededDerivative4Operator(FEType,FEOP)
    edim = dim_element(EG)
    xdim = size(FE.xgrid[Coordinates],1)
    xref = copy(qf.xref)
    resultdim = Int(Length4Operator(FEOP,edim,ncomponents))
    coefficients2 = zeros(T,0,0)
    coefficients3 = zeros(T,0,0)
    offsets = 0:edim:(ncomponents*edim);
    offsets2 = 0:ndofs4item:ncomponents*ndofs4item;
    compressiontargets = zeros(T,0)
    Dcfg = nothing
    Dresult = nothing
    current_eval = zeros(T,resultdim,ndofs4item,length(qf.w))
    if derivorder == 0
        refbasisderivvals = zeros(T,0,0,0);
        if FEOP == NormalFlux || (FEOP == TangentFlux && edim == 2)
            coefficients3 = FE.xgrid[FaceNormals]
            # current_eval = zeros(T,1,ndofs4item,length(qf.w));
        elseif FEOP == TangentFlux && edim == 3
            coefficients3 = FE.xgrid[EdgeTangents]
            # current_eval = zeros(T,1,ndofs4item,length(qf.w));
        elseif FEOP == Identity || FEOP == IdentityDisc{Jump}
            # current_eval = zeros(T,ncomponents,ndofs4item,length(qf.w));
            for i in eachindex(qf.w), j = 1 : ndofs4item, k = 1 : ncomponents
                current_eval[k,j,i] = refbasisvals[i][j,k]
            end
        elseif FEOP == IdentityComponent
            for i in eachindex(qf.w), j = 1 : ndofs4item
                current_eval[1,j,i] = refbasisvals[i][j,FEOP.parameters[1]]
            end
        end
    elseif derivorder == 1
        if FEOP == TangentialGradient
            coefficients3 = FE.xgrid[FaceNormals]
        end

        # derivatives of the basis on the reference domain are computed
        # by ForwardDiff, to minimise memory allocations and be able
        # to rebase the quadrature points later (e.g. when evaluating cuts through cells)
        # we use DiffResults and JacobianConfig of ForwardDiff and save these in the struct
        result_temp = zeros(Float64,ndofs4item,ncomponents)
        result_temp2 = zeros(Real,ndofs4item,ncomponents)
        input_temp = Vector{Float64}(undef,edim)
        jac_temp = Matrix{Float64}(undef,ndofs4item*ncomponents,edim)
        Dresult = DiffResults.DiffResult(result_temp,jac_temp)
        jac::Array{Float64,2} = DiffResults.jacobian(Dresult)
        Dcfg = ForwardDiff.JacobianConfig(refbasis, result_temp, input_temp)

        refbasisderivvals = zeros(T,ndofs4item*ncomponents,edim,length(qf.w));
       # current_eval = zeros(T,ncomponents*edim,ndofs4item,length(qf.w));
        for i in eachindex(qf.w)
            # evaluate gradients of basis function
            # = list of vectors [du_k/dx_1; du_k,dx_2]
            fill!(result_temp,0.0)
            ForwardDiff.jacobian!(Dresult, refbasis, result_temp, qf.xref[i], Dcfg)
            jac = DiffResults.jacobian(Dresult)

            for j = 1 : ndofs4item*ncomponents, k = 1 : edim
                refbasisderivvals[j,k,i] = jac[j,k];
            end
        end

        # specifications for compressed operator (Voigt notation)
        if FEOP == SymmetricGradient
            # the following mapping tells where each entry of the full gradient lands in the reduced vector
            if edim == 1
                compressiontargets = [1,1]
            elseif edim == 2
                # 2D Voigt accumulation positions of du1/dx1, du1/dx2, du2/dx1, d2/dx2
                compressiontargets = [1,3,3,2]
            elseif edim == 3
                # 3D Voigt accumulation positions of du1/dx1, du1/dx2, du1/dx3, d2/dx1,...
                compressiontargets = [1,6,5,6,2,4,5,4,3] 
            end
        end

   # elseif derivorder == 2
   #     refbasisderivvals = zeros(T,ndofs4item*ncomponents*edim,edim,length(qf.w));
   #     #current_eval = zeros(T,ncomponents*edim*edim,ndofs4item,length(qf.w));
   #     for i in eachindex(qf.w)
   #         # evaluate gradients of basis function
   #         refbasisderivvals[:,:,i] = vector_hessian(refbasis,qf.xref[i])
   #     end   
    end

    if verbosity > 0
        println("     size(cvals) = $(size(current_eval))")
    end

    return FEBasisEvaluator{T,FEType,EG,FEOP,AT}(FE,FE, ItemDofs,L2G,L2GM,L2GM2,zeros(T,xdim+1),xref,refbasisvals,refbasisderivvals,derivorder,Dresult,Dcfg,ncomponents,offsets,offsets2,0,current_eval,coefficients, coefficients2, coefficients3, coeff_handler, nothing, compressiontargets)
end    

# constructor for ReconstructionIdentity, ReconstructionDivergence
function FEBasisEvaluator{T,FEType,EG,FEOP,AT}(FE::FESpace, qf::QuadratureRule; verbosity::Int = 0) where {T <: Real, FEType <: AbstractFiniteElement, FETypeReconst <: AbstractFiniteElement, EG <: AbstractElementGeometry, FEOP <: Union{<:ReconstructionIdentity{FETypeReconst},ReconstructionDivergence{FETypeReconst},<:ReconstructionGradient{FETypeReconst}}, AT <: AbstractAssemblyType}
    
    if verbosity > 0
        println("  ...constructing FEBasisEvaluator for $FEOP operator of $FEType on $EG")
    end

    # generate reconstruction space
    # avoid computation of full dofmap
    # we will just use local basis functions
    FE2 = FESpace{FETypeReconst}(FE.xgrid; dofmaps_needed = [])
    ItemDofs = Dofmap4AssemblyType(FE, DofitemAT4Operator(AT, FEOP))
    L2G = L2GTransformer{T, EG, FE.xgrid[CoordinateSystem]}(FE.xgrid,AT)
    L2GM = copy(L2G.A)
    L2GM2 = copy(L2G.A)

    # collect basis function information
    ncomponents::Int = get_ncomponents(FEType)
    ncomponents2::Int = get_ncomponents(FETypeReconst)
    ndofs4item::Int = 0
    ndofs4item2::Int = 0
    if AT <: Union{ON_BFACES,<:ON_FACES}
        if FETypeReconst <: AbstractHdivFiniteElement
            refbasis = get_basis_on_face(FEType, EG)
            refbasis_reconst = get_basis_normalflux_on_face(FETypeReconst, EG)
            ncomponents2 = 1
            ndofs4item = get_ndofs_on_face(FEType, EG)
            ndofs4item2 = get_ndofs_on_face(FETypeReconst, EG)
        else
            refbasis = get_basis_on_face(FEType, EG)
            refbasis_reconst = get_basis_on_face(FETypeReconst, EG)
            ndofs4item = get_ndofs_on_face(FEType, EG)
            ndofs4item2 = get_ndofs_on_face(FETypeReconst, EG)
        end
    else
        refbasis = get_basis_on_cell(FEType, EG)
        refbasis_reconst = get_basis_on_cell(FETypeReconst, EG)
        ndofs4item = get_ndofs_on_cell(FEType, EG)
        ndofs4item2 = get_ndofs_on_cell(FETypeReconst, EG)
    end    

    # evaluate reconstruction basis
    refbasisvals = Array{Array{T,2},1}(undef,length(qf.w));
    for i in eachindex(qf.w)
        # evaluate basis functions at quadrature point
        refbasisvals[i] = zeros(T,ndofs4item2,ncomponents2)
        refbasis_reconst(refbasisvals[i], qf.xref[i])
    end    



    # set coefficient handlers needed for basis evaluation (for both FEType and FETypereconst)
    if FETypeReconst <: Union{AbstractH1FiniteElementWithCoefficients, AbstractHdivFiniteElement}
        coefficients = zeros(T,ncomponents,ndofs4item2)
        if AT == ON_CELLS
            coeff_handler = get_coefficients_on_cell!(FE2, EG)
        elseif AT <: Union{ON_BFACES,<:ON_FACES}
            coeff_handler = get_coefficients_on_face!(FE2, EG)
        end
    else
        coefficients = zeros(T,0,0)
        coeff_handler = nothing
    end    
    coefficients2 = zeros(T,ndofs4item,ndofs4item2)

    # compute refbasisderivvals and further coefficients needed for operator eval
    xref = copy(qf.xref)
    edim = dim_element(EG)
    xdim = size(FE.xgrid[Coordinates],1)
    offsets = 0:edim:(ncomponents*edim);
    offsets2 = 0:ndofs4item2:ncomponents*ndofs4item2;
    coefficients3 = zeros(T,0,0)
    Dcfg = nothing
    Dresult = nothing
    resultdim = Int(Length4Operator(FEOP,edim,ncomponents))
    current_eval = zeros(T,resultdim,ndofs4item,length(qf.w))
    derivorder = NeededDerivative4Operator(FEType,FEOP)
    if derivorder == 0
        # refbasisderivvals are used as a cache fpr the reconstruction basis
        refbasisderivvals = zeros(T,ncomponents,ndofs4item2,length(qf.w));
        current_eval = zeros(T,ncomponents,ndofs4item,length(qf.w));
    elseif derivorder == 1

        # derivatives of the reconstruction basis on the reference domain are computed
        # by ForwardDiff, to minimise memory allocations and be able
        # to rebase the quadrature points later (e.g. when evaluating cuts through cells)
        # we use DiffResults and JacobianConfig of ForwardDiff and save these in the struct
        result_temp = zeros(Float64,ndofs4item2,ncomponents2)
        result_temp2 = zeros(Real,ndofs4item2,ncomponents2)
        input_temp = Vector{Float64}(undef,edim)
        jac_temp = Matrix{Float64}(undef,ndofs4item2*ncomponents2,edim)
        Dresult = DiffResults.DiffResult(result_temp,jac_temp)
        jac::Array{Float64,2} = DiffResults.jacobian(Dresult)
        Dcfg = ForwardDiff.JacobianConfig(refbasis_reconst, result_temp, input_temp)

        refbasisderivvals = zeros(T,ndofs4item2*ncomponents2,edim,length(qf.w));
        # current_eval = zeros(T,ncomponents*edim,ndofs4item,length(qf.w));
        for i in eachindex(qf.w)
            # evaluate gradients of basis function
            # = list of vectors [du_k/dx_1; du_k,dx_2]
            fill!(result_temp,0.0)
            ForwardDiff.jacobian!(Dresult, refbasis_reconst, result_temp, qf.xref[i], Dcfg)
            jac = DiffResults.jacobian(Dresult)

            for j = 1 : ndofs4item*ncomponents, k = 1 : edim
                refbasisderivvals[j,k,i] = jac[j,k];
            end
        end
    end

    # get reconstruction coefficient handlers
    if AT <: Union{ON_BFACES,<:ON_FACES}
        rcoeff_handler = get_reconstruction_coefficients_on_face!(FE, FE2, EG)
    else
        rcoeff_handler = get_reconstruction_coefficients_on_cell!(FE, FE2, EG)
    end

    if verbosity > 0
        println("     size(cvals) = $(size(current_eval))")
    end
    
    return FEBasisEvaluator{T,FEType,EG,FEOP,AT}(FE,FE2, ItemDofs,L2G,L2GM,L2GM2,zeros(T,xdim+1),xref,refbasisvals,refbasisderivvals,derivorder,Dresult,Dcfg,ncomponents,offsets,offsets2,0,current_eval,coefficients, coefficients2,coefficients3,coeff_handler, rcoeff_handler,[])
end    


# IDENTITY OPERATOR
# H1 ELEMENTS (nothing has to be done)
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,<:Identity,AT}, item::Int) where {T <: Real, FEType <: AbstractH1FiniteElement, EG <: AbstractElementGeometry, AT <:AbstractAssemblyType}
    FEBE.citem = item
    return nothing
end

# IDENTITYCOMPONENT OPERATOR
# H1 ELEMENTS (nothing has to be done)
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,<:IdentityComponent,AT}, item::Int) where {T <: Real, FEType <: AbstractH1FiniteElement, EG <: AbstractElementGeometry, AT <:AbstractAssemblyType}
    FEBE.citem = item
    return nothing
end

# RECONSTRUCTION IDENTITY OPERATOR
# H1 ELEMENTS
# HDIV RECONSTRUCTION
# Piola transform Hdiv reference basis and multiply Hdiv coefficients and Trafo coefficients
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,FEOP,AT}, item::Int) where {T <: Real, FEType <: AbstractH1FiniteElement, EG <: AbstractElementGeometry, AT <:AbstractAssemblyType, FETypeReconst <: AbstractFiniteElement, FEOP <: ReconstructionIdentity{FETypeReconst}}
    if FEBE.citem != item
        FEBE.citem = item
    
        # cell update transformation
        update!(FEBE.L2G, item)
        FEBE.coeffs_handler(FEBE.coefficients, item)

        # use Piola transformation on Hdiv basis
        # and save it in refbasisderivvals
        for i = 1 : length(FEBE.xref)
            # evaluate Piola matrix at quadrature point
            if FEBE.L2G.nonlinear || i == 1
                FEBE.iteminfo[1] = piola!(FEBE.L2GM,FEBE.L2G,FEBE.xref[i])
            end
            for dof_i = 1 : size(FEBE.refbasisderivvals,2) # ndofs4item (Hdiv)
                for k = 1 : FEBE.offsets[2] # ncomponents
                    FEBE.refbasisderivvals[k,dof_i,i] = 0.0;
                    for l = 1 : FEBE.offsets[2] # ncomponents
                        FEBE.refbasisderivvals[k,dof_i,i] += FEBE.L2GM[k,l]*FEBE.refbasisvals[i][dof_i,l];
                    end    
                    FEBE.refbasisderivvals[k,dof_i,i] *= FEBE.coefficients[k,dof_i] / FEBE.iteminfo[1]
                end
            end
        end

        # get local reconstruction coefficients
        # and accumulate
        # get_reconstruction_coefficients_on_cell!(FEBE.coefficients2, FEBE.FE, FEBE.FE2, EG, item)
        FEBE.reconstcoeffs_handler(FEBE.coefficients2, item)

        fill!(FEBE.cvals,0.0)
        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : size(FEBE.cvals,2), dof_j = 1 : size(FEBE.refbasisderivvals,2) # ndofs4item (Hdiv)
                if FEBE.coefficients2[dof_i,dof_j] != 0
                    for k = 1 : FEBE.offsets[2] # ncomponents
                        FEBE.cvals[k,dof_i,i] += FEBE.coefficients2[dof_i,dof_j] * FEBE.refbasisderivvals[k,dof_j,i]; 
                    end
                end
            end
        end
    end
    return nothing
end


# IDENTITY OPERATOR
# Hcurl ELEMENTS (covariant Piola trafo)
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,<:Union{Identity,IdentityDisc{Jump}},AT}, item::Int) where {T <: Real, FEType <: AbstractHcurlFiniteElement, EG <: AbstractElementGeometry, AT <:AbstractAssemblyType}
    if FEBE.citem != item
        FEBE.citem = item
    
        # cell update transformation
        update!(FEBE.L2G, item)
        FEBE.coeffs_handler(FEBE.coefficients, item)

        # use Piola transformation on basisvals
        for i = 1 : length(FEBE.xref)
            # evaluate Piola matrix at quadrature point
            if FEBE.L2G.nonlinear || i == 1
                mapderiv!(FEBE.L2GM,FEBE.L2G,FEBE.xref[i])
            end
            for dof_i = 1 : FEBE.offsets2[2] # ndofs4item
                for k = 1 : FEBE.offsets[2] # ncomponents
                    FEBE.cvals[k,dof_i,i] = 0.0;
                    for l = 1 : FEBE.offsets[2] # ncomponents
                        FEBE.cvals[k,dof_i,i] += FEBE.L2GM[k,l]*FEBE.refbasisvals[i][dof_i,l];
                    end    
                    FEBE.cvals[k,dof_i,i] *= FEBE.coefficients[k,dof_i]
                end
            end
        end
    end
    return nothing
end

# IDENTITY OPERATOR
# Hdiv ELEMENTS (Piola trafo)
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,<:Union{Identity,IdentityDisc{Jump}},AT}, item::Int) where {T <: Real, FEType <: AbstractHdivFiniteElement, EG <: AbstractElementGeometry, AT <:AbstractAssemblyType}
    if FEBE.citem != item
        FEBE.citem = item
    
        # cell update transformation
        update!(FEBE.L2G, item)
        FEBE.coeffs_handler(FEBE.coefficients, item)

        # use Piola transformation on basisvals
        for i = 1 : length(FEBE.xref)
            # evaluate Piola matrix at quadrature point
            if FEBE.L2G.nonlinear || i == 1
                FEBE.iteminfo[1] = piola!(FEBE.L2GM,FEBE.L2G,FEBE.xref[i])
            end
            for dof_i = 1 : FEBE.offsets2[2] # ndofs4item
                for k = 1 : FEBE.offsets[2] # ncomponents
                    FEBE.cvals[k,dof_i,i] = 0.0;
                    for l = 1 : FEBE.offsets[2] # ncomponents
                        FEBE.cvals[k,dof_i,i] += FEBE.L2GM[k,l]*FEBE.refbasisvals[i][dof_i,l];
                    end    
                    FEBE.cvals[k,dof_i,i] *= FEBE.coefficients[k,dof_i] / FEBE.iteminfo[1]
                end
            end
        end
    end
    return nothing
end



# IDENTITY OPERATOR
# H1 ELEMENTS WITH COEFFICIENTS
# (no transformation needed, just multiply coefficients)
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,Identity,AT}, item::Int) where {T <: Real, FEType <: AbstractH1FiniteElementWithCoefficients, EG <: AbstractElementGeometry, AT <:AbstractAssemblyType}
    if FEBE.citem != item
        FEBE.citem = item
        
        # get coefficients
        FEBE.coeffs_handler(FEBE.coefficients, item)

        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : FEBE.offsets2[2] # ndofs4item
                for k = 1 : FEBE.offsets[2] # ncomponents
                    FEBE.cvals[k,dof_i,i] = FEBE.refbasisvals[i][dof_i,k] * FEBE.coefficients[k,dof_i];
                end    
            end
        end
    end
    return nothing
end


# IDENTITYCOMPONENT OPERATOR
# H1 ELEMENTS WITH COEFFICIENTS
# (no transformation needed, just multiply coefficients)
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,FEOP,AT}, item::Int) where {T <: Real, FEType <: AbstractH1FiniteElementWithCoefficients, FEOP <: IdentityComponent, EG <: AbstractElementGeometry, AT <:AbstractAssemblyType}
    if FEBE.citem != item
        FEBE.citem = item
        
        # get coefficients
        FEBE.coeffs_handler(FEBE.coefficients, item)

        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : FEBE.offsets2[2] # ndofs4item
                FEBE.cvals[1,dof_i,i] = FEBE.refbasisvals[i,dof_i,FEOP.parameters[1]] * FEBE.coefficients[FEOP.parameters[1],dof_i];
            end
        end
    end
    return nothing
end


# IDENTITYCOMPONENT OPERATOR
# Hdiv ELEMENTS (Piola trafo)
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,FEOP,AT}, item::Int) where {T <: Real, FEType <: AbstractHdivFiniteElement, FEOP <: IdentityComponent, EG <: AbstractElementGeometry, AT <:AbstractAssemblyType}
    if FEBE.citem != item
        FEBE.citem = item
    
        # cell update transformation
        update!(FEBE.L2G, item)
        FEBE.coeffs_handler(FEBE.coefficients, item)

        # use Piola transformation on basisvals
        for i = 1 : length(FEBE.xref)
            # evaluate Piola matrix at quadrature point
            if FEBE.L2G.nonlinear || i == 1
                FEBE.iteminfo[1] = piola!(FEBE.L2GM,FEBE.L2G,FEBE.xref[i])
            end
            for dof_i = 1 : FEBE.offsets2[2] # ndofs4item
                FEBE.cvals[1,dof_i,i] = 0.0;
                for l = 1 : FEBE.offsets[2] # ncomponents
                    FEBE.cvals[1,dof_i,i] += FEBE.L2GM[FEOP.parameters[1],l]*FEBE.refbasisvals[i][dof_i,l];
                end    
                FEBE.cvals[1,dof_i,i] *= FEBE.coefficients[FEOP.parameters[1],dof_i] / FEBE.iteminfo[1]
            end
        end
    end
    return nothing
end



# NORMALFLUX OPERATOR
# H1 ELEMENTS
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,NormalFlux,AT}, item::Int) where {T <: Real, FEType <: AbstractH1FiniteElement, EG <: AbstractElementGeometry, AT <:AbstractAssemblyType}
    if FEBE.citem != item
        FEBE.citem = item

        # fetch normal of item
        for k = 1 : size(FEBE.coefficients3,1) # ncomponents of normal
            FEBE.iteminfo[k] = FEBE.coefficients3[k,item]
        end

        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : FEBE.offsets2[2] # ndofs4item
                FEBE.cvals[1,dof_i,i] = 0.0
                for k = 1 : size(FEBE.coefficients3,1) # ncomponents of normal
                    FEBE.cvals[1,dof_i,i] += FEBE.refbasisvals[i][dof_i,k] * FEBE.iteminfo[k];
                end    
            end
        end
    end
    return nothing
end

# NORMALFLUX OPERATOR
# H1 ELEMENTS WITH COEFFICIENTS
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,NormalFlux,AT}, item::Int) where {T <: Real, FEType <: AbstractH1FiniteElementWithCoefficients, EG <: AbstractElementGeometry, AT <:AbstractAssemblyType}
    if FEBE.citem != item
        FEBE.citem = item

        # fetch normal of item
        for k = 1 : size(FEBE.coefficients3,1) # ncomponents of normal
            FEBE.iteminfo[k] = FEBE.coefficients3[k,item]
        end
        
        # get coefficients
        FEBE.coeffs_handler(FEBE.coefficients, item)

        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : FEBE.offsets2[2] # ndofs4item
                FEBE.cvals[1,dof_i,i] = 0.0
                for k = 1 : size(FEBE.coefficients3,1) # ncomponents of normal
                    FEBE.cvals[1,dof_i,i] += FEBE.refbasisvals[i][dof_i,k] * FEBE.coefficients[k,dof_i] * FEBE.iteminfo[k];
                end    
            end
        end
    end
    return nothing
end

# NORMALFLUX OPERATOR
# Hdiv ELEMENTS (just divide by face volume)
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,FEOP,AT}, item::Int) where {T <: Real, FEType <: AbstractHdivFiniteElement, EG <: AbstractElementGeometry, FEOP <: NormalFlux, AT <:AbstractAssemblyType}
    if FEBE.citem != item
        FEBE.citem = item
    
        # use Piola transformation on basisvals
        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : FEBE.offsets2[2], k = 1 : FEBE.ncomponents
                FEBE.cvals[k,dof_i,i] = FEBE.refbasisvals[i][dof_i,k] / FEBE.L2G.ItemVolumes[item]
            end
        end
    end
    return nothing
end


# TANGENTLFLUX OPERATOR
# Hdiv ELEMENTS (just divide by face volume)
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,FEOP,AT}, item::Int) where {T <: Real, FEType <: AbstractHcurlFiniteElement, EG <: AbstractElementGeometry, FEOP <: TangentFlux, AT <:AbstractAssemblyType}
    if FEBE.citem != item
        FEBE.citem = item
    
        # use Piola transformation on basisvals
        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : FEBE.offsets2[2], k = 1 : FEBE.ncomponents
                FEBE.cvals[k,dof_i,i] = FEBE.refbasisvals[i][dof_i,k] / FEBE.L2G.ItemVolumes[item]
            end
        end
    end
    return nothing
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
        update!(FEBE.L2G, item)

        for i = 1 : length(FEBE.xref)
            if FEBE.L2G.nonlinear || i == 1
                mapderiv!(FEBE.L2GM,FEBE.L2G,FEBE.xref[i])
            end
            for dof_i = 1 : FEBE.offsets2[2] # ndofs4item
                for c = 1 : FEBE.ncomponents, k = 1 : FEBE.offsets[2] # edim
                    FEBE.cvals[k + FEBE.offsets[c],dof_i,i] = 0.0;
                    for j = 1 : FEBE.offsets[2] # edim
                        # compute duc/dxk
                        FEBE.cvals[k + FEBE.offsets[c],dof_i,i] += FEBE.L2GM[k,j]*FEBE.refbasisderivvals[dof_i + FEBE.offsets2[c],j,i]
                    end    
                end    
            end    
        end  
    end  
    return nothing  
end



# GRADIENT OPERATOR
# Hdiv ELEMENTS (Piola trafo)
#
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,<:Gradient,AT}, item::Int) where {T <: Real, FEType <: AbstractHdivFiniteElement, EG <: AbstractElementGeometry, AT <:AbstractAssemblyType}
    if FEBE.citem != item
        FEBE.citem = item
    
        # cell update transformation
        update!(FEBE.L2G, item)
        FEBE.coeffs_handler(FEBE.coefficients, item)

        # use Piola transformation on basisvals
        fill!(FEBE.cvals,0.0);
        for i = 1 : length(FEBE.xref)
            # evaluate Piola matrix at quadrature point
            if FEBE.L2G.nonlinear || i == 1
                FEBE.iteminfo[1] = piola!(FEBE.L2GM,FEBE.L2G,FEBE.xref[i])
                mapderiv!(FEBE.L2GM2,FEBE.L2G,FEBE.xref[i])
            end
            for dof_i = 1 : FEBE.offsets2[2] # ndofs4item
                for c = 1 : FEBE.ncomponents, k = 1 : FEBE.offsets[2] # edim
                    # compute duc/dxk
                    for j = 1 : FEBE.offsets[2] # ncomponents
                        for m = 1 : FEBE.offsets[2]
                            FEBE.cvals[k + FEBE.offsets[c],dof_i,i] += FEBE.L2GM2[k,m] * FEBE.L2GM[c,j] * FEBE.refbasisderivvals[dof_i + FEBE.offsets2[j],m,i];
                        end
                    end    
                    FEBE.cvals[k + FEBE.offsets[c],dof_i,i] *= FEBE.coefficients[c,dof_i] / FEBE.iteminfo[1]
                end
            end
        end
    end
    return nothing
end


# CURLSCALAR OPERATOR
# H1 ELEMENTS
#
# This operator can only be applied to scalar elements and produces the rotated 2D Gradient
# only works in 2D/Cartesian2D at the moment
#
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,FEOP}, item::Int) where {T <: Real, FEType <: AbstractH1FiniteElement, EG <: AbstractElementGeometry, FEOP <: CurlScalar, AT <:AbstractAssemblyType}
    if FEBE.citem != item
        FEBE.citem = item

        # update L2G (we need the matrix)
        update!(FEBE.L2G, item)

        for i = 1 : length(FEBE.xref)
            if FEBE.L2G.nonlinear || i == 1
                mapderiv!(FEBE.L2GM,FEBE.L2G,FEBE.xref[i])
            end
            for dof_i = 1 : FEBE.offsets2[2] # ndofs4item
                FEBE.cvals[1,dof_i,i] = 0.0;
                FEBE.cvals[2,dof_i,i] = 0.0;
                for j = 1 : FEBE.offsets[2] # edim
                    FEBE.cvals[1,dof_i,i] -= FEBE.L2GM[2,j]*FEBE.refbasisderivvals[dof_i,j,i] # -du/dy
                    FEBE.cvals[2,dof_i,i] += FEBE.L2GM[1,j]*FEBE.refbasisderivvals[dof_i,j,i] # du/dx
                end    
            end    
        end  
    end  
    return nothing  
end


# CURL2D OPERATOR
# H1 ELEMENTS
#
# This operator can only be applied to two-dimensional vector fields and produces the 1D curl
# only works in 2D/Cartesian2D at the moment
#
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,FEOP}, item::Int) where {T <: Real, FEType <: AbstractH1FiniteElement, EG <: AbstractElementGeometry, FEOP <: Curl2D, AT <:AbstractAssemblyType}
    if FEBE.citem != item
        FEBE.citem = item

        # update L2G (we need the matrix)
        update!(FEBE.L2G, item)

        for i = 1 : length(FEBE.xref)
            if FEBE.L2G.nonlinear || i == 1
                mapderiv!(FEBE.L2GM,FEBE.L2G,FEBE.xref[i])
            end
            for dof_i = 1 : FEBE.offsets2[2] # ndofs4item
                FEBE.cvals[1,dof_i,i] = 0.0;
                for j = 1 : FEBE.offsets[2] # edim
                    FEBE.cvals[1,dof_i,i] -= FEBE.L2GM[2,j]*FEBE.refbasisderivvals[dof_i,j,i]  # -du1/dy
                    FEBE.cvals[1,dof_i,i] += FEBE.L2GM[1,j]*FEBE.refbasisderivvals[dof_i + FEBE.offsets2[2],j,i]  # du2/dx
                end    
            end    
        end  
    end  
    return nothing  
end


# TANGENTGRADIENT OPERATOR
# H1 ELEMENTS
#
# only 1D/Cartesian2D at the moment
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,FEOP}, item::Int) where {T <: Real, FEType <: AbstractH1FiniteElement, EG <: AbstractElementGeometry, FEOP <: TangentialGradient, AT <:AbstractAssemblyType}
    if FEBE.citem != item
        FEBE.citem = item

        # update L2G (we need the matrix)
        update!(FEBE.L2G, item)

        # compute tangent of item
        FEBE.iteminfo[1] = FEBE.coefficients3[2,item]
        FEBE.iteminfo[2] = -FEBE.coefficients3[1,item]

        for i = 1 : length(FEBE.xref)
            if FEBE.L2G.nonlinear || i == 1
                mapderiv!(FEBE.L2GM,FEBE.L2G,FEBE.xref[i])
            end
            for dof_i = 1 : FEBE.offsets2[2] # ndofs4item
                for c = 1 : FEBE.ncomponents, k = 1 : FEBE.offsets[2] # edim
                    FEBE.cvals[k + FEBE.offsets[c],dof_i,i] = 0.0;
                    for j = 1 : FEBE.offsets[2] # edim
                        # compute duc/dxk
                        FEBE.cvals[1,dof_i,i] += FEBE.L2GM[k,j]*FEBE.refbasisderivvals[dof_i + FEBE.offsets2[c],j,i] * FEBE.iteminfo[c]
                    end    
                end    
            end    
        end  
    end    
    return nothing
end


# SYMMETRIC GRADIENT OPERATOR
# H1 ELEMENTS
# multiply tinverted jacobian of element trafo with gradient of basis function
# which yields (by chain rule) the gradient in x coordinates
# symmetric matrices are saved in reduced Voigt notation (compression specified by FEBE.compressiontargets)
# in 1D: (du1/dx1)
# in 2D: (du1/dx1, du2/dx2, du1/dx2 + du2/dx1)
# in 3D: (du1/dx1, du2/dx2, du3/dx3, du1/dx2 + du2/dx1, du1/dx3 + du3/dx1, du2/dx3 + du3/dx2)

function update!(FEBE::FEBasisEvaluator{T,FEType,EG,FEOP}, item::Int) where {T <: Real, FEType <: AbstractH1FiniteElement, EG <: AbstractElementGeometry, FEOP <: SymmetricGradient, AT <:AbstractAssemblyType}
    if FEBE.citem != item
        FEBE.citem = item

        # update L2G (we need the matrix)
        update!(FEBE.L2G, item)

        fill!(FEBE.cvals,0.0)
        for i = 1 : length(FEBE.xref)
            if FEBE.L2G.nonlinear || i == 1
                mapderiv!(FEBE.L2GM,FEBE.L2G,FEBE.xref[i])
            end
            for dof_i = 1 : FEBE.offsets2[2] # ndofs4item
                for c = 1 : FEBE.ncomponents, k = 1 : FEBE.offsets[2] # edim
                    for j = 1 : FEBE.offsets[2] # edim
                        # compute duc/dxk and put it into the right spot in the Voigt vector
                        FEBE.cvals[FEBE.compressiontargets[k + FEBE.offsets[c]],dof_i,i] += FEBE.L2GM[k,j]*FEBE.refbasisderivvals[dof_i + FEBE.offsets2[c],j,i]
                    end    
                end    
            end    
        end  
    end    
    return nothing
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
        update!(FEBE.L2G, item)

        # get coefficients
        FEBE.coeffs_handler(FEBE.coefficients, item)

        for i = 1 : length(FEBE.xref)
            if FEBE.L2G.nonlinear || i == 1
                mapderiv!(FEBE.L2GM,FEBE.L2G,FEBE.xref[i])
            end
            for dof_i = 1 : FEBE.offsets2[2] # ndofs4item
                for c = 1 : FEBE.ncomponents, k = 1 : FEBE.offsets[2] # edim
                    FEBE.cvals[k + FEBE.offsets[c],dof_i,i] = 0.0;
                    for j = 1 : FEBE.offsets[2] # edim
                        # compute duc/dxk
                        FEBE.cvals[k + FEBE.offsets[c],dof_i,i] += FEBE.L2GM[k,j]*FEBE.refbasisderivvals[dof_i + FEBE.offsets2[c],j,i]
                    end    
                    FEBE.cvals[k + FEBE.offsets[c], dof_i,i] *= FEBE.coefficients[c, dof_i]
                end    
            end    
        end  
    end   
    return nothing 
end

# DIVERGENCE OPERATOR
# H1 ELEMENTS
# multiply tinverted jacobian of element trafo with gradient of basis function
# which yields (by chain rule) the gradient in x coordinates
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,Divergence}, item::Int) where {T <: Real, FEType <: AbstractH1FiniteElement, EG <: AbstractElementGeometry, AT <:AbstractAssemblyType}
    if FEBE.citem != item
        FEBE.citem = item

        # update L2G (we need the matrix)
        update!(FEBE.L2G, item)

        for i = 1 : length(FEBE.xref)
            if FEBE.L2G.nonlinear || i == 1
                mapderiv!(FEBE.L2GM,FEBE.L2G,FEBE.xref[i])
            end
            for dof_i = 1 : FEBE.offsets2[2] # ndofs4item
                FEBE.cvals[1,dof_i,i] = 0.0;
                for k = 1 : FEBE.offsets[2] # edim
                    for j = 1 : FEBE.offsets[2] # edim
                        # compute duk/dxk
                        FEBE.cvals[1,dof_i,i] += FEBE.L2GM[k,j]*FEBE.refbasisderivvals[dof_i + FEBE.offsets2[k],j,i]
                    end    
                end    
            end    
        end  
    end    
    return nothing
end

# DIVERGENCE OPERATOR
# H1 ELEMENTS WITH COEFFICIENTS
# multiply tinverted jacobian of element trafo with gradient of basis function
# which yields (by chain rule) the gradient in x coordinates
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,Divergence,AT}, item::Int) where {T <: Real, FEType <: AbstractH1FiniteElementWithCoefficients, EG <: AbstractElementGeometry, AT <:AbstractAssemblyType}
    if FEBE.citem != item
        FEBE.citem = item

        # update L2G (we need the matrix)
        update!(FEBE.L2G, item)

        # get coefficients
        FEBE.coeffs_handler(FEBE.coefficients, item)

        for i = 1 : length(FEBE.xref)
            if FEBE.L2G.nonlinear || i == 1
                mapderiv!(FEBE.L2GM,FEBE.L2G,FEBE.xref[i])
            end
            for dof_i = 1 : FEBE.offsets2[2] # ndofs4item
                FEBE.cvals[1,dof_i,i] = 0.0;
                for k = 1 : FEBE.offsets[2] # edim
                    for j = 1 : FEBE.offsets[2] # edim
                        # compute duk/dxk
                        FEBE.cvals[1,dof_i,i] += FEBE.L2GM[k,j]*FEBE.refbasisderivvals[dof_i + FEBE.offsets2[k],j,i] * FEBE.coefficients[k, dof_i]
                    end    
                end    
            end    
        end  
    end  
    return nothing  
end


# RECONSTRUCTION DIVERGENCE OPERATOR
# H1 ELEMENTS
# HDIV RECONSTRUCTION
# Piola transform Hdiv reference basis and multiply Hdiv coefficients and Trafo coefficients
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,FEOP,AT}, item::Int) where {T <: Real, FEType <: AbstractH1FiniteElement, EG <: AbstractElementGeometry, AT <:AbstractAssemblyType, FETypeReconst <: AbstractFiniteElement, FEOP <: ReconstructionDivergence{FETypeReconst}}
    if FEBE.citem != item
        FEBE.citem = item
    
        # cell update transformation
        update!(FEBE.L2G, item)
        FEBE.coeffs_handler(FEBE.coefficients, item)

        # get local reconstruction coefficients
        FEBE.reconstcoeffs_handler(FEBE.coefficients2, item)

        # use Piola transformation on Hdiv basis
        # and accumulate according to reconstruction coefficients
        fill!(FEBE.cvals,0.0)
        for i = 1 : length(FEBE.xref)
            # evaluate Piola matrix at quadrature point
            if FEBE.L2G.nonlinear || i == 1
                FEBE.iteminfo[1] = piola!(FEBE.L2GM,FEBE.L2G,FEBE.xref[i])
            end
            for dof_i = 1 : size(FEBE.cvals,2) # ndofs4item (H1)
                for dof_j = 1 : FEBE.offsets2[2] # ndofs4item (Hdiv)
                    if FEBE.coefficients2[dof_i,dof_j] != 0
                        for j = 1 : FEBE.offsets[2] # edim
                            FEBE.cvals[1,dof_i,i] += FEBE.coefficients2[dof_i,dof_j] * FEBE.refbasisderivvals[dof_j + FEBE.offsets2[j],j,i] * FEBE.coefficients[1,dof_j]/FEBE.iteminfo[1]
                        end  
                    end
                end
            end
        end  
    end
    return nothing
end


# RECONSTRUCTION GRADIENT OPERATOR
# Hdiv ELEMENTS (Piola trafo)
#
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,FEOP,AT}, item::Int) where {T <: Real, FEType <: AbstractH1FiniteElementWithCoefficients, EG <: AbstractElementGeometry, AT <:AbstractAssemblyType, FETypeReconst <: AbstractFiniteElement, FEOP <: ReconstructionGradient{FETypeReconst}}
    if FEBE.citem != item
        FEBE.citem = item
    
        # cell update transformation
        update!(FEBE.L2G, item)
        FEBE.coeffs_handler(FEBE.coefficients, item)

        # get local reconstruction coefficients
        FEBE.reconstcoeffs_handler(FEBE.coefficients2, item)

        # use Piola transformation on basisvals
        fill!(FEBE.cvals,0.0)
        for i = 1 : length(FEBE.xref)
            # evaluate Piola matrix at quadrature point
            if FEBE.L2G.nonlinear || i == 1
                FEBE.iteminfo[1] = piola!(FEBE.L2GM,FEBE.L2G,FEBE.xref[i])
                mapderiv!(FEBE.L2GM2,FEBE.L2G,FEBE.xref[i])
            end
            for dof_i = 1 : size(FEBE.cvals,2) # ndofs4item
                for dof_j = 1 : FEBE.offsets2[2] # ndofs4item (Hdiv)
                    if FEBE.coefficients2[dof_i,dof_j] != 0
                        for c = 1 : FEBE.ncomponents, k = 1 : FEBE.offsets[2] # edim
                            # compute duc/dxk
                            for j = 1 : FEBE.offsets[2] # ncomponents
                                for m = 1 : FEBE.offsets[2]
                                    FEBE.cvals[k + FEBE.offsets[c],dof_i,i] += FEBE.coefficients2[dof_i,dof_j] * FEBE.L2GM2[k,m] * FEBE.L2GM[c,j] * FEBE.refbasisderivvals[dof_j + FEBE.offsets2[j],m,i] * FEBE.coefficients[c,dof_j] / FEBE.iteminfo[1];
                                end
                            end    
                        end
                    end
                end
            end
        end
    end
    return nothing
end


# DIVERGENCE OPERATOR
# HDIV ELEMENTS
# Piola transformation preserves divergence (up to a factor 1/det(A))
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,Divergence}, item::Int) where {T <: Real, FEType <: AbstractHdivFiniteElement, EG <: AbstractElementGeometry, AT <:AbstractAssemblyType}
    if FEBE.citem != item
        FEBE.citem = item
        
        # cell update transformation
        update!(FEBE.L2G, item)
        FEBE.coeffs_handler(FEBE.coefficients, item)

        # use Piola transformation on basisvals
        for i = 1 : length(FEBE.xref)
            # evaluate Piola matrix at quadrature point
            if FEBE.L2G.nonlinear || i == 1
                FEBE.iteminfo[1] = piola!(FEBE.L2GM,FEBE.L2G,FEBE.xref[i])
            end
            for dof_i = 1 : FEBE.offsets2[2] # ndofs4item
                FEBE.cvals[1,dof_i,i] = 0.0;
                for j = 1 : FEBE.offsets[2] # edim
                    FEBE.cvals[1,dof_i,i] += FEBE.refbasisderivvals[dof_i + FEBE.offsets2[j],j,i]
                end  
                FEBE.cvals[1,dof_i,i] *= FEBE.coefficients[1,dof_i]/FEBE.iteminfo[1];
            end
        end    
    end  
    return nothing
end



# CURL2D OPERATOR
# HCURL ELEMENTS on 2D domains
# covariant Piola transformation preserves curl2D (up to a factor 1/det(A))
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,Curl2D}, item::Int) where {T <: Real, FEType <: AbstractHcurlFiniteElement, EG <: AbstractElementGeometry2D, AT <:AbstractAssemblyType}
    if FEBE.citem != item
        FEBE.citem = item
        
        # cell update transformation
        update!(FEBE.L2G, item)
        FEBE.coeffs_handler(FEBE.coefficients, item)

        # use Piola transformation on basisvals
        for i = 1 : length(FEBE.xref)
            # evaluate Piola matrix at quadrature point
            if FEBE.L2G.nonlinear || i == 1
                FEBE.iteminfo[1] = piola!(FEBE.L2GM,FEBE.L2G,FEBE.xref[i])
            end
            for dof_i = 1 : FEBE.offsets2[2] # ndofs4item
                FEBE.cvals[1,dof_i,i] = FEBE.refbasisderivvals[dof_i + FEBE.offsets2[1],2,i]
                FEBE.cvals[1,dof_i,i] -= FEBE.refbasisderivvals[dof_i + FEBE.offsets2[2],1,i]
                FEBE.cvals[1,dof_i,i] *= FEBE.coefficients[1,dof_i]/FEBE.iteminfo[1];
            end
        end    
    end  
    return nothing
end


# CURL3D OPERATOR
# HCURL ELEMENTS on 3D domains
# covariant Piola transformation preserves curl3D (up to a factor 1/det(A))
function update!(FEBE::FEBasisEvaluator{T,FEType,EG,Curl3D}, item::Int) where {T <: Real, FEType <: AbstractHcurlFiniteElement, EG <: AbstractElementGeometry3D, AT <:AbstractAssemblyType}
    if FEBE.citem != item
        FEBE.citem = item
        
        # cell update transformation
        update!(FEBE.L2G, item)
        fill!(FEBE.cvals,0.0)
        FEBE.coeffs_handler(FEBE.coefficients, item)

        # use Piola transformation on basisvals
        for i = 1 : length(FEBE.xref)
            # evaluate Piola matrix at quadrature point
            if FEBE.L2G.nonlinear || i == 1
                FEBE.iteminfo[1] = piola!(FEBE.L2GM,FEBE.L2G,FEBE.xref[i])
            end
            for dof_i = 1 : FEBE.offsets2[2] # ndofs4item
                for k = 1 : 3
                    FEBE.cvals[k,dof_i,i] += FEBE.L2GM[k,1] * FEBE.refbasisderivvals[dof_i + FEBE.offsets2[3],2,i] # du3/dx2
                    FEBE.cvals[k,dof_i,i] -= FEBE.L2GM[k,1] * FEBE.refbasisderivvals[dof_i + FEBE.offsets2[2],3,i] # - du2/dx3
                    FEBE.cvals[k,dof_i,i] += FEBE.L2GM[k,2] * FEBE.refbasisderivvals[dof_i + FEBE.offsets2[1],3,i] # du3/dx1
                    FEBE.cvals[k,dof_i,i] -= FEBE.L2GM[k,2] * FEBE.refbasisderivvals[dof_i + FEBE.offsets2[3],1,i] # - du1/dx3
                    FEBE.cvals[k,dof_i,i] += FEBE.L2GM[k,3] * FEBE.refbasisderivvals[dof_i + FEBE.offsets2[2],1,i] # du2/dx1
                    FEBE.cvals[k,dof_i,i] -= FEBE.L2GM[k,3] * FEBE.refbasisderivvals[dof_i + FEBE.offsets2[1],2,i] # - du1/dx2
                    FEBE.cvals[k,dof_i,i] *= FEBE.coefficients[k,dof_i]/FEBE.iteminfo[1];
                end
            end
        end    
    end  
    return nothing
end


# use basisevaluator to evaluate j-th basis function at quadrature point i
function eval!(result, FEBE::FEBasisEvaluator, j::Integer, i; offset::Int = 0, factor = 1)
    for k = 1 : size(FEBE.cvals,1) # resultdim
        result[offset + k] = FEBE.cvals[k,j,i] * factor
    end  
    return nothing
end

# use basisevaluator to evaluate some function at quadrature point i with the given coefficients
function eval!(result, FEBE::FEBasisEvaluator{T,FEType,EG,FEOP}, coefficients::Array{T,1}, i; offset::Int = 0, factor = 1) where {T <: Real, FEType, FEOP, EG}
    for dof_i = 1 : size(FEBE.cvals,2) # ndofs4item
        for k = 1 : size(FEBE.cvals,1) # resultdim
            result[offset+k] += coefficients[dof_i] * FEBE.cvals[k,dof_i,i] * factor 
        end    
    end 
    return nothing
end