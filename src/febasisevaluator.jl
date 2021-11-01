####################
# FEBasisEvaluator #
####################
#
# steers the evaluation of finite element basis functions within an assembly pattern
#
# the reconstruction evaluators are a bit larger as they have additional information for a second reconstruction space
#
# the mutable subtypes maybe used for assembly where the quadrature points change on each item
# (e.g. when integrating over cuts through cells in future)

abstract type AbstractFEBasisEvaluator{T, TvG, TiG} end
abstract type FEBasisEvaluator{T, TvG, TiG, FEType, EG, FEOP, AT, edim, ncomponents, ndofs} <: AbstractFEBasisEvaluator{T, TvG, TiG} end

mutable struct StandardFEBasisEvaluator{T, TvG, TiG, FEType <: AbstractFiniteElement, EG <: AbstractElementGeometry, FEOP <: AbstractFunctionOperator, AT <: AssemblyType,edim,ncomponents, ndofs, ndofs_all,nentries, FType_basis, FType_coeffs <: Function, FType_subset <: Function, FType_jac <: Function} <: FEBasisEvaluator{T, TvG, TiG, FEType, EG, FEOP, AT, edim, ncomponents, ndofs}
    FE::FESpace{TvG,TiG,FEType}                 # link to full FE (e.g. for coefficients)
    L2G::L2GTransformer{TvG, TiG, EG}    # local2global mapper
    L2GAinv::Array{TvG,2}                    # 2nd heap for transformation matrix (e.g. Piola + mapderiv)
    iteminfo::Array{TvG,1}                 # (e.g. current determinant for Hdiv, current tangent)
    xref::Array{Array{T,1},1}            # xref of quadrature formula
    refbasis::FType_basis                # function to call to evaluate basis function on reference geometry
    refbasisvals::Array{Array{T,2},1}    # basis evaluation on EG reference cell 
    refbasisderivvals::Array{T,3}        # additional values to evaluate operator
    derivorder::Int                      # order of derivatives that are needed
    Dresult::Union{Nothing,DiffResults.DiffResult}      # DiffResults for ForwardDiff handling
    Dcfg::Union{Nothing,ForwardDiff.DerivativeConfig, ForwardDiff.JacobianConfig}    # config for ForwardDiff handling
    jacobian_wrap::FType_jac
    offsets::SVector{ncomponents,Int}    # offsets for gradient entries of each dof
    offsets2::Array{Int,1}               # offsets for dof entries of each gradient (on ref)
    citem::Int                           # current item
    cvals::Array{T,3}                    # current operator vals on item
    coefficients::Array{T,2}             # coefficients for finite element
    coefficients3::Array{T,2}            # coefficients for operator (e.g. TangentialGradient)
    coeffs_handler::FType_coeffs         # function to call to get coefficients for finite element
    subset_handler::FType_subset         # function to call to get linear independent subset of basis on cell
    current_subset::Array{Int,1}         # current indices of subset of linear independent basis functions
    compressiontargets::Array{Int,1}     # some operators allow for compressed storage (e.g. SymmetricGradient)
end

mutable struct ReconstructionFEBasisEvaluator{T, TvG, TiG, FEType <: AbstractFiniteElement, FETypeR <: AbstractFiniteElement, EG <: AbstractElementGeometry, FEOP <: AbstractFunctionOperator, AT <: AssemblyType,edim,ncomponents, ndofs, ndofs2, ndofs_all,nentries, FType_basis, FType_coeffs <: Function, FType_subset <: Function, FType_jac <: Function} <: FEBasisEvaluator{T, TvG, TiG, FEType, EG, FEOP, AT, edim, ncomponents, ndofs}
    FE::FESpace{TvG,TiG,FEType}          # link to full FE (e.g. for coefficients)
    FE2::FESpace{TvG,TiG,FETypeR}        # link to reconstruction FE
    L2G::L2GTransformer{TvG, TiG, EG}    # local2global mapper
    L2GAinv::Array{TvG,2}                    # 2nd heap for transformation matrix (e.g. Piola + mapderiv)
    iteminfo::Array{TvG,1}                 # (e.g. current determinant for Hdiv, current tangent)
    xref::Array{Array{T,1},1}            # xref of quadrature formula
    refbasis::FType_basis                # function to call to evaluate basis function on reference geometry
    refbasisvals::Array{Array{T,2},1}    # basis evaluation on EG reference cell 
    refbasisderivvals::Array{T,3}        # additional values to evaluate operator
    derivorder::Int                      # order of derivatives that are needed
    Dresult::Union{Nothing,DiffResults.DiffResult}      # DiffResults for ForwardDiff handling
    Dcfg::Union{Nothing,ForwardDiff.DerivativeConfig, ForwardDiff.JacobianConfig}    # config for ForwardDiff handling
    jacobian_wrap::FType_jac
    offsets::SVector{ncomponents,Int}    # offsets for gradient entries of each dof
    offsets2::Array{Int,1}               # offsets for dof entries of each gradient (on ref)
    citem::Int                           # current item
    cvals::Array{T,3}                    # current operator vals on item
    coefficients::Array{T,2}             # coefficients for finite element
    coefficients2::Array{T,2}            # coefficients for reconstruction
    coefficients3::Array{T,2}            # coefficients for operator (e.g. TangentialGradient)
    coeffs_handler::FType_coeffs         # function to call to get coefficients for finite element
    reconst_handler::ReconstructionHandler{T, Int32, FEType,FETypeR,AT,EG} # hanlder for reconstruction coefficients
    subset_handler::FType_subset         # function to call to get linear independent subset of basis on cell
    current_subset::Array{Int,1}         # current indices of subset of linear independent basis functions
    compressiontargets::Array{Int,1}     # some operators allow for compressed storage (e.g. SymmetricGradient)
end

function prepareFEBasisDerivs!(refbasisderivvals, refbasis, xref, derivorder, ndofs4item_all, ncomponents; Dcfg = "init", Dresult = "init", jac_wrap = NothingFunction)

    # derivatives of the basis on the reference domain are computed
    # by ForwardDiff, to minimise memory allocations and be able
    # to rebase the quadrature points later (e.g. when evaluating cuts through cells)
    # we use DiffResults and JacobianConfig of ForwardDiff and save these in the struct

    if Dcfg == "init" || Dresult == "init" || jac_wrap == NothingFunction

        edim = size(refbasisderivvals,2)
        result_temp = zeros(Float64,ndofs4item_all,ncomponents)
        result_temp2 = zeros(Real,ndofs4item_all,ncomponents)
        input_temp = Vector{Float64}(undef,edim)
        jac_temp = Matrix{Float64}(undef,ndofs4item_all*ncomponents,edim)
        Dresult = DiffResults.DiffResult(result_temp,jac_temp)
        Dcfg = ForwardDiff.JacobianConfig(refbasis, result_temp, input_temp)

        function jacobian_wrap(x)
            fill!(result_temp,0.0)
            ForwardDiff.vector_mode_jacobian!(Dresult, refbasis, result_temp, x, Dcfg)
            return DiffResults.jacobian(Dresult)
        end
        jac_wrap = jacobian_wrap
        if derivorder == 1 ## first order derivatives
            jac::Array{Float64,2} = DiffResults.jacobian(Dresult)
        elseif derivorder == 2
            # todo: use DiffResults for hessian evaluation 
            function refbasis_wrap(xref)
                fill!(result_temp2,0)
                refbasis(result_temp2,xref)
                return result_temp2
            end

            jac_function = x -> ForwardDiff.jacobian(refbasis_wrap, x)
            function hessian_wrap(xref)
                return ForwardDiff.jacobian(jac_function, xref)
            end
        end
    end

    if derivorder == 1 ## first order derivatives

        for i = 1 : length(xref)
            # evaluate gradients of basis function
            # = list of vectors [du_k/dx_1; du_k,dx_2]

            jac = jac_wrap(xref[i])

            for j = 1 : ndofs4item_all*ncomponents, k = 1 : edim
                refbasisderivvals[j,k,i] = jac[j,k];
            end
        end
    elseif derivorder == 2 # second order derivatives
        for i = 1 : length(xref)
            # evaluate second derivatives of basis function
            refbasisderivvals[:,:,i] = hessian_wrap(xref[i])
        end 
    end


    return Dresult, Dcfg, jac_wrap
end

## relocates evaluation points (needs mutable FEB) used by segment integrator/point evaluator
function relocate_xref!(FEB::FEBasisEvaluator{T,TvG,TiG,FEType,EG,FEOP,AT}, new_xref) where {T, TvG, TiG, FEType, EG, FEOP, AT}
    for j = 1 : length(FEB.xref[1])
        FEB.xref[1][j] = new_xref[j]
    end
    if FEB.derivorder == 0
        # evaluate basis functions at quadrature point
        FEB.refbasis(FEB.refbasisvals[1], new_xref)
        if FEType <: AbstractH1FiniteElement # also reset cvals for operator evals that stay the same for all cells
            if FEOP <: Identity || FEOP <: IdentityDisc
                for j = 1 : size(FEB.refbasisvals[1],1), k = 1 : size(FEB.refbasisvals[1],2)
                FEB.cvals[k,j,1] = FEB.refbasisvals[1][j,k]
                end
            elseif FEOP <: IdentityComponent
                for j = 1 : size(FEB.refbasisvals[1],1)
                FEB.cvals[1,j,1] = FEB.refbasisvals[1][j,FEOP.parameters[1]]
                end
            end
        end
    elseif FEB.derivorder > 0
        prepareFEBasisDerivs!(FEB.refbasisderivvals, FEB.refbasis, FEB.xref, FEB.derivorder, size(FEB.refbasisvals[1],1), length(FEB.offsets); Dcfg = FEB.Dcfg, Dresult = FEB.Dresult)
    end
    FEB.citem = 0 # reset citem to allows recomputation if FEB.cvals (if multiple consecutive relocations are done in the same cell)
    return nothing
end

function FEBasisEvaluator{T,EG,FEOP,AT}(FE::FESpace{TvG,TiG,FEType,FEAPT}, xref::Array{Array{T,1},1}) where {T, TvG, TiG, FEType <: AbstractFiniteElement, EG <: AbstractElementGeometry, FEOP <: AbstractFunctionOperator, AT <: AssemblyType, FEAPT <: AssemblyType}
    xgrid = FE.xgrid
    L2G = L2GTransformer(EG, xgrid,AT)
    L2GAinv = copy(L2G.A)

    # get effective assembly type for basis
    # depending on the AT and the AT of the FESpace
    FEAT = EffAT4AssemblyType(FEAPT,AT)

    @debug "Creating FEBasisEvaluator for $FEType, EG = $EG, operator = $FEOP, FEAT = $FEAT, AT =$AT"

    # collect basis function information
    ncomponents::Int = get_ncomponents(FEType)
    ndofs4item::Int = 0
    refbasis = get_basis(FEAT, FEType, EG)
    ndofs4item = get_ndofs(FEAT, FEType, EG)
    ndofs4item_all = get_ndofs_all(FEAT, FEType, EG)
    if AT <: Union{ON_BFACES,<:ON_FACES,<:ON_EDGES,ON_BEDGES}
        if FEType <: AbstractHdivFiniteElement ||  FEType <: AbstractHcurlFiniteElement
            ncomponents = 1
        end
    end    

    # evaluate basis on reference domain
    refbasisvals = Array{Array{T,2},1}(undef,length(xref));
    for i = 1 : length(xref)
        refbasisvals[i] = zeros(T,ndofs4item_all,ncomponents)
    end    

    # set coefficient handlers needed for basis evaluation
    coefficients = zeros(T,0,0)
    coeff_handler = NothingFunction
    if FEType <: Union{AbstractH1FiniteElementWithCoefficients, AbstractHdivFiniteElement, AbstractHcurlFiniteElement}
        coefficients = zeros(T,ncomponents,ndofs4item)
        coeff_handler = get_coefficients(FEAT, FE, EG)
    end    

    # set subset handler (only relevant if ndofs4item_all > ndofs4item)
    subset_handler = get_basissubset(FEAT, FE, EG)

    # compute refbasisderivvals and further coefficients needed for operator eval
    derivorder = NeededDerivative4Operator(FEOP)
    edim = max(1,dim_element(EG))
    xdim = size(FE.xgrid[Coordinates],1)
    resultdim = Int(Length4Operator(FEOP,edim,ncomponents))
    coefficients3 = zeros(T,0,0)
    offsets = 0:edim:((ncomponents-1)*edim); # edim steps
    offsets2 = []
    if ndofs4item_all > 0
        offsets2 = 0:ndofs4item_all:ncomponents*ndofs4item_all
    else
        @warn "ndofs = 0 for FEType = $FEType on EG = $EG"
    end
    compressiontargets = zeros(T,0)
    current_eval = zeros(T,resultdim,ndofs4item,length(xref))
    if derivorder == 0

        for i = 1 : length(xref)
            # evaluate basis functions at quadrature point
            refbasis(refbasisvals[i], xref[i])
        end    

        refbasisderivvals = zeros(T,0,0,0);
        if FEOP == NormalFlux || (FEOP == TangentFlux && edim == 2)
            coefficients3 = FE.xgrid[FaceNormals]
        elseif FEOP == TangentFlux && edim == 3
            coefficients3 = FE.xgrid[EdgeTangents]
        elseif FEOP <: Identity || FEOP <: IdentityDisc
            for i = 1 : length(xref), j = 1 : ndofs4item, k = 1 : ncomponents
                current_eval[k,j,i] = refbasisvals[i][j,k]
            end
        elseif FEOP <: IdentityComponent
            for i = 1 : length(xref), j = 1 : ndofs4item
                current_eval[1,j,i] = refbasisvals[i][j,FEOP.parameters[1]]
            end
        end
        Dcfg = nothing
        Dresult = nothing
        jacobian_wrap = NothingFunction
    elseif derivorder > 0
        # get derivatives of basis on reference geometry
        if derivorder == 1
            refbasisderivvals = zeros(T,ndofs4item_all*ncomponents,edim,length(xref));
        elseif derivorder == 2
            refbasisderivvals = zeros(T,ndofs4item_all*ncomponents*edim,edim,length(xref))
            offsets2 = 0:edim*edim:ncomponents*edim*edim;
        end
        Dresult, Dcfg, jacobian_wrap = prepareFEBasisDerivs!(refbasisderivvals, refbasis, xref, derivorder, ndofs4item_all, ncomponents)

        # specifications for special operators (e.g. compressing Voigt notation)
        if FEOP == TangentialGradient
            coefficients3 = FE.xgrid[FaceNormals]
        elseif FEOP <: SymmetricGradient
            # the following mapping tells where each entry of the full gradient lands in the reduced vector
            @assert ncomponents == edim "SymmetricGradient requires a square matrix as gradient (ncomponents == dim is needed)"
            @assert edim > 1 "SymmetricGradient makes only sense for dim > 1, please use Gradient"
            if edim == 2
                # 2D Voigt accumulation positions of du1/dx1, du1/dx2, du2/dx1, d2/dx2
                compressiontargets = [1,3,3,2]
            elseif edim == 3
                # 3D Voigt accumulation positions of du1/dx1, du1/dx2, du1/dx3, d2/dx1,...
                compressiontargets = [1,6,5,6,2,4,5,4,3] 
            end
        elseif FEOP <: SymmetricHessian
            cl::Int = 0
            if edim == 1
                compressiontargets = [1,1]
                cl = 1
            elseif edim == 2
                # 2D Voigt accumulation positions of du1/dx1, du1/dx2, du2/dx1, d2/dx2
                compressiontargets = [1,3,3,2]
                cl = 3
            elseif edim == 3
                # 3D Voigt accumulation positions of du1/dx1, du1/dx2, du1/dx3, d2/dx1,...
                compressiontargets = [1,6,5,6,2,4,5,4,3] 
                cl = 6
            end
            for n = 1 : ncomponents
                append!(compressiontargets,compressiontargets .+ cl)
            end
        end
    end

    StandardFEBasisEvaluator{T,TvG,TiG,FEType,EG,FEOP,AT,edim,ncomponents,ndofs4item,ndofs4item_all,ndofs4item_all*ncomponents,typeof(refbasis),typeof(coeff_handler),typeof(subset_handler),typeof(jacobian_wrap)}(FE,L2G,L2GAinv,zeros(T,xdim+1),xref,refbasis,refbasisvals,refbasisderivvals,derivorder,Dresult,Dcfg,jacobian_wrap,offsets,offsets2,0,current_eval,coefficients, coefficients3, coeff_handler, subset_handler, 1:ndofs4item, compressiontargets)
end    

# constructor for ReconstructionIdentity, ReconstructionDivergence, ReconstructionGradient
function FEBasisEvaluator{T,EG,FEOP,AT}(FE::FESpace{TvG,TiG,FEType,FEAPT}, xref::Array{Array{T,1},1}) where {T, TvG, TiG, FEType <: AbstractFiniteElement, FETypeReconst <: AbstractFiniteElement, EG <: AbstractElementGeometry, FEOP <: Union{<:ReconstructionIdentity{FETypeReconst},ReconstructionNormalFlux{FETypeReconst},ReconstructionDivergence{FETypeReconst},<:ReconstructionGradient{FETypeReconst}}, AT <: AssemblyType, FEAPT <: AssemblyType}
    
    @debug "Creating FEBasisEvaluator for $FEOP operator of $FEType on $EG"

    # generate reconstruction space
    # avoid computation of full dofmap
    # we will just use local basis functions
    xgrid = FE.xgrid
    FE2 = FESpace{FETypeReconst}(xgrid)
    L2G = L2GTransformer(EG, xgrid,AT)
    L2GAinv = copy(L2G.A)

    # collect basis function information
    ncomponents::Int = get_ncomponents(FEType)
    ncomponents2::Int = get_ncomponents(FETypeReconst)
    if AT <: Union{ON_BFACES,<:ON_FACES}
        if FETypeReconst <: AbstractHdivFiniteElement
            ncomponents2 = 1
        end
    end
    refbasis = get_basis(AT, FEType, EG)
    refbasis_reconst = get_basis(AT, FETypeReconst, EG)
    ndofs4item = get_ndofs(AT, FEType, EG)
    ndofs4item2 = get_ndofs(AT, FETypeReconst, EG)
    ndofs4item2_all = get_ndofs_all(AT, FETypeReconst, EG)    

    # evaluate reconstruction basis
    refbasisvals = Array{Array{T,2},1}(undef,length(xref));
    for i = 1 : length(xref)
        # evaluate basis functions at quadrature point
        refbasisvals[i] = zeros(T,ndofs4item2_all,ncomponents2)
        refbasis_reconst(refbasisvals[i], xref[i])
    end    

    # set coefficient handlers needed for basis evaluation (for both FEType and FETypereconst)
    if FETypeReconst <: Union{AbstractH1FiniteElementWithCoefficients, AbstractHdivFiniteElement}
        coefficients = zeros(T,ncomponents,ndofs4item2)
        coeff_handler = get_coefficients(AT, FE2, EG)
    else
        coefficients = zeros(T,0,0)
        coeff_handler = NothingFunction
    end    
    coefficients2 = zeros(T,ndofs4item,ndofs4item2)

    # set subset handler (only relevant if ndofs4item_all > ndofs4item)
    subset_handler = get_basissubset(AT, FE2, EG)

    # get reconstruction coefficient handlers
    reconst_handler = ReconstructionHandler(FE,FE2,AT,EG)

    # compute refbasisderivvals and further coefficients needed for operator eval
    edim = dim_element(EG)
    xdim = size(FE.xgrid[Coordinates],1)
    offsets = 0:edim:((ncomponents-1)*edim);
    offsets2 = 0:ndofs4item2_all:ncomponents*ndofs4item2_all;
    coefficients3 = zeros(T,0,0)
    resultdim = Int(Length4Operator(FEOP,edim,ncomponents))
    current_eval = zeros(T,resultdim,ndofs4item,length(xref))
    derivorder = NeededDerivative4Operator(FEOP)
    if derivorder == 0
        # refbasisderivvals are used as a cache for the reconstruction basis
        refbasisderivvals = zeros(T,ncomponents,ndofs4item2_all,length(xref));
        Dcfg = nothing
        Dresult = nothing
        jacobian_wrap = NothingFunction
    elseif derivorder == 1
        # derivatives of the reconstruction basis on the reference domain are computed
        refbasisderivvals = zeros(T,ndofs4item2_all*ncomponents2,edim,length(xref));
        Dresult, Dcfg, jacobian_wrap = prepareFEBasisDerivs!(refbasisderivvals, refbasis_reconst, xref, derivorder, ndofs4item2_all, ncomponents2)
        coefficients3 = zeros(T,resultdim,ndofs4item2)
    end
    
    return ReconstructionFEBasisEvaluator{T,TvG,TiG,FEType,FETypeReconst,EG,FEOP,AT,edim,ncomponents,ndofs4item,ndofs4item2,ndofs4item2_all,ndofs4item2_all*ncomponents2,typeof(refbasis),typeof(coeff_handler),typeof(subset_handler),typeof(jacobian_wrap)}(FE,FE2,L2G,L2GAinv,zeros(T,xdim+1),xref,refbasis,refbasisvals,refbasisderivvals,derivorder,Dresult,Dcfg,jacobian_wrap,offsets,offsets2,0,current_eval,coefficients, coefficients2,coefficients3,coeff_handler, reconst_handler, subset_handler,1:ndofs4item2,[])
end    

"""
````
FEBasisEvaluator{T,EG,FEOP,AT}(FE::FESpace, qf::QuadratureRule; mutable = false) where {T <: Real, EG <: AbstractElementGeometry, FEOP <: AbstractFunctionOperator, AT <: AssemblyType}
````

Constructor for an evaluator for the basis of the FESpace FES evaluated with the function operator FEOP on the given element geometry EG with AssemblyType AT
at the points of the quadrature rule qf.

"""
function FEBasisEvaluator{T,EG,FEOP,AT}(FE::FESpace{TvG,TiG,FEType,FEAT}, qf::QuadratureRule) where {T, TvG, TiG, FEType, EG, FEOP, AT, FEAT}
    FEBasisEvaluator{T,EG,FEOP,AT}(FE, Array{Array{T,1},1}(qf.xref))
end


# IDENTITY OPERATOR
# H1 ELEMENTS (nothing has to be done)
"""
````
    update_febe!(FEBE::FEBasisEvaluator, item::Int)
````

Update the FEBasisEvaluator on the given item number of the grid items associated to the AssemblyType. During the update the FEBasisevaluator computes all evaluations of all basis functions at all quadrature points and stores
them in FEBE.cvals. From there they can be accessed directly or via the eval_febe! functions.

"""
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractH1FiniteElement,<:AbstractElementGeometry,<:Identity,<:AssemblyType,edim,ncomponents,ndofs}, item) where {T,TvG,TiG,edim,ncomponents,ndofs}
    if FEBE.citem[] != item
        FEBE.citem = item
        # needs only to be updated if basis sets can change (e.g. for P3 in 2D/3D)
        if FEBE.subset_handler != NothingFunction
            fill!(FEBE.cvals,0.0)
            FEBE.subset_handler(FEBE.current_subset, item)
            for i = 1 : length(FEBE.xref)
                for dof_i = 1 : ndofs, k = 1 : ncomponents
                    FEBE.cvals[k,dof_i,i] += FEBE.refbasisvals[i][FEBE.current_subset[dof_i],k]
                end
            end
        end
    end
    return nothing
end


# IDENTITYCOMPONENT OPERATOR
# H1 ELEMENTS (nothing has to be done)
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractH1FiniteElement,<:AbstractElementGeometry,<:IdentityComponent,<:AssemblyType}, item) where {T,TvG,TiG}
    FEBE.citem = item
    return nothing
end

# RECONSTRUCTION IDENTITY OPERATOR
# H1 ELEMENTS
# HDIV RECONSTRUCTION
# Piola transform Hdiv reference basis and multiply Hdiv coefficients and Trafo coefficients
function update_febe!(FEBE::ReconstructionFEBasisEvaluator{T,TvG,TiG,<:AbstractH1FiniteElement,<:AbstractHdivFiniteElement,<:AbstractElementGeometry,<:ReconstructionIdentity,<:AssemblyType,edim,ncomponents,ndofs,ndofs2}, item) where {T,TvG,TiG,edim,ncomponents,ndofs,ndofs2}
    if FEBE.citem[] != item
        FEBE.citem = item
    
        # update transformation
        update_trafo!(FEBE.L2G, item)
        L2GM::Array{T,2} = FEBE.L2G.A # need matrix for Piola trafo
        if FEBE.L2G.nonlinear
            @error "nonlinear local2global transformations not yet supported"
        end
        coeffs::Array{T,2} = FEBE.coefficients
        subset::Array{Int,1} = FEBE.current_subset
        FEBE.coeffs_handler(coeffs, item)
        FEBE.subset_handler(subset, item)

        # use Piola transformation on Hdiv basis
        # and save it in refbasisderivvals
        tempeval::Array{T,3} = FEBE.refbasisderivvals
        fill!(tempeval,0)
        det = FEBE.L2G.det # 1 alloc
        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs2
                for k = 1 : ncomponents
                    for l = 1 : ncomponents
                        tempeval[k,dof_i,i] += L2GM[k,l]*FEBE.refbasisvals[i][subset[dof_i],l];
                    end    
                    tempeval[k,dof_i,i] *= coeffs[k,dof_i] / det
                end
            end
        end

        # get local reconstruction coefficients
        # and accumulate
        rcoeffs::Array{T,2} = FEBE.coefficients2
        #FEBE.reconstcoeffs_handler(rcoeffs, item)
        #@show item rcoeffs; fill!(rcoeffs,0)
        get_rcoefficients!(rcoeffs, FEBE.reconst_handler, item)
        #@show rcoeffs

        fill!(FEBE.cvals,0.0)
        for dof_i = 1 : ndofs, dof_j = 1 : ndofs2
            if rcoeffs[dof_i,dof_j] != 0
                for i = 1 : length(FEBE.xref)
                    for k = 1 : ncomponents
                        FEBE.cvals[k,dof_i,i] += rcoeffs[dof_i,dof_j] * tempeval[k,dof_j,i]; 
                    end
                end
            end
        end
    end
    return nothing
end


# RECONSTRUCTION NORMALFLUX OPERATOR
# Hdiv ELEMENTS (just divide by face volume)
function update_febe!(FEBE::ReconstructionFEBasisEvaluator{T,TvG,TiG,<:AbstractH1FiniteElement,<:AbstractHdivFiniteElement,<:AbstractElementGeometry,<:ReconstructionNormalFlux,<:ON_FACES,edim,ncomponents,ndofs,ndofs2}, item) where {T,TvG,TiG,edim,ncomponents,ndofs,ndofs2}
    if FEBE.citem[] != item
        FEBE.citem = item

        # get local reconstruction coefficients
        # and accumulate
        get_rcoefficients!(FEBE.coefficients2, FEBE.reconst_handler, item)

        fill!(FEBE.cvals,0.0)
        xItemVolumes::Array{T,1} = FEBE.L2G.ItemVolumes
        for dof_i = 1 : ndofs, dof_j = 1 : ndofs2
            if FEBE.coefficients2[dof_i,dof_j] != 0
                for i = 1 : length(FEBE.xref)
                    for k = 1 : ncomponents
                        FEBE.cvals[k,dof_i,i] += FEBE.coefficients2[dof_i,dof_j] * FEBE.refbasisvals[i][dof_j,k] / xItemVolumes[item]
                    end
                end
            end
        end
    end
    return nothing
end


# RECONSTRUCTION NORMALFLUX OPERATOR
# Hdiv ELEMENTS (just divide by face volume)
function update_febe!(FEBE::ReconstructionFEBasisEvaluator{T,TvG,TiG,<:AbstractH1FiniteElement,<:AbstractElementGeometry,<:ReconstructionNormalFlux,<:ON_BFACES,edim,ncomponents,ndofs,ndofs2}, item) where {T,TvG,TiG,edim,ncomponents,ndofs,ndofs2}
    if FEBE.citem[] != item
        FEBE.citem = item

        # get local reconstruction coefficients
        # and accumulate
        get_rcoefficients!(FEBE.coefficients2, FEBE.reconst_handler, FEBE.FE.xgrid[BFaceFaces][item])

        fill!(FEBE.cvals,0.0)
        xItemVolumes::Array{T,1} = FEBE.L2G.ItemVolumes
        for dof_i = 1 : ndofs, dof_j = 1 : ndofs2
            if FEBE.coefficients2[dof_i,dof_j] != 0
                for i = 1 : length(FEBE.xref)
                    for k = 1 : ncomponents
                        FEBE.cvals[k,dof_i,i] += FEBE.coefficients2[dof_i,dof_j] *  FEBE.refbasisvals[i][dof_j,k] / xItemVolumes[item]
                    end
                end
            end
        end
    end
    return nothing
end


# IDENTITY OPERATOR
# Hcurl ELEMENTS (covariant Piola trafo)
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractHcurlFiniteElement,<:AbstractElementGeometry,<:Identity,<:AssemblyType,edim,ncomponents,ndofs}, item) where {T,TvG,TiG,edim,ncomponents,ndofs}
    if FEBE.citem[] != item
        FEBE.citem = item
    
        # update transformation
        update_trafo!(FEBE.L2G, item)
        if !FEBE.L2G.nonlinear
            mapderiv!(FEBE.L2GAinv,FEBE.L2G,nothing)
        else
            @error "nonlinear local2global transformations not yet supported"
        end
        FEBE.coeffs_handler(FEBE.coefficients, item)

        # use Piola transformation on basisvals
        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs
                for k = 1 : ncomponents
                    FEBE.cvals[k,dof_i,i] = 0.0;
                    for l = 1 : ncomponents
                        FEBE.cvals[k,dof_i,i] += FEBE.L2GAinv[k,l]*FEBE.refbasisvals[i][dof_i,l];
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
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractHdivFiniteElement,<:AbstractElementGeometry,<:Union{Identity,<:IdentityDisc{Jump}},<:AssemblyType,edim,ncomponents,ndofs}, item) where {T,TvG,TiG,edim,ncomponents,ndofs}
    if FEBE.citem[] != item
        FEBE.citem = item
    
        # update transformation
        update_trafo!(FEBE.L2G, item)
        L2GM::Array{T,2} = FEBE.L2G.A # need matrix for Piola trafo
        if FEBE.L2G.nonlinear
            @error "nonlinear local2global transformations not yet supported"
        end
        FEBE.coeffs_handler(FEBE.coefficients, item)
        FEBE.subset_handler(FEBE.current_subset, item)

        # use Piola transformation on basisvals
        det = FEBE.L2G.det # 1 alloc
        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs
                for k = 1 : ncomponents
                    FEBE.cvals[k,dof_i,i] = 0.0;
                    for l = 1 : ncomponents
                        FEBE.cvals[k,dof_i,i] += L2GM[k,l]*FEBE.refbasisvals[i][FEBE.current_subset[dof_i],l];
                    end    
                    FEBE.cvals[k,dof_i,i] *= FEBE.coefficients[k,dof_i] / det
                end
            end
        end
    end
    return nothing
end



# IDENTITY OPERATOR
# H1 ELEMENTS WITH COEFFICIENTS
# (no transformation needed, just multiply coefficients)
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractH1FiniteElementWithCoefficients,<:AbstractElementGeometry,<:Identity,<:AssemblyType,edim,ncomponents,ndofs}, item) where {T,TvG,TiG,edim,ncomponents,ndofs}
    if FEBE.citem[] != item
        FEBE.citem = item
        
        # get coefficients
        FEBE.coeffs_handler(FEBE.coefficients, item)

        if FEBE.subset_handler != NothingFunction
            FEBE.subset_handler(FEBE.current_subset, item)
        end

        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs
                for k = 1 : ncomponents
                    FEBE.cvals[k,dof_i,i] = FEBE.refbasisvals[i][FEBE.current_subset[dof_i],k] * FEBE.coefficients[k,dof_i];
                end    
            end
        end
    end
    return nothing
end


# IDENTITYCOMPONENT OPERATOR
# H1 ELEMENTS WITH COEFFICIENTS
# (no transformation needed, just multiply coefficients)
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractH1FiniteElementWithCoefficients,<:AbstractElementGeometry,FEOP,<:AssemblyType,edim,ncomponents,ndofs}, item) where {T,TvG,TiG,FEOP<:IdentityComponent,edim,ncomponents,ndofs}
    if FEBE.citem[] != item
        FEBE.citem = item
        
        # get coefficients
        FEBE.coeffs_handler(FEBE.coefficients, item)

        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs
                FEBE.cvals[1,dof_i,i] = FEBE.refbasisvals[i][dof_i,FEOP.parameters[1]] * FEBE.coefficients[FEOP.parameters[1],dof_i];
            end
        end
    end
    return nothing
end


# IDENTITYCOMPONENT OPERATOR
# Hdiv ELEMENTS (Piola trafo)
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractHdivFiniteElement,<:AbstractElementGeometry,FEOP,<:AssemblyType,edim,ncomponents,ndofs}, item) where {T,TvG,TiG,FEOP<:IdentityComponent,edim,ncomponents,ndofs}
    if FEBE.citem[] != item
        FEBE.citem = item
    
        # update transformation
        update_trafo!(FEBE.L2G, item)
        L2GM::Array{T,2} = FEBE.L2G.A # need matrix for Piola trafo
        if FEBE.L2G.nonlinear
            @error "nonlinear local2global transformations not yet supported"
        end
        FEBE.coeffs_handler(FEBE.coefficients, item)
        FEBE.subset_handler(FEBE.current_subset, item)

        # use Piola transformation on basisvals
        det = FEBE.L2G.det # 1 alloc
        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs
                FEBE.cvals[1,dof_i,i] = 0.0;
                for l = 1 : ncomponents
                    FEBE.cvals[1,dof_i,i] += L2GM[FEOP.parameters[1],l]*FEBE.refbasisvals[i][FEBE.current_subset[dof_i],l];
                end    
                FEBE.cvals[1,dof_i,i] *= FEBE.coefficients[FEOP.parameters[1],dof_i] / det
            end
        end
    end
    return nothing
end



# NORMALFLUX OPERATOR
# H1 ELEMENTS
# ON_FACES
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractH1FiniteElement,<:AbstractElementGeometry,<:NormalFlux,<:ON_FACES,edim,ncomponents,ndofs}, item) where {T,TvG,TiG,edim,ncomponents,ndofs}
    if FEBE.citem[] != item
        FEBE.citem = item

        # fetch normal of item
        for k = 1 : ncomponents # ncomponents of normal
            FEBE.iteminfo[k] = FEBE.coefficients3[k,item]
        end

        if FEBE.subset_handler != NothingFunction
            FEBE.subset_handler(FEBE.current_subset, item)
        end

        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs
                FEBE.cvals[1,dof_i,i] = 0.0
                for k = 1 : ncomponents
                    FEBE.cvals[1,dof_i,i] += FEBE.refbasisvals[i][FEBE.current_subset[dof_i],k] * FEBE.iteminfo[k];
                end    
            end
        end
    end
    return nothing
end


# NORMALFLUX OPERATOR
# H1 ELEMENTS
# ON_BFACES
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractH1FiniteElement,<:AbstractElementGeometry,<:NormalFlux,<:ON_BFACES,edim,ncomponents,ndofs}, item) where {T,TvG,TiG,edim,ncomponents,ndofs}
    if FEBE.citem[] != item
        FEBE.citem = item

        # fetch normal of item
        for k = 1 : ncomponents
            FEBE.iteminfo[k] = FEBE.coefficients3[k,FEBE.FE.xgrid[BFaceFaces][item]]
        end

        if FEBE.subset_handler != NothingFunction
            FEBE.subset_handler(FEBE.current_subset, item)
        end

        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs
                FEBE.cvals[1,dof_i,i] = 0.0
                for k = 1 : ncomponents
                    FEBE.cvals[1,dof_i,i] += FEBE.refbasisvals[i][FEBE.current_subset[dof_i],k] * FEBE.iteminfo[k];
                end    
            end
        end
    end
    return nothing
end

# NORMALFLUX OPERATOR
# H1 ELEMENTS WITH COEFFICIENTS
# ON_FACES
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractH1FiniteElementWithCoefficients,<:AbstractElementGeometry,<:NormalFlux,<:ON_FACES,edim,ncomponents,ndofs}, item) where {T,TvG,TiG,edim,ncomponents,ndofs}
    if FEBE.citem[] != item
        FEBE.citem = item

        # fetch normal of item
        for k = 1 : ncomponents
            FEBE.iteminfo[k] = FEBE.coefficients3[k,item]
        end
        
        # get coefficients
        FEBE.coeffs_handler(FEBE.coefficients, item)

        if FEBE.subset_handler != NothingFunction
            FEBE.subset_handler(FEBE.current_subset, item)
        end

        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs
                FEBE.cvals[1,dof_i,i] = 0.0
                for k = 1 : ncomponents
                    FEBE.cvals[1,dof_i,i] += FEBE.refbasisvals[i][FEBE.current_subset[dof_i],k] * FEBE.coefficients[k,dof_i] * FEBE.iteminfo[k];
                end    
            end
        end
    end
    return nothing
end


# NORMALFLUX OPERATOR
# H1 ELEMENTS WITH COEFFICIENTS
# ON_BFACES
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractH1FiniteElementWithCoefficients,<:AbstractElementGeometry,NormalFlux,<:ON_BFACES,edim,ncomponents,ndofs}, item) where {T,TvG,TiG,edim,ncomponents,ndofs}
    if FEBE.citem[] != item
        FEBE.citem = item

        # fetch normal of item
        for k = 1 : ncomponents
            FEBE.iteminfo[k] = FEBE.coefficients3[k,FEBE.FE.xgrid[BFaceFaces][item]]
        end
        
        # get coefficients
        FEBE.coeffs_handler(FEBE.coefficients, item)

        if FEBE.subset_handler != NothingFunction
            FEBE.subset_handler(FEBE.current_subset, item)
        end

        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs
                FEBE.cvals[1,dof_i,i] = 0.0
                for k = 1 : ncomponents
                    FEBE.cvals[1,dof_i,i] += FEBE.refbasisvals[i][FEBE.current_subset[dof_i],k] * FEBE.coefficients[k,dof_i] * FEBE.iteminfo[k];
                end    
            end
        end
    end
    return nothing
end

# NORMALFLUX OPERATOR
# Hdiv ELEMENTS (just divide by face volume)
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractHdivFiniteElement,<:AbstractElementGeometry,<:NormalFlux,<:AssemblyType,edim,ncomponents,ndofs}, item) where {T,TvG,TiG,edim,ncomponents,ndofs}
    if FEBE.citem[] != item
        FEBE.citem = item
    
        # use Piola transformation on basisvals
        xItemVolumes::Array{T,1} = FEBE.L2G.ItemVolumes
        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs, k = 1 : ncomponents
                FEBE.cvals[k,dof_i,i] = FEBE.refbasisvals[i][dof_i,k] / xItemVolumes[item]
            end
        end
    end
    return nothing
end


# TANGENTLFLUX OPERATOR
# Hcurl ELEMENTS (just divide by face volume)
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractHcurlFiniteElement,<:AbstractElementGeometry,<:TangentFlux,<:AssemblyType,edim,ncomponents,ndofs}, item) where {T,TvG,TiG,edim,ncomponents,ndofs}
    if FEBE.citem[] != item
        FEBE.citem = item
    
        # use Piola transformation on basisvals
        xItemVolumes::Array{T,1} = FEBE.L2G.ItemVolumes
        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs, k = 1 : ncomponents
                FEBE.cvals[k,dof_i,i] = FEBE.refbasisvals[i][dof_i,k] / xItemVolumes[item]
            end
        end
    end
    return nothing
end


# HESSIAN OPERATOR
# H1 ELEMENTS
# multiply tinverted jacobian of element trafo with gradient of basis function
# which yields (by chain rule) the gradient in x coordinates
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractH1FiniteElement,<:AbstractElementGeometry,<:Hessian,<:AssemblyType,edim,ncomponents,ndofs}, item) where {T,TvG,TiG,edim,ncomponents,ndofs}
    if FEBE.citem[] != item
        FEBE.citem = item

        # update transformation
        update_trafo!(FEBE.L2G, item)
        if !FEBE.L2G.nonlinear
            mapderiv!(FEBE.L2GAinv,FEBE.L2G,nothing)
        else
            @error "nonlinear local2global transformations not yet supported"
        end

        if FEBE.subset_handler != NothingFunction
            FEBE.subset_handler(FEBE.current_subset, item)
        end

        fill!(FEBE.cvals,0)
        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs
                for c = 1 : ncomponents
                    for k = 1 : edim, l = 1 : edim
                        # second derivatives partial^2 (x_k x_l)
                        for xi = 1 : edim, xj = 1 : edim
                            FEBE.cvals[(c-1)*edim^2 + (k-1)*edim + l,dof_i,i] += FEBE.L2GAinv[k,xi]*FEBE.L2GAinv[l,xj]*FEBE.refbasisderivvals[FEBE.current_subset[dof_i] + (xi-1)*ndofs*ncomponents + (c-1)*ndofs,xj,i]
                        end
                    end    
                end    
            end  
        end  
    end  
    return nothing  
end


# SYMMETRICHESSIAN OPERATOR
# H1 ELEMENTS
# multiply tinverted jacobian of element trafo with gradient of basis function
# which yields (by chain rule) the gradient in x coordinates
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractH1FiniteElement,<:AbstractElementGeometry,<:SymmetricHessian{offdiagval},<:AssemblyType,edim,ncomponents,ndofs}, item) where {T,TvG,TiG,edim,ncomponents,ndofs,offdiagval}
    if FEBE.citem[] != item
        FEBE.citem = item

        # update transformation
        update_trafo!(FEBE.L2G, item)
        if !FEBE.L2G.nonlinear
            mapderiv!(FEBE.L2GAinv,FEBE.L2G,nothing)
        else
            @error "nonlinear local2global transformations not yet supported"
        end

        if FEBE.subset_handler != NothingFunction
            FEBE.subset_handler(FEBE.current_subset, item)
        end

        fill!(FEBE.cvals,0)
        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs
                for c = 1 : ncomponents
                    for k = 1 : edim, l = k : edim
                        # compute second derivatives  ∂^2 (x_k x_l) and put it in the right spot of Voigt vector
                        # note: if l > k the derivative is multiplied with offdiagval
                        if k != l
                            for xi = 1 : edim, xj = 1 : edim
                                FEBE.cvals[FEBE.compressiontargets[(c-1)*edim^2 + (k-1)*edim + l],dof_i,i] += offdiagval * FEBE.L2GAinv[k,xi]*FEBE.L2GAinv[l,xj]*FEBE.refbasisderivvals[FEBE.current_subset[dof_i] + (xi-1)*ndofs*ncomponents + (c-1)*ndofs,xj,i]
                            end
                        else
                            for xi = 1 : edim, xj = 1 : edim
                                FEBE.cvals[FEBE.compressiontargets[(c-1)*edim^2 + (k-1)*edim + l],dof_i,i] += FEBE.L2GAinv[k,xi]*FEBE.L2GAinv[l,xj]*FEBE.refbasisderivvals[FEBE.current_subset[dof_i] + (xi-1)*ndofs*ncomponents + (c-1)*ndofs,xj,i]
                            end
                        end
                    end    
                end    
            end  
        end  
    end  
    return nothing  
end



# LAPLACE OPERATOR
# H1 ELEMENTS
# multiply tinverted jacobian of element trafo with gradient of basis function
# which yields (by chain rule) the gradient in x coordinates
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractH1FiniteElement,<:AbstractElementGeometry,<:Laplacian,<:AssemblyType,edim,ncomponents,ndofs}, item) where {T,TvG,TiG,edim,ncomponents,ndofs}
    if FEBE.citem[] != item
        FEBE.citem = item

        # update transformation
        update_trafo!(FEBE.L2G, item)
        if !FEBE.L2G.nonlinear
            mapderiv!(FEBE.L2GAinv,FEBE.L2G,nothing)
        else
            @error "nonlinear local2global transformations not yet supported"
        end

        if FEBE.subset_handler != NothingFunction
            FEBE.subset_handler(FEBE.current_subset, item)
        end

        fill!(FEBE.cvals,0)
        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs
                for c = 1 : ncomponents
                    for k = 1 : edim
                        # second derivatives partial^2 (x_k x_l)
                        for xi = 1 : edim, xj = 1 : edim
                            FEBE.cvals[c,dof_i,i] += FEBE.L2GAinv[k,xi]*FEBE.L2GAinv[k,xj]*FEBE.refbasisderivvals[FEBE.current_subset[dof_i] + (xi-1)*ndofs*ncomponents + (c-1)*ndofs,xj,i]
                        end
                    end   
                end    
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
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractH1FiniteElement,<:AbstractElementGeometry,<:Gradient,<:AssemblyType,edim,ncomponents,ndofs}, item) where {T,TvG,TiG,edim,ncomponents,ndofs}
    if FEBE.citem[] != item
        FEBE.citem = item

        # update transformation
        update_trafo!(FEBE.L2G, item)
        if !FEBE.L2G.nonlinear
            mapderiv!(FEBE.L2GAinv,FEBE.L2G,nothing)
        else
            @error "nonlinear local2global transformations not yet supported"
        end

        if FEBE.subset_handler != NothingFunction
            FEBE.subset_handler(FEBE.current_subset, item)
        end

        fill!(FEBE.cvals,0)
        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs
                for c = 1 : ncomponents, k = 1 : edim
                    for j = 1 : edim
                        # compute duc/dxk
                        FEBE.cvals[k + FEBE.offsets[c],dof_i,i] += FEBE.L2GAinv[k,j]*FEBE.refbasisderivvals[FEBE.current_subset[dof_i] + FEBE.offsets2[c],j,i]
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
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractHdivFiniteElement,<:AbstractElementGeometry,<:Gradient,<:AssemblyType,edim,ncomponents,ndofs}, item) where {T,TvG,TiG,edim,ncomponents,ndofs}
    if FEBE.citem[] != item
        FEBE.citem = item
    
        # update transformation
        update_trafo!(FEBE.L2G, item)
        FEBE.coeffs_handler(FEBE.coefficients, item)
        FEBE.subset_handler(FEBE.current_subset, item)
        L2GM::Array{T,2} = FEBE.L2G.A # need matrix for Piola trafo
        if !FEBE.L2G.nonlinear
            mapderiv!(FEBE.L2GAinv,FEBE.L2G,nothing)
        else
            @error "nonlinear local2global transformations not yet supported"
        end

        # use Piola transformation on basisvals
        fill!(FEBE.cvals,0.0)
        det = FEBE.L2G.det # 1 alloc
        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs
                for c = 1 : ncomponents, k = 1 : edim
                    # compute duc/dxk
                    for j = 1 : edim
                        for m = 1 : edim
                            FEBE.cvals[k + FEBE.offsets[c],dof_i,i] += FEBE.L2GAinv[k,m] * L2GM[c,j] * FEBE.refbasisderivvals[FEBE.current_subset[dof_i] + FEBE.offsets2[j],m,i];
                        end
                    end    
                    FEBE.cvals[k + FEBE.offsets[c],dof_i,i] *= FEBE.coefficients[c,dof_i] / det
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
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractH1FiniteElement,<:AbstractElementGeometry2D,<:CurlScalar,<:AssemblyType,edim,ncomponents,ndofs}, item) where {T,TvG,TiG,edim,ncomponents,ndofs}
    if FEBE.citem[] != item
        FEBE.citem = item

        # update transformation
        update_trafo!(FEBE.L2G, item)
        if !FEBE.L2G.nonlinear
            mapderiv!(FEBE.L2GAinv,FEBE.L2G,nothing)
        else
            @error "nonlinear local2global transformations not yet supported"
        end

        if FEBE.subset_handler != NothingFunction
            FEBE.subset_handler(FEBE.current_subset, item)
        end

        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs
                FEBE.cvals[1,dof_i,i] = 0.0;
                FEBE.cvals[2,dof_i,i] = 0.0;
                for j = 1 : edim
                    FEBE.cvals[1,dof_i,i] -= FEBE.L2GAinv[2,j]*FEBE.refbasisderivvals[FEBE.current_subset[dof_i],j,i] # -du/dy
                    FEBE.cvals[2,dof_i,i] += FEBE.L2GAinv[1,j]*FEBE.refbasisderivvals[FEBE.current_subset[dof_i],j,i] # du/dx
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
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractH1FiniteElement,<:AbstractElementGeometry2D,<:Curl2D,<:AssemblyType,edim,ncomponents,ndofs}, item) where {T,TvG,TiG,edim,ncomponents,ndofs}
    if FEBE.citem[] != item
        FEBE.citem = item

        # update transformation
        update_trafo!(FEBE.L2G, item)
        if !FEBE.L2G.nonlinear
            mapderiv!(FEBE.L2GAinv,FEBE.L2G,nothing)
        else
            @error "nonlinear local2global transformations not yet supported"
        end

        if FEBE.subset_handler != NothingFunction
            FEBE.subset_handler(FEBE.current_subset, item)
        end

        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs
                FEBE.cvals[1,dof_i,i] = 0.0;
                for j = 1 : edim
                    FEBE.cvals[1,dof_i,i] -= FEBE.L2GAinv[2,j]*FEBE.refbasisderivvals[FEBE.current_subset[dof_i],j,i]  # -du1/dy
                    FEBE.cvals[1,dof_i,i] += FEBE.L2GAinv[1,j]*FEBE.refbasisderivvals[FEBE.current_subset[dof_i] + FEBE.offsets2[2],j,i]  # du2/dx
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
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractH1FiniteElement,<:AbstractElementGeometry,<:TangentialGradient,<:AssemblyType,edim,ncomponents,ndofs}, item) where {T,TvG,TiG,edim,ncomponents,ndofs}
    if FEBE.citem[] != item
        FEBE.citem = item

        # update transformation
        update_trafo!(FEBE.L2G, item)
        if !FEBE.L2G.nonlinear
            mapderiv!(FEBE.L2GAinv,FEBE.L2G,nothing)
        else
            @error "nonlinear local2global transformations not yet supported"
        end

        if FEBE.subset_handler != NothingFunction
            FEBE.subset_handler(FEBE.current_subset, item)
        end

        # compute tangent of item
        FEBE.iteminfo[1] = FEBE.coefficients3[2,item]
        FEBE.iteminfo[2] = -FEBE.coefficients3[1,item]

        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs
                for c = 1 : ncomponents, k = 1 : edim
                    FEBE.cvals[k + FEBE.offsets[c],dof_i,i] = 0.0;
                    for j = 1 : edim
                        # compute duc/dxk
                        FEBE.cvals[1,dof_i,i] += FEBE.L2GAinv[k,j]*FEBE.refbasisderivvals[FEBE.current_subset[dof_i] + FEBE.offsets2[c],j,i] * FEBE.iteminfo[c]
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

function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractH1FiniteElement,<:AbstractElementGeometry,<:SymmetricGradient{offdiagval},<:AssemblyType,edim,ncomponents,ndofs}, item) where {T,TvG,TiG,edim,ncomponents,ndofs,offdiagval}
    if (FEBE.citem[] != item)
        FEBE.citem = item

        # update transformation
        update_trafo!(FEBE.L2G, item)
        if !FEBE.L2G.nonlinear
            mapderiv!(FEBE.L2GAinv,FEBE.L2G,nothing)
        else
            @error "nonlinear local2global transformations not yet supported"
        end

        if FEBE.subset_handler != NothingFunction
            FEBE.subset_handler(FEBE.current_subset, item)
        end

        fill!(FEBE.cvals,0.0)
        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs
                for c = 1 : ncomponents, k = 1 : edim
                    for j = 1 : edim
                        # compute duc/dxk and put it into the right spot in the Voigt vector
                        if k != c
                            FEBE.cvals[FEBE.compressiontargets[k + FEBE.offsets[c]],dof_i,i] += offdiagval * FEBE.L2GAinv[k,j]*FEBE.refbasisderivvals[FEBE.current_subset[dof_i] + FEBE.offsets2[c],j,i]
                        else
                            FEBE.cvals[FEBE.compressiontargets[k + FEBE.offsets[c]],dof_i,i] += FEBE.L2GAinv[k,j]*FEBE.refbasisderivvals[FEBE.current_subset[dof_i] + FEBE.offsets2[c],j,i]
                        end
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
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractH1FiniteElementWithCoefficients,<:AbstractElementGeometry,<:Gradient,<:AssemblyType,edim, ncomponents,ndofs}, item) where {T,TvG,TiG,edim,ncomponents,ndofs}
    if FEBE.citem[] != item
        FEBE.citem = item

        # update transformation
        update_trafo!(FEBE.L2G, item)
        if !FEBE.L2G.nonlinear
            mapderiv!(FEBE.L2GAinv,FEBE.L2G,nothing)
        else
            @error "nonlinear local2global transformations not yet supported"
        end

        if FEBE.subset_handler != NothingFunction
            FEBE.subset_handler(FEBE.current_subset, item)
        end

        # get coefficients
        FEBE.coeffs_handler(FEBE.coefficients, item)

        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs
                for c = 1 : ncomponents, k = 1 : edim
                    FEBE.cvals[k + FEBE.offsets[c],dof_i,i] = 0.0;
                    for j = 1 : edim
                        # compute duc/dxk
                        FEBE.cvals[k + FEBE.offsets[c],dof_i,i] += FEBE.L2GAinv[k,j]*FEBE.refbasisderivvals[FEBE.current_subset[dof_i] + FEBE.offsets2[c],j,i]
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
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractH1FiniteElement,<:AbstractElementGeometry,<:Divergence,<:AssemblyType,edim,ncomponents,ndofs}, item) where {T,TvG,TiG,edim,ncomponents,ndofs}
    if FEBE.citem[] != item
        FEBE.citem = item

        # update transformation
        update_trafo!(FEBE.L2G, item)
        if !FEBE.L2G.nonlinear
            mapderiv!(FEBE.L2GAinv,FEBE.L2G,nothing)
        else
            @error "nonlinear local2global transformations not yet supported"
        end

        if FEBE.subset_handler != NothingFunction
            FEBE.subset_handler(FEBE.current_subset, item)
        end


        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs
                FEBE.cvals[1,dof_i,i] = 0.0;
                for k = 1 : edim
                    for j = 1 : edim
                        # compute duk/dxk
                        FEBE.cvals[1,dof_i,i] += FEBE.L2GAinv[k,j]*FEBE.refbasisderivvals[FEBE.current_subset[dof_i] + FEBE.offsets2[k],j,i]
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
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractH1FiniteElementWithCoefficients,<:AbstractElementGeometry,<:Divergence,<:AssemblyType,edim,ncomponents,ndofs}, item) where {T,TvG,TiG,edim,ncomponents,ndofs}
    if FEBE.citem[] != item
        FEBE.citem = item

        # update transformation
        update_trafo!(FEBE.L2G, item)
        if !FEBE.L2G.nonlinear
            mapderiv!(FEBE.L2GAinv,FEBE.L2G,nothing)
        else
            @error "nonlinear local2global transformations not yet supported"
        end

        # get coefficients
        FEBE.coeffs_handler(FEBE.coefficients, item)

        if FEBE.subset_handler != NothingFunction
            FEBE.subset_handler(FEBE.current_subset, item)
        end
        
        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs
                FEBE.cvals[1,dof_i,i] = 0.0;
                for k = 1 : edim
                    for j = 1 : edim
                        # compute duk/dxk
                        FEBE.cvals[1,dof_i,i] += FEBE.L2GAinv[k,j]*FEBE.refbasisderivvals[FEBE.current_subset[dof_i] + FEBE.offsets2[k],j,i] * FEBE.coefficients[k, dof_i]
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
function update_febe!(FEBE::ReconstructionFEBasisEvaluator{T,TvG,TiG,<:AbstractH1FiniteElement,<:AbstractHdivFiniteElement,<:AbstractElementGeometry,<:ReconstructionDivergence,<:AssemblyType,edim,ncomponents,ndofs,ndofs2}, item) where {T,TvG,TiG,edim,ncomponents,ndofs,ndofs2}
    if FEBE.citem[] != item
        FEBE.citem = item
    
        # update transformation
        update_trafo!(FEBE.L2G, item)
        if FEBE.L2G.nonlinear
            @error "nonlinear local2global transformations not yet supported"
        end
        FEBE.coeffs_handler(FEBE.coefficients, item)
        FEBE.subset_handler(FEBE.current_subset, item)

        # get local reconstruction coefficients
        get_rcoefficients!(FEBE.coefficients2, FEBE.reconst_handler, item)

        # use Piola transformation on Hdiv basis
        # and accumulate according to reconstruction coefficients
        fill!(FEBE.cvals,0.0)
        det = FEBE.L2G.det # 1 alloc
        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs # ndofs4item (H1)
                for dof_j = 1 : ndofs2
                    if FEBE.coefficients2[dof_i,dof_j] != 0
                        for j = 1 : edim
                            FEBE.cvals[1,dof_i,i] += FEBE.coefficients2[dof_i,dof_j] * FEBE.refbasisderivvals[FEBE.current_subset[dof_j] + FEBE.offsets2[j],j,i] * FEBE.coefficients[1,dof_j]
                        end  
                    end
                end
                FEBE.cvals[1,dof_i,i] /= det
            end
        end  
    end
    return nothing
end


# RECONSTRUCTION GRADIENT OPERATOR
# Hdiv ELEMENTS (Piola trafo)
#
function update_febe!(FEBE::ReconstructionFEBasisEvaluator{T,TvG,TiG,<:AbstractH1FiniteElementWithCoefficients,<:AbstractHdivFiniteElement,<:AbstractElementGeometry,<:ReconstructionGradient,<:AssemblyType,edim,ncomponents,ndofs,ndofs2}, item) where {T,TvG,TiG,edim,ncomponents,ndofs, ndofs2}
    if FEBE.citem[] != item
        FEBE.citem = item
    
        # update transformation
        update_trafo!(FEBE.L2G, item)
        L2GM::Array{T,2} = FEBE.L2G.A # need matrix for Piola trafo
        if !FEBE.L2G.nonlinear
            mapderiv!(FEBE.L2GAinv,FEBE.L2G,nothing)
        else
            @error "nonlinear local2global transformations not yet supported"
        end
        FEBE.coeffs_handler(FEBE.coefficients, item)

        # get local reconstruction coefficients
        get_rcoefficients!(FEBE.coefficients2, FEBE.reconst_handler, item)
        FEBE.subset_handler(FEBE.current_subset, item)

        # use Piola transformation on basisvals
        fill!(FEBE.cvals,0)
        for i = 1 : length(FEBE.xref)
            # calculate gradients of Hdiv basis functions and save them to coefficients3
            fill!(FEBE.coefficients3,0)
            det = FEBE.L2G.det
            for dof_i = 1 : ndofs2
                for c = 1 : ncomponents, k = 1 : edim
                    # compute duc/dxk
                    for j = 1 : edim
                        for m = 1 : edim
                            FEBE.coefficients3[k + FEBE.offsets[c],dof_i] += FEBE.L2GAinv[k,m] * L2GM[c,j] * FEBE.refbasisderivvals[FEBE.current_subset[dof_i] + FEBE.offsets2[j],m,i];
                        end
                    end    
                    FEBE.coefficients3[k + FEBE.offsets[c],dof_i] *= FEBE.coefficients[c,dof_i] / det
                end
            end

            # accumulate with reconstruction coefficients
            for dof_i = 1 : ndofs
                for dof_j = 1 : ndofs2
                    if FEBE.coefficients2[dof_i,dof_j] != 0
                        for k = 1 : size(FEBE.cvals,1)
                            FEBE.cvals[k,dof_i,i] += FEBE.coefficients2[dof_i,dof_j] * FEBE.coefficients3[k,dof_j]
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
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractHdivFiniteElement,<:AbstractElementGeometry,<:Divergence,<:AssemblyType,edim,ncomponents,ndofs,ndofs2}, item) where {T,TvG,TiG,edim,ncomponents,ndofs,ndofs2}
    if FEBE.citem[] != item
        FEBE.citem = item
        
        # update transformation
        update_trafo!(FEBE.L2G, item)
        if FEBE.L2G.nonlinear
            @error "nonlinear local2global transformations not yet supported"
        end
        FEBE.coeffs_handler(FEBE.coefficients, item)
        FEBE.subset_handler(FEBE.current_subset, item)

        # use Piola transformation on basisvals
        det = FEBE.L2G.det
        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs
                FEBE.cvals[1,dof_i,i] = 0.0;
                for j = 1 : edim
                    FEBE.cvals[1,dof_i,i] += FEBE.refbasisderivvals[FEBE.current_subset[dof_i] + FEBE.offsets2[j],j,i]
                end  
                FEBE.cvals[1,dof_i,i] *= FEBE.coefficients[1,dof_i] / det
            end
        end   
    end  
    return nothing
end



# CURL2D OPERATOR
# HCURL ELEMENTS on 2D domains
# covariant Piola transformation preserves curl2D (up to a factor 1/det(A))
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractHcurlFiniteElement,<:AbstractElementGeometry2D,<:Curl2D,<:AssemblyType,edim,ncomponents,ndofs,ndofs2}, item) where {T,TvG,TiG,edim,ncomponents,ndofs,ndofs2}
    if FEBE.citem[] != item
        FEBE.citem = item
        
        # update transformation
        update_trafo!(FEBE.L2G, item)
        if FEBE.L2G.nonlinear
            @error "nonlinear local2global transformations not yet supported"
        end
        FEBE.coeffs_handler(FEBE.coefficients, item)

        # use Piola transformation on basisvals
        det = FEBE.L2G.det
        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs
                FEBE.cvals[1,dof_i,i] = FEBE.refbasisderivvals[dof_i + FEBE.offsets2[1],2,i]
                FEBE.cvals[1,dof_i,i] -= FEBE.refbasisderivvals[dof_i + FEBE.offsets2[2],1,i]
                FEBE.cvals[1,dof_i,i] *= FEBE.coefficients[1,dof_i] / det
            end
        end    
    end  
    return nothing
end


# CURL3D OPERATOR
# HCURL ELEMENTS on 3D domains
# covariant Piola transformation preserves curl3D (up to a factor 1/det(A))
function update_febe!(FEBE::StandardFEBasisEvaluator{T,TvG,TiG,<:AbstractHcurlFiniteElement,<:AbstractElementGeometry3D,Curl3D,<:AssemblyType,edim,ncomponents,ndofs,ndofs2}, item) where {T,TvG,TiG,edim,ncomponents,ndofs,ndofs2}
    if FEBE.citem[] != item
        FEBE.citem = item
        
        # update transformation
        update_trafo!(FEBE.L2G, item)
        L2GM::Array{T,2} = FEBE.L2G.A # need matrix for Piola trafo
        if FEBE.L2G.nonlinear
            @error "nonlinear local2global transformations not yet supported"
        end
        fill!(FEBE.cvals,0.0)
        FEBE.coeffs_handler(FEBE.coefficients, item)

        # use Piola transformation on basisvals
        det = FEBE.L2G.det
        for i = 1 : length(FEBE.xref)
            for dof_i = 1 : ndofs
                for k = 1 : 3
                    FEBE.cvals[k,dof_i,i] += L2GM[k,1] * FEBE.refbasisderivvals[dof_i + FEBE.offsets2[3],2,i] # du3/dx2
                    FEBE.cvals[k,dof_i,i] -= L2GM[k,1] * FEBE.refbasisderivvals[dof_i + FEBE.offsets2[2],3,i] # - du2/dx3
                    FEBE.cvals[k,dof_i,i] += L2GM[k,2] * FEBE.refbasisderivvals[dof_i + FEBE.offsets2[1],3,i] # du3/dx1
                    FEBE.cvals[k,dof_i,i] -= L2GM[k,2] * FEBE.refbasisderivvals[dof_i + FEBE.offsets2[3],1,i] # - du1/dx3
                    FEBE.cvals[k,dof_i,i] += L2GM[k,3] * FEBE.refbasisderivvals[dof_i + FEBE.offsets2[2],1,i] # du2/dx1
                    FEBE.cvals[k,dof_i,i] -= L2GM[k,3] * FEBE.refbasisderivvals[dof_i + FEBE.offsets2[1],2,i] # - du1/dx2
                    FEBE.cvals[k,dof_i,i] *= FEBE.coefficients[k,dof_i] / det
                end
            end
        end    
    end  
    return nothing
end


"""
````
    eval_febe!(result, FEBE::FEBasisEvaluator, j::Int, i::Int, offset::Int = 0, factor = 1)
````

Evaluate the j-th basis function of the FEBasisEvaluator at the i-th quadrature point adn writes the (possibly vector-valued) evaluation into result (beginning at offset and with the specified factor).

"""
function eval_febe!(result::Array{T,1}, FEBE::FEBasisEvaluator{T,TvG,TiG,FEType,EG,FEOP,AT,edim,ncomponents,ndofs}, j::Int, i::Int, offset::Int = 0, factor = 1) where {T,TvG,TiG,FEType,FEOP,EG,AT,edim,ncomponents,ndofs}
    for k = 1 : size(FEBE.cvals,1)
        result[offset + k] = FEBE.cvals[k,j,i] * factor
    end  
    return nothing
end

"""
````
    eval_febe!(result, FEBE::FEBasisEvaluator, j::Int, i::Int, offset::Int = 0, factor = 1)
````

Evaluates the linear combination of the basisfunction with given coefficients at the i-th quadrature point and writes the (possibly vector-valued) evaluation into result (beginning at offset and with the specified factor).

"""
function eval_febe!(result::Array{T,1}, FEBE::FEBasisEvaluator{T,TvG,TiG,FEType,EG,FEOP,AT,edim,ncomponents,ndofs}, coefficients::Array{T,1}, i::Int, offset = 0, factor = 1) where {T,TvG,TiG,FEType,FEOP,EG,AT,edim,ncomponents,ndofs}
    for dof_i = 1 : ndofs # ndofs4item
        for k = 1 : size(FEBE.cvals,1)
            result[offset+k] += coefficients[dof_i] * FEBE.cvals[k,dof_i,i] * factor 
        end    
    end 
    return nothing
end




##### additional infrastructure for pairs of FE evaluators

struct SharedCValView{T} <: AbstractArray{T,3}
    cvals::Array{Array{T,3},1}
    k2cvalindex::Array{Int,1}
    offsets::Array{Int,1}
end

Base.getindex(SCV::SharedCValView{T},i::Int,j::Int,k::Int) where {T} = SCV.cvals[SCV.k2cvalindex[i]][i-SCV.offsets[i],j,k]
Base.size(SCV::SharedCValView{T}) where {T} = [SCV.offsets[end] + size(SCV.cvals[end],1), size(SCV.cvals[end],2), size(SCV.cvals[end],3)]
Base.size(SCV::SharedCValView{T},i) where {T} = (i == 1) ? SCV.offsets[end] + size(SCV.cvals[end],1) : size(SCV.cvals[end],i)

# collects two FEBasisEvaluators
struct FEBasisEvaluatorPair{T,TvG,TiG,FEB1Type,FEB2Type,FEType,EG,FEOP,AT,edim,ncomponents,ndofs} <: FEBasisEvaluator{T,TvG,TiG,FEType,EG,FEOP,AT,edim,ncomponents,ndofs}
    FE::FESpace{TvG,TiG,FEType}                          # link to full FE (e.g. for coefficients)
    FEB1::FEB1Type # first FEBasisEvaluator
    FEB2::FEB2Type # second FEBasisEvaluator
    cvals::SharedCValView{T}
    L2G::L2GTransformer{TvG,TiG, EG}           # local2global mapper
    xref::Array{Array{T,1},1} # xref of quadrature formula
end

# collects three FEBasisEvaluators
struct FEBasisEvaluatorTriple{T,TvG,TiG,FEB1Type,FEB2Type,FEB3Type,FEType,EG,FEOP,AT,edim,ncomponents,ndofs} <: FEBasisEvaluator{T,TvG,TiG,FEType,EG,FEOP,AT,edim,ncomponents,ndofs}
    FE::FESpace{TvG,TiG,FEType}       # link to full FE (e.g. for coefficients)
    FEB1::FEB1Type # first FEBasisEvaluator
    FEB2::FEB2Type # second FEBasisEvaluator
    FEB3::FEB3Type # third FEBasisEvaluator
    cvals::SharedCValView{T}
    L2G::L2GTransformer{TvG,TiG, EG}           # local2global mapper
    xref::Array{Array{T,1},1} # xref of quadrature formula
end

function FEBasisEvaluator{T,EG,FEOP,AT}(FE::FESpace{TvG,TiG,FEType,FEAT}, xref::Array{Array{T,1},1}) where {T, TvG, TiG, FEType <: AbstractFiniteElement, EG <: AbstractElementGeometry, FEOP <: OperatorPair, AT <: AssemblyType, FEAT <: AssemblyType}
    FEOP1 = FEOP.parameters[1]
    FEOP2 = FEOP.parameters[2]
    FEB1 = FEBasisEvaluator{T,EG,FEOP1,AT}(FE,xref)
    FEB2 = FEBasisEvaluator{T,EG,FEOP2,AT}(FE,xref)
    ncomponents = size(FEB1.cvals,1) + size(FEB2.cvals,1)
    indexes = ones(Int,ncomponents)
    offsets = zeros(Int,ncomponents)
    for j = 1 : size(FEB2.cvals,1)
        indexes[size(FEB1.cvals,1)+j] = 2
        offsets[size(FEB1.cvals,1)+j] = size(FEB1.cvals,1)
    end
    cvals = SharedCValView([FEB1.cvals,FEB2.cvals],indexes,offsets)
    edim = dim_element(EG)
    ndofs = size(FEB1.cvals,2)
    return FEBasisEvaluatorPair{T,TvG,TiG,typeof(FEB1),typeof(FEB2),FEType, EG, FEOP, AT, edim, ncomponents, ndofs}(FEB1.FE,FEB1,FEB2,cvals,FEB1.L2G,FEB1.xref)
end


function FEBasisEvaluator{T,EG,FEOP,AT}(FE::FESpace{TvG,TiG,FEType,FEAT}, xref::Array{Array{T,1},1}) where {T, TvG, TiG, FEType <: AbstractFiniteElement, EG <: AbstractElementGeometry, FEOP <: OperatorTriple, AT <: AssemblyType, FEAT <: AssemblyType}
    FEOP1 = FEOP.parameters[1]
    FEOP2 = FEOP.parameters[2]
    FEOP3 = FEOP.parameters[3]
    FEB1 = FEBasisEvaluator{T,EG,FEOP1,AT}(FE,xref)
    FEB2 = FEBasisEvaluator{T,EG,FEOP2,AT}(FE,xref)
    FEB3 = FEBasisEvaluator{T,EG,FEOP3,AT}(FE,xref)
    ncomponents = size(FEB1.cvals,1) + size(FEB2.cvals,1) + size(FEB3.cvals,1)
    indexes = ones(Int,ncomponents)
    offsets = zeros(Int,ncomponents)
    for j = 1 : size(FEB2.cvals,1)
        indexes[size(FEB1.cvals,1)+j] = 2
        offsets[size(FEB1.cvals,1)+j] = size(FEB1.cvals,1)
    end
    for j = 1 : size(FEB3.cvals,1)
        indexes[size(FEB1.cvals,1)+size(FEB1.cvals,2)+j] = 2
        offsets[size(FEB1.cvals,1)+size(FEB1.cvals,2)+j] = size(FEB1.cvals,1)
    end
    cvals = SharedCValView([FEB1.cvals,FEB2.cvals,FEB3.cvals],indexes,offsets)
    edim = dim_element(EG)
    ndofs = size(FEB1.cvals,2)
    return FEBasisEvaluatorPair{T,TvG,TiG,typeof(FEB1),typeof(FEB2),typeof(FEB3),FEType, EG, FEOP, AT, edim, ncomponents, ndofs}(FEB1.FE,FEB1,FEB2,FEB3,cvals,FEB1.L2G,FEB1.xref)
end

function update_febe!(FEBE::FEBasisEvaluatorPair, item)
    update_febe!(FEBE.FEB1, item)
    update_febe!(FEBE.FEB2, item)
    return nothing
end

function update_febe!(FEBE::FEBasisEvaluatorTriple, item)
    update_febe!(FEBE.FEB1, item)
    update_febe!(FEBE.FEB2, item)
    update_febe!(FEBE.FEB3, item)
    return nothing
end
