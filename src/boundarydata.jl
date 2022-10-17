
abstract type AbstractBoundaryType end
abstract type DirichletBoundary <: AbstractBoundaryType end
"""
$(TYPEDEF)

DirichletBoundary where data is computed by L2-bestapproximation
"""
abstract type BestapproxDirichletBoundary <: DirichletBoundary end
"""
$(TYPEDEF)

DirichletBoundary where data is computed by interpolation
"""
abstract type InterpolateDirichletBoundary <: DirichletBoundary end
"""
$(TYPEDEF)

Corrects boundary data of specified id, such that piecewise integral means of the sum of both unknowns are preserved
"""
abstract type CorrectDirichletBoundary{id} <: DirichletBoundary end
"""
$(TYPEDEF)

homogeneous Dirichlet data
"""
abstract type HomogeneousDirichletBoundary <: DirichletBoundary end

# operator to be used for Dirichlet boundary data
DefaultDirichletBoundaryOperator4FE(::Type{<:AbstractH1FiniteElement}) = Identity
DefaultDirichletBoundaryOperator4FE(::Type{<:AbstractHdivFiniteElement}) = NormalFlux
DefaultDirichletBoundaryOperator4FE(::Type{<:AbstractHcurlFiniteElement}) = TangentFlux

"""
$(TYPEDEF)

collects boundary data for a component of the system and allows to specify a AbstractBoundaryType for each boundary region
so far only DirichletBoundary types (see above)

"""
mutable struct BoundaryData{BDT <: AbstractBoundaryType, MType, FType}
    data::FType
    mask::MType   # which components are involved?
    bregions::Array{Int,1}  # which regions are involved?
    timedependent::Bool
    ifaces::Array{Int,1}
    ibfaces::Array{Int,1}
    bdofs::Array{Int,1}
end

BoundaryDataType(::BoundaryData{BDT}) where {BDT} = BDT
is_timedependent(BD::BoundaryData) = BD.timedependent

function BoundaryData(BDT::Type{<:AbstractBoundaryType}; data = nothing, regions = [0], mask = 1)
    if data == nothing
        timedependent = false
    else
        timedependent = is_timedependent(data)
    end
    return BoundaryData{BDT, typeof(mask), typeof(data)}(data, mask, regions, timedependent, zeros(Int,0), zeros(Int,0), zeros(Int,0))
end


# this function assembles all boundary data for the Target block at once
# first all interpolation Dirichlet boundaries are assembled
# then all homogeneous Dirichlet boundaries are set to zero
# then all DirichletBestapprox boundaries are handled (previous data is fixed)
# at last all CorrectDirichlet boundaries are assembled (taking into account previous data)
function boundarydata!(
    Target::FEVectorBlock{T,Tv,Ti},
    O::Array{BoundaryData,1},
    OtherData = [];
    time = 0,
    fixed_penalty = 1e60,
    skip_enumerations = false) where {T,Tv,Ti}

    fixed_dofs = []
  
    FE = Target.FES
    xdim::Int = size(FE.xgrid[Coordinates],1) 
    FEType = eltype(FE)
    ncomponents::Int = get_ncomponents(FEType)
    nbfaces::Int = 0
    if length(O) > 0
        xBFaceDofs::DofMapTypes{Ti} = FE[BFaceDofs]
        nbfaces = num_sources(xBFaceDofs)
        xBFaceFaces::Array{Ti,1} = FE.xgrid[BFaceFaces]
        xBFaceRegions = FE.xgrid[BFaceRegions]
    end

    ######################
    # Dirichlet boundary # TODO : APPLY MASK!!!
    ######################

    # INTERPOLATION DIRICHLET BOUNDARY
    InterDirichletBoundaryRegions = []
    InterDirichletBoundaryOperators = []
    for j = 1 : length(O)
        if BoundaryDataType(O[j]) == InterpolateDirichletBoundary
            append!(InterDirichletBoundaryRegions, O[j].bregions)
            push!(InterDirichletBoundaryOperators,j)
            regions::Array{Int,1} = O[j].bregions
            ifaces::Array{Int,1} = O[j].ifaces
            ibfaces::Array{Int,1} = O[j].ibfaces
            bdofs::Array{Int,1} = O[j].bdofs
            mask = O[j].mask
            if skip_enumerations == false
                bdofs = []
                ibfaces = []
                ifaces = []
                if any(mask .== 0)
                    # only some components are Dirichlet
                    @assert ncomponents == length(mask)
                    @assert FEType <: AbstractH1FiniteElement && !(FEType <: AbstractH1FiniteElementWithCoefficients)
                    @assert length(FE.xgrid[UniqueBFaceGeometries]) == 1
                    coffsets = get_local_coffsets(FEType, ON_BFACES, FE.xgrid[UniqueBFaceGeometries][1])
                    dofmask = []
                    for j = 1 : length(mask)
                        if mask[j] == 1
                            for dof = coffsets[j]+1 : coffsets[j+1]
                                push!(dofmask,dof)
                            end
                        end
                    end
                    for bface = 1 : nbfaces
                        if xBFaceRegions[bface] in regions
                            append!(ifaces,xBFaceFaces[bface])
                            append!(ibfaces,bface)
                            for dof in dofmask
                                append!(bdofs, xBFaceDofs[dof,bface])
                            end
                        end    
                    end    
                else
                    for bface = 1 : nbfaces
                        if xBFaceRegions[bface] in regions
                            append!(ifaces,xBFaceFaces[bface])
                            append!(ibfaces,bface)
                            for dof = 1 : num_targets(xBFaceDofs,bface)
                                append!(bdofs, xBFaceDofs[dof,bface])
                            end
                        end
                    end    
                end   
                bdofs = Base.unique(bdofs)
                append!(fixed_dofs,bdofs)
                fixed_dofs = Base.unique(fixed_dofs)
            end
            if length(ifaces) > 0
                if FE.broken == true || any(mask .== 0)
                    @show mask
                    # face interpolation expects continuous dofmaps
                    # quick and dirty fix: use face interpolation and remap dofs to broken dofs
                    FESc = FESpace{FEType}(FE.xgrid)
                    Targetc = FEVector{T}(FESc)
                    interpolate!(Targetc[1], FESc, ON_FACES, O[j].data; items = ifaces, time = time)
                    xBFaceDofsc = FESc[BFaceDofs]
                    dof::Int = 0
                    dofc::Int = 0
                    if any(mask .== 0)
                        for j = 1 : length(mask)
                            if mask[j] == 1
                                for dof = coffsets[j]+1 : coffsets[j+1]
                                    push!(dofmask,dof)
                                end
                            end
                        end
                        for bface = 1 : nbfaces
                            for k in dofmask
                                dof = xBFaceDofs[k,bface]
                                dofc = xBFaceDofsc[k,bface]
                                Target[dof] = Targetc.entries[dofc]
                            end
                        end    
                    else
                        for bface in ibfaces
                            for k = 1 : num_targets(xBFaceDofs,bface)
                                dof = xBFaceDofs[k,bface]
                                dofc = xBFaceDofsc[k,bface]
                                Target[dof] = Targetc.entries[dofc]
                            end
                        end
                    end
                else
                    # use face interpolation
                    interpolate!(Target, ON_BFACES, O[j].data; items = ibfaces, time = time)
                end
            end
        end
    end

    if length(InterDirichletBoundaryRegions) > 0
        @debug "Int-DBnd = $InterDirichletBoundaryRegions"
    end

    # HOMOGENEOUS DIRICHLET BOUNDARY
    HomDirichletBoundaryRegions = []
    HomDirichletBoundaryOperators = []
    for j = 1 : length(O)
        if BoundaryDataType(O[j]) == HomogeneousDirichletBoundary
            append!(HomDirichletBoundaryRegions, O[j].bregions)
            push!(HomDirichletBoundaryOperators,j)

            # find Dirichlet dofs
            regions::Array{Int,1} = O[j].bregions
            bdofs::Array{Int,1} = O[j].bdofs
            if skip_enumerations == false
                bdofs = []
                mask = O[j].mask
                if any(mask .== 0)
                    # only some components are Dirichlet
                    @assert ncomponents == length(mask)
                    @assert FEType <: AbstractH1FiniteElement && !(FEType <: AbstractH1FiniteElementWithCoefficients)
                    @assert length(FE.xgrid[UniqueBFaceGeometries]) == 1
                    coffsets = get_local_coffsets(FEType, ON_BFACES, FE.xgrid[UniqueBFaceGeometries][1])
                    dofmask = []
                    for j = 1 : length(mask)
                        if mask[j] == 1
                            for dof = coffsets[j]+1 : coffsets[j+1]
                                push!(dofmask,dof)
                            end
                        end
                    end
                    for bface = 1 : nbfaces
                        if xBFaceRegions[bface] in regions
                            for dof in dofmask
                                append!(bdofs, xBFaceDofs[dof,bface])
                            end
                        end    
                    end    
                else
                    for bface = 1 : nbfaces
                        if xBFaceRegions[bface] in regions
                            for dof = 1 : num_targets(xBFaceDofs,bface)
                                append!(bdofs, xBFaceDofs[dof,bface])
                            end
                        end    
                    end    
                end   
                bdofs = Base.unique(bdofs)
                append!(fixed_dofs,bdofs)
                fixed_dofs = Base.unique(fixed_dofs)
            end

            # set homdofs to zero

            for j in bdofs
                Target[j] = 0
            end
        end
    end
    if length(HomDirichletBoundaryOperators) > 0
        @debug "Hom-DBnd = $HomDirichletBoundaryRegions"
    end

    # BEST-APPROXIMATION DIRICHLET BOUNDARY
    BADirichletBoundaryRegions = []
    BADirichletBoundaryOperators = []
    for j = 1 : length(O)
        if BoundaryDataType(O[j]) == BestapproxDirichletBoundary
            append!(BADirichletBoundaryRegions, O[j].bregions)
            push!(BADirichletBoundaryOperators,j)
        end
    end

    if length(BADirichletBoundaryRegions) > 0

        # prepare vector to store bestapproximation boundary data
        Dboperator = DefaultDirichletBoundaryOperator4FE(FEType)
        b::Array{T,1} = zeros(T,FE.ndofs)

        # find Dirichlet dofs
        BAdofs::Array{Ti,1} = zeros(Ti,0)
        exclude_dofs = zeros(Ti,0)
        for j in BADirichletBoundaryOperators
            bdofs::Array{Int,1} = O[j].bdofs
            if skip_enumerations == false
                bdofs = []
                mask = O[j].mask
                if any(mask .== 0)
                    @warn "mask for BestapproximationDirichletBoundary not available (ignoring mask)"
                end
                regions = O[j].bregions
                for bface = 1 : nbfaces
                    if xBFaceRegions[bface] in regions
                        for dof = 1 : num_targets(xBFaceDofs,bface)
                            push!(bdofs,xBFaceDofs[dof,bface])
                        end
                    end   
                end
            end

            ## assemble rhs for best-approximation problem
            if Dboperator == Identity
                action = fdot_action(O[j].data)
            elseif Dboperator == NormalFlux
                action = fdotn_action(O[j].data, FE.xgrid; bfaces = true)
            elseif Dboperator == TangentFlux && xdim == 2 # Hcurl on 2D domains
                action = fdott2d_action(O[j].data, FE.xgrid; bfaces = true)
            elseif Dboperator == TangentFlux && xdim == 3 # Hcurl on 3D domains, does not work properly yet
                @warn "Hcurl boundary data in 3D may not work properly yet"
                action = fdott23_action(O[j].data, FE.xgrid; bedges = true)
            end    
            set_time!(O[j].data, time)
            set_time!(action, time)
            RHS_bnd = DiscreteLinearForm([Dboperator], [FE], action; T = T, AT = ON_BFACES, regions = regions, name = "RHS bnd data bestapprox")
            assemble!(b, RHS_bnd)

            append!(BAdofs, bdofs)
        end
        Base.unique!(BAdofs)

        @debug "BA-DBnd = $BADirichletBoundaryRegions (ndofs = $(length(BAdofs)))"

        ## assemble matrix
        A = FEMatrix{T}(FE; name = "mass matrix bnd")
        if Dboperator == Identity
            L2ProductBnd = DiscreteSymmetricBilinearForm([Dboperator, Dboperator], [FE, FE]; T = T, AT = ON_BFACES, regions = BADirichletBoundaryRegions, name = "LHS bnd data bestapprox")    
            assemble!(A[1],L2ProductBnd)
        elseif Dboperator == NormalFlux
            L2ProductBnd = DiscreteSymmetricBilinearForm([Dboperator, Dboperator], [FE, FE]; T = T, AT = ON_BFACES, regions = BADirichletBoundaryRegions, name = "LHS bnd data NormalFlux bestapprox")    
            assemble!(A[1],L2ProductBnd)
        elseif Dboperator == TangentFlux && xdim == 2 # Hcurl on 2D domains
            L2ProductBnd = DiscreteSymmetricBilinearForm([Dboperator, Dboperator], [FE, FE]; T = T, AT = ON_BFACES, regions = BADirichletBoundaryRegions, name = "LHS bnd data TangentFlux bestapprox")    
            assemble!(A[1],L2ProductBnd)
        elseif Dboperator == TangentFlux && xdim == 3 # Hcurl on 3D domains, does not work properly yet
            L2ProductBnd = DiscreteSymmetricBilinearForm([Dboperator, Dboperator], [FE, FE]; T = T, AT = ON_BFACES, regions = BADirichletBoundaryRegions, name = "LHS bnd data TangentFlux bestapprox")    
            assemble!(A[1],L2ProductBnd)
        end    

        # TODO: remove dofs that do not match mask
        for j in exclude_dofs
            _addnz(A.entries,j,j,fixed_penalty,1)
            b[j,1] = 0
        end

        # fix already set dofs by other boundary conditions
        for j in fixed_dofs
            _addnz(A.entries,j,j,fixed_penalty,1)
            b[j,1] = Target[j]*fixed_penalty
        end

        flush!(A.entries)
        # add new fixed dofs from best approximation boundary
        append!(fixed_dofs,BAdofs)
        fixed_dofs::Array{Ti,1} = Base.unique(fixed_dofs)

        # solve best approximation problem on boundary and write into Target
        @debug "solving for best-approximation boundary data"

        if (true) # compress matrix by removing all dofs in the interior
            dof2sparsedof::Array{Ti,1} = zeros(Ti,FE.ndofs)
            newcolptr::Array{Int64,1} = zeros(Int64,0)
            newrowval::Array{Int64,1} = zeros(Int64,0)
            dof = 0
            diff::Int64 = 0
            for j = 1 : FE.ndofs
                diff = A.entries.cscmatrix.colptr[j] != A.entries.cscmatrix.colptr[j+1]
                if diff > 0
                    dof += 1
                    dof2sparsedof[j] = dof
                end
            end

            smallb::Array{T,1} = zeros(T,dof)
            sparsedof2dof::Array{Ti,1} = zeros(Ti,dof)
            for j = 1 : FE.ndofs
                if dof2sparsedof[j] > 0
                    push!(newcolptr,A.entries.cscmatrix.colptr[j])
                    append!(newrowval,dof2sparsedof[A.entries.cscmatrix.rowval[A.entries.cscmatrix.colptr[j]:A.entries.cscmatrix.colptr[j+1]-1]])
                    smallb[dof2sparsedof[j]] = b[j]
                    sparsedof2dof[dof2sparsedof[j]] = j
                end
            end
            push!(newcolptr,A.entries.cscmatrix.colptr[end])
            A.entries.cscmatrix = SparseMatrixCSC{T,Int64}(dof,dof,newcolptr,newrowval,A.entries.cscmatrix.nzval)

            Target[sparsedof2dof] = A.entries\smallb

            ## check target_residual
            ## @show sum((A.entries*Target[sparsedof2dof] - smallb).^2)
        else # old way: penalize all interior dofs

            for j in setdiff(1:FE.ndofs,fixed_dofs)
                A[1][j,j] = fixed_penalty
                b[j] = 0
            end

            Target[fixed_dofs] = (A.entries\b[:,1])[fixed_dofs]
        end
    end

    # CORRECTING INTERPOLATION DIRICHLET BOUNDARY
    for id = 1 : length(OtherData)
        CorrectDirichletBoundaryRegions = []
        for j = 1 : length(O)
            if BoundaryDataType(O[j]) == CorrectDirichletBoundary{id}
                regions::Array{Int,1} = O[j].bregions
                append!(CorrectDirichletBoundaryRegions, regions)

                ## generate a copy of the TargetData and interpolate the OtherData[id]
                TargetCopy = deepcopy(Target)
                interpolate!(TargetCopy, OtherData[id])

                # find Dirichlet dofs
                ifaces::Array{Int,1} = O[j].ifaces
                ibfaces::Array{Int,1} = O[j].ibfaces
                bdofs::Array{Int,1} = O[j].bdofs
                if skip_enumerations == false
                    bdofs = []
                    ibfaces = []
                    ifaces = []
                    for bface = 1 : nbfaces
                        if xBFaceRegions[bface] in regions
                            append!(ifaces,xBFaceFaces[bface])
                            append!(ibfaces,bface)
                            for dof = 1 : num_targets(xBFaceDofs,bface)
                                append!(bdofs, xBFaceDofs[dof,bface])
                            end
                        end
                    end    
                    bdofs = Base.unique(bdofs)
                    append!(fixed_dofs,bdofs)
                    fixed_dofs = Base.unique(fixed_dofs)
                end
                if length(ifaces) > 0
                    interpolate!(Target, ON_BFACES, O[j].data; items = ibfaces, time = time)
                    ## subtract interpolation of OtherData
                    for dof in bdofs
                        Target[dof] -= TargetCopy[dof]
                    end
                end
            end
        end
        if length(CorrectDirichletBoundaryRegions) > 0
            @debug "Corr-DBnd = $CorrectDirichletBoundaryRegions"# (ndofs = $(length(fixed_dofs)))"
        end
    end
    
    if skip_enumerations
        return nothing
    else
        return fixed_dofs
    end
end    
