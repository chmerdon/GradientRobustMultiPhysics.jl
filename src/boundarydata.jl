
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
mutable struct BoundaryOperator <: AbstractPDEOperator
    regions4boundarytype :: Dict{Type{<:AbstractBoundaryType},Array{Int,1}}
    data4bregion :: Array{Any,1}
    timedependent :: Array{Bool,1}
    quadorder4bregion :: Array{Int,1}
    ifaces4bregion::Array{Array{Int,1},1}
    ibfaces4bregion::Array{Array{Int,1},1}
    homdofs::Array{Int,1}
end

function BoundaryOperator()
    regions4boundarytype = Dict{Type{<:AbstractBoundaryType},Array{Int,1}}()
    quadorder4bregion = zeros(Int,0)
    timedependent = Array{Bool,1}([])
    return BoundaryOperator(regions4boundarytype, [], timedependent, quadorder4bregion,Array{Array{Int,1},1}(undef,0),Array{Array{Int,1},1}(undef,0),[])
end

function Base.append!(O::BoundaryOperator,region::Int, btype::Type{<:AbstractBoundaryType}; data = Nothing)
    O.regions4boundarytype[btype]=push!(get(O.regions4boundarytype, btype, []),region)
    while length(O.data4bregion) < region
        push!(O.data4bregion, Nothing)
    end
    while length(O.quadorder4bregion) < region
        push!(O.quadorder4bregion, 0)
        push!(O.timedependent, false)
    end
    if typeof(data) <: AbstractUserDataType
        O.quadorder4bregion[region] = data.bonus_quadorder
        O.timedependent[region] = is_timedependent(data)
    end
    O.data4bregion[region] = data
end


function Base.append!(O::BoundaryOperator,regions::Array{Int,1}, btype::Type{<:AbstractBoundaryType}; data = Nothing)
    for j = 1 : length(regions)
        append!(O,regions[j], btype; data = data)
    end
end



# this function assembles all boundary data at once
# first all interpolation Dirichlet boundaries are assembled
# then all hoomogeneous Dirichlet boundaries are set to zero
# then all DirichletBestapprox boundaries are handled (previous data is fixed)
function boundarydata!(
    Target::FEVectorBlock{T,Tv,Ti},
    O::BoundaryOperator,
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
    if length(O.regions4boundarytype) > 0
        xBFaceDofs::DofMapTypes{Ti} = FE[BFaceDofs]
        nbfaces = num_sources(xBFaceDofs)
        xBFaceFaces::Array{Ti,1} = FE.xgrid[BFaceFaces]
        xBFaceRegions = FE.xgrid[BFaceRegions]
    end

    ######################
    # Dirichlet boundary #
    ######################

    # INTERPOLATION DIRICHLET BOUNDARY
    InterDirichletBoundaryRegions = get(O.regions4boundarytype,InterpolateDirichletBoundary,[])
    if length(InterDirichletBoundaryRegions) > 0

        # find Dirichlet dofs
        for r = 1 : length(InterDirichletBoundaryRegions)
            if skip_enumerations == false
                while length(O.ifaces4bregion) < r
                    push!(O.ifaces4bregion,[])
                    push!(O.ibfaces4bregion,[])
                end
                O.ifaces4bregion[r] = []
                O.ibfaces4bregion[r] = []
                ifaces = O.ifaces4bregion[r]
                ibfaces = O.ibfaces4bregion[r]
                bregiondofs = []
                for bface = 1 : nbfaces
                    if xBFaceRegions[bface] == InterDirichletBoundaryRegions[r]
                        append!(ifaces,xBFaceFaces[bface])
                        append!(ibfaces,bface)
                        append!(bregiondofs,xBFaceDofs[:,bface])
                    end
                end    
                bregiondofs = Base.unique(bregiondofs)
                append!(fixed_dofs,bregiondofs)
            end
            bregion::Int = InterDirichletBoundaryRegions[r]
            ifaces::Array{Int,1} = O.ifaces4bregion[r]
            ibfaces::Array{Int,1} = O.ibfaces4bregion[r]
            if length(ifaces) > 0
                if FE.broken == true
                    # face interpolation expects continuous dofmaps
                    # quick and dirty fix: use face interpolation and remap dofs to broken dofs
                    FESc = FESpace{FEType}(FE.xgrid)
                    Targetc = FEVector{T}("auxiliary data",FESc)
                    interpolate!(Targetc[1], FESc, ON_FACES, O.data4bregion[bregion]; items = ifaces, time = time)
                    xBFaceDofsc = FESc[BFaceDofs]
                    dof::Int = 0
                    dofc::Int = 0
                    for bface in ibfaces
                        for k = 1 : num_targets(xBFaceDofs,bface)
                            dof = xBFaceDofs[k,bface]
                            dofc = xBFaceDofsc[k,bface]
                            Target[dof] = Targetc.entries[dofc]
                        end
                    end
                else
                    # use face interpolation
                    interpolate!(Target, ON_BFACES, O.data4bregion[bregion]; items = ibfaces, time = time)
                end
            end
        end   

        @debug "Int-DBnd = $InterDirichletBoundaryRegions"# (ndofs = $(length(fixed_dofs)))"
    end

    # HOMOGENEOUS DIRICHLET BOUNDARY
    HomDirichletBoundaryRegions = get(O.regions4boundarytype,HomogeneousDirichletBoundary,[])
    if length(HomDirichletBoundaryRegions) > 0

        # find Dirichlet dofs
        hom_dofs = O.homdofs
        if skip_enumerations == false
            hom_dofs = []
            for r = 1 : length(HomDirichletBoundaryRegions)
                for bface = 1 : nbfaces
                    if xBFaceRegions[bface] == HomDirichletBoundaryRegions[r]
                        append!(hom_dofs,xBFaceDofs[:,bface])
                    end    
                end    
            end
            hom_dofs = Base.unique(hom_dofs)
            append!(fixed_dofs,hom_dofs)
            fixed_dofs = Base.unique(fixed_dofs)
        end

        # set homdofs to zero
        for j in hom_dofs
            Target[j] = 0
        end    

        @debug "Hom-DBnd = $HomDirichletBoundaryRegions (ndofs = $(length(hom_dofs)))"
    end

    # BEST-APPROXIMATION DIRICHLET BOUNDARY
    BADirichletBoundaryRegions = get(O.regions4boundarytype,BestapproxDirichletBoundary,[])
    if length(BADirichletBoundaryRegions) > 0

        # find Dirichlet dofs
        BAdofs::Array{Ti,1} = zeros(Ti,0)
        for bface = 1 : nbfaces
            for r = 1 : length(BADirichletBoundaryRegions)
                if xBFaceRegions[bface] == BADirichletBoundaryRegions[r]
                    for dof = 1 : num_targets(xBFaceDofs,bface)
                        push!(BAdofs,xBFaceDofs[dof,bface])
                    end
                    break
                end    
            end    
        end
        Base.unique!(BAdofs)

        @debug "BA-DBnd = $BADirichletBoundaryRegions (ndofs = $(length(BAdofs)))"

        bonus_quadorder::Int = maximum(O.quadorder4bregion[BADirichletBoundaryRegions[:]])
        Dboperator = DefaultDirichletBoundaryOperator4FE(FEType)
        b::Array{T,1} = zeros(T,FE.ndofs)
        A = FEMatrix{T}("MassMatrixBnd", FE)

        if Dboperator == Identity
            for region in BADirichletBoundaryRegions
                action = fdot_action(O.data4bregion[region])
                set_time!(O.data4bregion[region], time)
                set_time!(action, time)
                RHS_bnd = DiscreteLinearForm([Dboperator], [FE], action; T = T, AT = ON_BFACES, regions = [region], name = "RHS bnd data bestapprox")
                assemble!(b, RHS_bnd)
            end
            L2ProductBnd = DiscreteSymmetricBilinearForm([Dboperator, Dboperator], [FE, FE]; T = T, AT = ON_BFACES, regions = BADirichletBoundaryRegions, name = "LHS bnd data bestapprox")    
            assemble!(A[1],L2ProductBnd)
        elseif Dboperator == NormalFlux
            for region in BADirichletBoundaryRegions
                action = fdotn_action(O.data4bregion[region], FE.xgrid; bfaces = true)
                set_time!(O.data4bregion[region], time)
                set_time!(action, time)
                RHS_bnd = DiscreteLinearForm([Dboperator], [FE], action; T = T, AT = ON_BFACES, regions = [region], name = "RHS bnd data bestapprox")
                assemble!(b, RHS_bnd)
            end
            L2ProductBnd = DiscreteSymmetricBilinearForm([Dboperator, Dboperator], [FE, FE]; T = T, AT = ON_BFACES, regions = BADirichletBoundaryRegions, name = "LHS bnd data NormalFlux bestapprox")    
            assemble!(A[1],L2ProductBnd)
        elseif Dboperator == TangentFlux && xdim == 2 # Hcurl on 2D domains
            for region in BADirichletBoundaryRegions
                action = fdott2d_action(O.data4bregion[region], FE.xgrid; bfaces = true)
                set_time!(O.data4bregion[region], time)
                set_time!(action, time)
                RHS_bnd = DiscreteLinearForm([Dboperator], [FE], action; T = T, AT = ON_BFACES, regions = [region], name = "RHS bnd data bestapprox")
                assemble!(b, RHS_bnd)
            end
            L2ProductBnd = DiscreteSymmetricBilinearForm([Dboperator, Dboperator], [FE, FE]; T = T, AT = ON_BFACES, regions = BADirichletBoundaryRegions, name = "LHS bnd data TangentFlux bestapprox")    
            assemble!(A[1],L2ProductBnd)
        elseif Dboperator == TangentFlux && xdim == 3 # Hcurl on 3D domains, does not work properly yet
            @warn "Hcurl boundary data in 3D may not work properly yet"
            for region in BADirichletBoundaryRegions
                action = fdott23_action(O.data4bregion[region], FE.xgrid; bedges = true)
                set_time!(O.data4bregion[region], time)
                set_time!(action, time)
                RHS_bnd = DiscreteLinearForm([Dboperator], [FE], action; T = T, AT = ON_BFACES, regions = [region], name = "RHS bnd data bestapprox")
                assemble!(b, RHS_bnd)
            end
            L2ProductBnd = DiscreteSymmetricBilinearForm([Dboperator, Dboperator], [FE, FE]; T = T, AT = ON_BFACES, regions = BADirichletBoundaryRegions, name = "LHS bnd data TangentFlux bestapprox")    
            assemble!(A[1],L2ProductBnd)
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

    # INTERPOLATION DIRICHLET BOUNDARY
    for id = 1 : length(OtherData)
        CorrectDirichletBoundaryRegions = get(O.regions4boundarytype,CorrectDirichletBoundary{id},[])
        if length(CorrectDirichletBoundaryRegions) > 0
            ## generate a copy of the TargetData and interpolate the OtherData[id]
            TargetCopy = deepcopy(Target)
            interpolate!(TargetCopy, OtherData[id])

            # find Dirichlet dofs
            for r = 1 : length(CorrectDirichletBoundaryRegions)
                bregion = CorrectDirichletBoundaryRegions[r]
                data_exact = O.data4bregion[bregion]
                if skip_enumerations == false
                    while length(O.ifaces4bregion) < r
                        push!(O.ifaces4bregion,[])
                        push!(O.ibfaces4bregion,[])
                    end
                    O.ifaces4bregion[r] = []
                    O.ibfaces4bregion[r] = []
                    ifaces = O.ifaces4bregion[r]
                    ibfaces = O.ibfaces4bregion[r]
                    bregiondofs = []
                    for bface = 1 : nbfaces
                        if xBFaceRegions[bface] == CorrectDirichletBoundaryRegions[r]
                            append!(ifaces,xBFaceFaces[bface])
                            append!(ibfaces,bface)
                            append!(bregiondofs,xBFaceDofs[:,bface])
                        end
                    end    
                    bregiondofs = Base.unique(bregiondofs)
                    append!(fixed_dofs,bregiondofs)
                end
                bregion::Int = CorrectDirichletBoundaryRegions[r]
                ifaces::Array{Int,1} = O.ifaces4bregion[r]
                ibfaces::Array{Int,1} = O.ibfaces4bregion[r]
                if length(ifaces) > 0
                    interpolate!(Target, ON_BFACES, data_exact; items = ibfaces, time = time)
                    ## subtract interpolation of OtherData
                    for dof in bregiondofs
                        Target[dof] -= TargetCopy[dof]
                    end
                end
            end   
    
            @debug "Corr-DBnd = $CorrectDirichletBoundaryRegions"# (ndofs = $(length(fixed_dofs)))"
        end
    end
    
    if skip_enumerations
        return nothing
    else
        return fixed_dofs
    end
end    
