
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
struct BoundaryOperator <: AbstractPDEOperator
    regions4boundarytype :: Dict{Type{<:AbstractBoundaryType},Array{Int,1}}
    data4bregion :: Array{Any,1}
    timedependent :: Array{Bool,1}
    quadorder4bregion :: Array{Int,1}
end

function BoundaryOperator()
    regions4boundarytype = Dict{Type{<:AbstractBoundaryType},Array{Int,1}}()
    quadorder4bregion = zeros(Int,0)
    timedependent = Array{Bool,1}([])
    return BoundaryOperator(regions4boundarytype, [], timedependent, quadorder4bregion)
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
    if typeof(data) <: UserData{<:AbstractDataFunction}
        O.quadorder4bregion[region] = data.quadorder
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
    Target::FEVectorBlock,
    O::BoundaryOperator;
    time = 0,
    fixed_penalty::Float64 = 1e60)
    fixed_dofs = []
  
    FE = Target.FES
    xdim = size(FE.xgrid[Coordinates],1) 
    FEType = eltype(FE)
    ncomponents::Int = get_ncomponents(FEType)
    xBFaceDofs = nothing
    nbfaces::Int = 0
    if length(O.regions4boundarytype) > 0
        xBFaceDofs = FE[BFaceDofs]
        nbfaces = num_sources(xBFaceDofs)
        xBFaces = FE.xgrid[BFaces]
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
            bregiondofs = []
            ifaces = []
            ibfaces = []
            for bface = 1 : nbfaces
                if xBFaceRegions[bface] == InterDirichletBoundaryRegions[r]
                    append!(ifaces,xBFaces[bface])
                    append!(ibfaces,bface)
                    append!(bregiondofs,xBFaceDofs[:,bface])
                end
            end    
            bregiondofs = Base.unique(bregiondofs)
            bregion = InterDirichletBoundaryRegions[r]
            append!(fixed_dofs,bregiondofs)
            if length(ifaces) > 0
                if FE.broken == true
                    # face interpolation expects continuous dofmaps
                    # quick and dirty fix: use face interpolation and remap dofs to broken dofs
                    FESc = FESpace{FEType}(FE.xgrid)
                    Targetc = FEVector{Float64}("auxiliary data",FESc)
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

        @debug "Int-DBnd = $InterDirichletBoundaryRegions (ndofs = $(length(fixed_dofs)))"
    end

    # HOMOGENEOUS DIRICHLET BOUNDARY
    HomDirichletBoundaryRegions = get(O.regions4boundarytype,HomogeneousDirichletBoundary,[])
    if length(HomDirichletBoundaryRegions) > 0

        # find Dirichlet dofs
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
        for j in hom_dofs
            Target[j] = 0
        end    

        @debug "Hom-DBnd = $HomDirichletBoundaryRegions (ndofs = $(length(hom_dofs)))"
    end

    # BEST-APPROXIMATION DIRICHLET BOUNDARY
    BADirichletBoundaryRegions = get(O.regions4boundarytype,BestapproxDirichletBoundary,[])
    if length(BADirichletBoundaryRegions) > 0

        # find Dirichlet dofs
        BAdofs = []
        for bface = 1 : nbfaces
            for r = 1 : length(BADirichletBoundaryRegions)
                if xBFaceRegions[bface] == BADirichletBoundaryRegions[r]
                    append!(BAdofs,xBFaceDofs[:,bface])
                    break
                end    
            end    
        end
        BAdofs = Base.unique(BAdofs)

        @debug "BA-DBnd = $BADirichletBoundaryRegions (ndofs = $(length(BAdofs)))"

        bonus_quadorder = maximum(O.quadorder4bregion[BADirichletBoundaryRegions[:]])
        FEType = eltype(FE)
        Dboperator = DefaultDirichletBoundaryOperator4FE(FEType)
        b = zeros(Float64,FE.ndofs,1)
        A = FEMatrix{Float64}("MassMatrixBnd", FE)

        if Dboperator == Identity
            function bnd_rhs_function_h1()
                temp = zeros(Float64,ncomponents)
                function closure(result, input, x, region)
                    eval!(temp, O.data4bregion[region], x, time)
                    result[1] = 0.0
                    for j = 1 : ncomponents
                        result[1] += temp[j]*input[j] 
                    end 
                end   
            end   
            action_kernel = ActionKernel(bnd_rhs_function_h1(), [1, ncomponents]; dependencies = "XR", quadorder = bonus_quadorder)
            RHS_bnd = LinearForm(Float64, ON_BFACES, [FE], [Dboperator], Action(Float64, action_kernel); regions = BADirichletBoundaryRegions, name = "RHS bnd data bestapprox")
            assemble!(b, RHS_bnd)
            L2ProductBnd = SymmetricBilinearForm(Float64, ON_BFACES, [FE, FE], [Dboperator, Dboperator], DoNotChangeAction(ncomponents); regions = BADirichletBoundaryRegions, name = "LHS bnd data bestapprox")    
            assemble!(A[1],L2ProductBnd)
        elseif Dboperator == NormalFlux
            xFaceNormals = FE.xgrid[FaceNormals]
            function bnd_rhs_function_hdiv()
                temp = zeros(Float64,ncomponents)
                function closure(result, input, x, region, bface)
                    eval!(temp, O.data4bregion[region], x, time)
                    result[1] = 0.0
                    for j = 1 : ncomponents
                        result[1] += temp[j] * xFaceNormals[j,xBFaces[bface]]
                    end 
                    result[1] *= input[1] 
                end   
            end   
            action_kernel = ActionKernel(bnd_rhs_function_hdiv(), [1, ncomponents]; dependencies = "XRI", quadorder = bonus_quadorder)
            RHS_bnd = LinearForm(Float64, ON_BFACES, [FE], [Dboperator], Action(Float64, action_kernel); regions = BADirichletBoundaryRegions, name = "RHS bnd data NormalFlux bestapprox")
            assemble!(b, RHS_bnd)
            L2ProductBnd = SymmetricBilinearForm(Float64, ON_BFACES, [FE, FE], [Dboperator, Dboperator], DoNotChangeAction(1); regions = BADirichletBoundaryRegions, name = "LHS bnd data NormalFlux bestapprox")    
            assemble!(A[1],L2ProductBnd)
        elseif Dboperator == TangentFlux && xdim == 2 # Hcurl on 2D domains
            xFaceNormals = FE.xgrid[FaceNormals]
            function bnd_rhs_function_hcurl2d()
                temp = zeros(Float64,ncomponents)
                function closure(result, input, x, region, bface)
                    eval!(temp, O.data4bregion[region], x, time)
                    result[1] = -temp[1] * xFaceNormals[2,xBFaces[bface]]
                    result[1] += temp[2] * xFaceNormals[1,xBFaces[bface]]
                    result[1] *= input[1] 
                end   
            end   
            action_kernel = ActionKernel(bnd_rhs_function_hcurl2d(), [1, ncomponents]; dependencies = "XRI", quadorder = bonus_quadorder)
            RHS_bnd = LinearForm(Float64, ON_BFACES, [FE], [Dboperator], Action(Float64, action_kernel); regions = BADirichletBoundaryRegions, name = "RHS bnd data TangentFlux bestapprox")
            assemble!(b, RHS_bnd)
            L2ProductBnd = SymmetricBilinearForm(Float64, ON_BFACES, [FE, FE], [Dboperator, Dboperator], DoNotChangeAction(1); regions = BADirichletBoundaryRegions, name = "LHS bnd data TangentFlux bestapprox")    
            assemble!(A[1],L2ProductBnd)
        elseif Dboperator == TangentFlux && xdim == 3 # Hcurl on 3D domains, does not work properly yet
            @warn "Hcurl boundary data in 3D may not work properly yet"
            xEdgeTangents = FE.xgrid[EdgeTangents]
            xBEdgeRegions = FE.xgrid[BEdgeRegions]
            xBEdges = FE.xgrid[BEdges]

            function bnd_rhs_function_hcurl3d()
                temp = zeros(Float64,ncomponents)
                fixed_region::Int = 1
                function closure(result, input, x, region, bedge)
                    eval!(temp, O.data4bregion[fixed_region], x, time)
                    result[1] = temp[1] * xEdgeTangents[1,xBEdges[bedge]]
                    result[1] += temp[2] * xEdgeTangents[2,xBEdges[bedge]]
                    result[1] += temp[3] * xEdgeTangents[3,xBEdges[bedge]]
                    result[1] *= input[1]
                end   
            end   
            action_kernel = ActionKernel(bnd_rhs_function_hcurl3d(), [1, ncomponents]; dependencies = "XRI", quadorder = bonus_quadorder)
            RHS_bnd = LinearForm(Float64, ON_BEDGES, [FE], [Dboperator], Action(Float64, action_kernel); regions = [0], name = "RHS bnd data TangentFlux bestapprox")
            assemble!(b, RHS_bnd)
            L2ProductBnd = SymmetricBilinearForm(Float64, ON_BEDGES, [FE, FE], [Dboperator, Dboperator], DoNotChangeAction(1); regions = [0], name = "LHS bnd data TangentFlux bestapprox")    
            assemble!(A[1],L2ProductBnd)
        end    

        # fix already set dofs by other boundary conditions
        for j in fixed_dofs
            A[1][j,j] = fixed_penalty
            b[j] = Target[j]*fixed_penalty
        end

        flush!(A.entries)
        # add new fixed dofs from best approximation boundary
        append!(fixed_dofs,BAdofs)
        fixed_dofs = Base.unique(fixed_dofs)

        # solve best approximation problem on boundary and write into Target
        @debug "solving for best-approximation boundary data"

        if (true) # compress matrix by removing all dofs in the interior
            dof2sparsedof = zeros(Int32,FE.ndofs)
            newcolptr = zeros(Int32,0)
            newrowval = zeros(Int32,0)
            dof = 0
            diff = 0
            for j = 1 : FE.ndofs
                diff = A.entries.cscmatrix.colptr[j] != A.entries.cscmatrix.colptr[j+1]
                if diff > 0
                    dof += 1
                    dof2sparsedof[j] = dof
                end
            end

            smallb = zeros(Float64,dof)
            sparsedof2dof = zeros(Int32,dof)
            for j = 1 : FE.ndofs
                if dof2sparsedof[j] > 0
                    push!(newcolptr,A.entries.cscmatrix.colptr[j])
                    append!(newrowval,dof2sparsedof[A.entries.cscmatrix.rowval[A.entries.cscmatrix.colptr[j]:A.entries.cscmatrix.colptr[j+1]-1]])
                    smallb[dof2sparsedof[j]] = b[j]
                    sparsedof2dof[dof2sparsedof[j]] = j
                end
            end
            push!(newcolptr,A.entries.cscmatrix.colptr[end])
            A.entries.cscmatrix = SparseMatrixCSC{Float64,Int32}(dof,dof,newcolptr,newrowval,A.entries.cscmatrix.nzval)

            try
                Target[sparsedof2dof] = SparseArrays.SparseMatrixCSC{Float64,Int64}(A.entries)\smallb
            catch
                Target[sparsedof2dof] = A.entries\smallb
            end
        else # old way: penalize all interior dofs

            for j in setdiff(1:FE.ndofs,fixed_dofs)
                A[1][j,j] = fixed_penalty
                b[j] = 0
            end

            Target[fixed_dofs] = (A.entries\b[:,1])[fixed_dofs]
        end


    end
    
    return fixed_dofs
end    



