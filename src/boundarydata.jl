
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
    dirichlet_penalty::Float64 = 1e60,
    verbosity::Int = 0)
    fixed_dofs = []
  
    FE = Target.FES
    xdim = size(FE.xgrid[Coordinates],1) 
    FEType = eltype(FE)
    ncomponents::Int = get_ncomponents(FEType)
    xBFaceDofs = FE.dofmaps[BFaceDofs]
    xBFaces = FE.xgrid[BFaces]
    nbfaces = num_sources(xBFaceDofs)
    xBFaceRegions = FE.xgrid[BFaceRegions]

    ######################
    # Dirichlet boundary #
    ######################

    # INTERPOLATION DIRICHLET BOUNDARY
    InterDirichletBoundaryRegions = get(O.regions4boundarytype,InterpolateDirichletBoundary,[])
    if length(InterDirichletBoundaryRegions) > 0

        # find Dirichlet dofs
        for r = 1 : length(InterDirichletBoundaryRegions)
            bregiondofs = []
            ibfaces = []
            for bface = 1 : nbfaces
                if xBFaceRegions[bface] == InterDirichletBoundaryRegions[r]
                    append!(ibfaces,xBFaces[bface])
                    append!(bregiondofs,xBFaceDofs[:,bface])
                end
            end    
            bregiondofs = Base.unique(bregiondofs)
            bregion = InterDirichletBoundaryRegions[r]
            append!(fixed_dofs,bregiondofs)
            interpolate!(Target, ON_FACES, O.data4bregion[bregion]; items = ibfaces, time = time)
        end   

        if verbosity > 0
            println("   Int-DBnd = $InterDirichletBoundaryRegions (ndofs = $(length(fixed_dofs)))")
        end    

        
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

        if verbosity > 0
            println("   Hom-DBnd = $HomDirichletBoundaryRegions (ndofs = $(length(hom_dofs)))")
        end    
        
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

        if verbosity > 0
            println("    BA-DBnd = $BADirichletBoundaryRegions (ndofs = $(length(BAdofs)))")
                
        end    

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
            RHS_bnd = LinearForm(Float64, ON_BFACES, FE, Dboperator, Action(Float64, action_kernel); regions = BADirichletBoundaryRegions)
            assemble!(b, RHS_bnd; verbosity = verbosity - 1)
            L2ProductBnd = SymmetricBilinearForm(Float64, ON_BFACES, FE, Dboperator, DoNotChangeAction(ncomponents); regions = BADirichletBoundaryRegions)    
            assemble!(A[1],L2ProductBnd; verbosity = verbosity - 1)
        elseif Dboperator == NormalFlux
            xFaceNormals = FE.xgrid[FaceNormals]
            function bnd_rhs_function_hdiv()
                temp = zeros(Float64,ncomponents)
                function closure(result, input, x, bface, region)
                    eval!(temp, O.data4bregion[region], x, time)
                    result[1] = 0.0
                    for j = 1 : ncomponents
                        result[1] += temp[j] * xFaceNormals[j,xBFaces[bface]]
                    end 
                    result[1] *= input[1] 
                end   
            end   
            action_kernel = ActionKernel(bnd_rhs_function_hdiv(), [1, ncomponents]; dependencies = "XRI", quadorder = bonus_quadorder)
            RHS_bnd = LinearForm(Float64, ON_BFACES, FE, Dboperator, Action(Float64, action_kernel); regions = BADirichletBoundaryRegions)
            assemble!(b, RHS_bnd; verbosity = verbosity - 1)
            L2ProductBnd = SymmetricBilinearForm(Float64, ON_BFACES, FE, Dboperator, DoNotChangeAction(1); regions = BADirichletBoundaryRegions)    
            assemble!(A[1],L2ProductBnd; verbosity = verbosity - 1)
        elseif Dboperator == TangentFlux && xdim == 2 # Hcurl on 2D domains
            xFaceNormals = FE.xgrid[FaceNormals]
            function bnd_rhs_function_hcurl2d()
                temp = zeros(Float64,ncomponents)
                function closure(result, input, x, bface, region)
                    eval!(temp, O.data4bregion[region], x, time)
                    result[1] = -temp[1] * xFaceNormals[2,xBFaces[bface]]
                    result[1] += temp[2] * xFaceNormals[1,xBFaces[bface]]
                    result[1] *= input[1] 
                end   
            end   
            action_kernel = ActionKernel(bnd_rhs_function_hcurl2d(), [1, ncomponents]; dependencies = "XRI", quadorder = bonus_quadorder)
            RHS_bnd = LinearForm(Float64, ON_BFACES, FE, Dboperator, Action(Float64, action_kernel); regions = BADirichletBoundaryRegions)
            assemble!(b, RHS_bnd; verbosity = verbosity - 1)
            L2ProductBnd = SymmetricBilinearForm(Float64, ON_BFACES, FE, Dboperator, DoNotChangeAction(1); regions = BADirichletBoundaryRegions)    
            assemble!(A[1],L2ProductBnd; verbosity = verbosity - 1)
        elseif Dboperator == TangentFlux && xdim == 3 # Hcurl on 3D domains
            xEdgeTangents = FE.xgrid[EdgeTangents]
            xBEdgeRegions = FE.xgrid[BEdgeRegions]
            xBEdges = FE.xgrid[BEdges]

            function bnd_rhs_function_hcurl3d()
                temp = zeros(Float64,ncomponents)
                region::Int = 1
                function closure(result, input, x, bedge, region)
                    eval!(temp, O.data4bregion[region], x, time)
                    if region == 0
                        region = 1
                    end
                    O.data4bregion[region](temp,x,time)
                    result[1] = temp[1] * xEdgeTangents[1,xBEdges[bedge]]
                    result[1] += temp[2] * xEdgeTangents[2,xBEdges[bedge]]
                    result[1] += temp[3] * xEdgeTangents[3,xBEdges[bedge]]
                    result[1] *= input[1]
                end   
            end   
            action_kernel = ActionKernel(bnd_rhs_function_hcurl3d(), [1, ncomponents]; dependencies = "XRI", quadorder = bonus_quadorder)
            RHS_bnd = LinearForm(Float64, ON_BFACES, FE, Dboperator, Action(Float64, action_kernel); regions = BADirichletBoundaryRegions)
            assemble!(b, RHS_bnd; verbosity = verbosity - 1)
            L2ProductBnd = SymmetricBilinearForm(Float64, ON_BEDGES, FE, Dboperator, DoNotChangeAction(1); regions = [0])    
            assemble!(A[1],L2ProductBnd; verbosity = verbosity - 1)
        end    

        # fix already set dofs by other boundary conditions
        for j in fixed_dofs
            A[1][j,j] = dirichlet_penalty
            b[j] = Target[j]*dirichlet_penalty
        end

        flush!(A.entries)
        # add new fixed dofs from best approximation boundary
        append!(fixed_dofs,BAdofs)
        fixed_dofs = Base.unique(fixed_dofs)

        # solve best approximation problem on boundary and write into Target

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

            if verbosity > 0
                println("    ...solving")
                try
                    @time Target[sparsedof2dof] = SparseArrays.SparseMatrixCSC{Float64,Int64}(A.entries)\smallb
                catch
                    @time Target[sparsedof2dof] = A.entries\smallb
                end
            else
                try
                    Target[sparsedof2dof] = SparseArrays.SparseMatrixCSC{Float64,Int64}(A.entries)\smallb
                catch
                    Target[sparsedof2dof] = A.entries\smallb
                end
            end
        else # old way: penalize all interior dofs

            for j in setdiff(1:FE.ndofs,fixed_dofs)
                A[1][j,j] = dirichlet_penalty
                b[j] = 0
            end

            if verbosity > 0
                println("    ...solving")
                @time Target[fixed_dofs] = (A.entries\b[:,1])[fixed_dofs]
            else
                Target[fixed_dofs] = (A.entries\b[:,1])[fixed_dofs]
            end
        end


    end
    
    return fixed_dofs
end    



