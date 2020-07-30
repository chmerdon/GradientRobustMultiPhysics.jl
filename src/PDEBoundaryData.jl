
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
    xdim :: Int
    ncomponents :: Int
end

function BoundaryOperator(xdim::Int, ncomponents::Int = 1)
    regions4boundarytype = Dict{Type{<:AbstractBoundaryType},Array{Int,1}}()
    quadorder4bregion = zeros(Int,0)
    timedependent = Array{Bool,1}([])
    return BoundaryOperator(regions4boundarytype, [], timedependent, quadorder4bregion, xdim, ncomponents)
end

function Base.append!(O::BoundaryOperator,region::Int, btype::Type{<:AbstractBoundaryType}; timedependent::Bool = false, data = Nothing, bonus_quadorder::Int = 0)
    O.regions4boundarytype[btype]=push!(get(O.regions4boundarytype, btype, []),region)
    while length(O.data4bregion) < region
        push!(O.data4bregion, Nothing)
    end
    while length(O.quadorder4bregion) < region
        push!(O.quadorder4bregion, 0)
        push!(O.timedependent, false)
    end
    O.quadorder4bregion[region] = bonus_quadorder
    O.data4bregion[region] = data
    O.timedependent[region] = timedependent
end


function Base.append!(O::BoundaryOperator,regions::Array{Int,1}, btype::Type{<:AbstractBoundaryType}; timedependent::Bool = false, data = Nothing, bonus_quadorder::Int = 0)
    for j = 1 : length(regions)
        append!(O,regions[j], btype; timedependent = timedependent, data = data, bonus_quadorder = bonus_quadorder)
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
    xBFaces = FE.xgrid[BFaces]
    xBFaceDofs = FE.BFaceDofs
    nbfaces = length(xBFaces)
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
            for bface = 1 : nbfaces
                if xBFaceRegions[bface] == InterDirichletBoundaryRegions[r]
                    append!(bregiondofs,xBFaceDofs[:,bface])
                end
            end    
            bregiondofs = Base.unique(bregiondofs)
            bregion = InterDirichletBoundaryRegions[r]
            if O.timedependent[bregion] == true
                    function bregion_data_at_time(result, input)
                        O.data4bregion[bregion](temp,x,time)
                    end   
                interpolate!(Target, bregion_data_at_time; dofs = bregiondofs)
            else
                interpolate!(Target, O.data4bregion[bregion]; dofs = bregiondofs)
            end
            append!(fixed_dofs,bregiondofs)
            bregiondofs = []
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
                    if O.timedependent[region] == true
                        O.data4bregion[region](temp,x,time)
                    else
                        O.data4bregion[region](temp,x)
                    end
                    result[1] = 0.0
                    for j = 1 : ncomponents
                        result[1] += temp[j]*input[j] 
                    end 
                end   
            end   
            action = RegionWiseXFunctionAction(bnd_rhs_function_h1(),1,xdim; bonus_quadorder = bonus_quadorder)
            RHS_bnd = LinearForm(Float64, AssemblyTypeBFACE, FE, Dboperator, action; regions = BADirichletBoundaryRegions)
            assemble!(b, RHS_bnd; verbosity = verbosity - 1)
            L2ProductBnd = SymmetricBilinearForm(Float64, AssemblyTypeBFACE, FE, Dboperator, DoNotChangeAction(ncomponents); regions = BADirichletBoundaryRegions)    
            assemble!(A[1],L2ProductBnd; verbosity = verbosity - 1)
        elseif Dboperator == NormalFlux
            xFaceNormals = FE.xgrid[FaceNormals]
            xBFaces = FE.xgrid[BFaces]
            function bnd_rhs_function_hdiv()
                temp = zeros(Float64,ncomponents)
                function closure(result, input, x, bface)
                    if O.timedependent[xBFaceRegions[bface]] == true
                        O.data4bregion[xBFaceRegions[bface]](temp,x,time)
                    else
                        O.data4bregion[xBFaceRegions[bface]](temp,x)
                    end
                    result[1] = 0.0
                    for j = 1 : ncomponents
                        result[1] += temp[j] * xFaceNormals[j,xBFaces[bface]]
                    end 
                    result[1] *= input[1] 
                end   
            end   
            action = ItemWiseXFunctionAction(bnd_rhs_function_hdiv(),1,xdim; bonus_quadorder = bonus_quadorder)
            RHS_bnd = LinearForm(Float64, AssemblyTypeBFACE, FE, Dboperator, action; regions = BADirichletBoundaryRegions)
            assemble!(b, RHS_bnd; verbosity = verbosity - 1)
            L2ProductBnd = SymmetricBilinearForm(Float64, AssemblyTypeBFACE, FE, Dboperator, DoNotChangeAction(1); regions = BADirichletBoundaryRegions)    
            assemble!(A[1],L2ProductBnd; verbosity = verbosity - 1)
        end    

        # fix already set dofs by other boundary conditions
        for j in fixed_dofs
            A[1][j,j] = dirichlet_penalty
            b[j] = Target[j]*dirichlet_penalty
        end

        # add new fixed dofs from best approximation boundary
        append!(fixed_dofs,BAdofs)
        fixed_dofs = Base.unique(fixed_dofs)

        # solve best approximation problem on boundary and write into Target
        # the uncommented line below is very slow (possibly because dense matrix is extracted and solved)
        #   Target[fixed_dofs] = A.entries[fixed_dofs,fixed_dofs]\b[fixed_dofs,1]
        # so instead we solve for all dofs but fix all non-involved dofs in the FE space

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
    
    return fixed_dofs
end    



