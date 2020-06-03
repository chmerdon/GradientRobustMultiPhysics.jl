
abstract type AbstractBoundaryType end
abstract type DirichletBoundary <: AbstractBoundaryType end
abstract type BestapproxDirichletBoundary <: DirichletBoundary end
abstract type InterpolateDirichletBoundary <: DirichletBoundary end
abstract type HomogeneousDirichletBoundary <: DirichletBoundary end


########################
### BoundaryOperator ###
########################
#
# collects boundary data for a component of the system and allows to specify a AbstractBoundaryType for each boundary region
# so far only DirichletBoundary types (see above)
#
# later also SymmetryBoundary
#
# Note: NeumannBoundary has to be implemented as a RhsOperator with on_boundary = true
# Note: PeriodicBoundary has to be implemented as a CombineDofs <: AbstractGlobalConstraint
#
struct BoundaryOperator <: AbstractPDEOperator
    regions4boundarytype :: Dict{Type{<:AbstractBoundaryType},Array{Int,1}}
    data4bregion :: Array{Any,1}
    quadorder4bregion :: Array{Int,1}
    xdim :: Int
    ncomponents :: Int
end

function BoundaryOperator(xdim::Int, ncomponents::Int = 1)
    regions4boundarytype = Dict{Type{<:AbstractBoundaryType},Array{Int,1}}()
    quadorder4bregion = zeros(Int,0)
    return BoundaryOperator(regions4boundarytype, [], quadorder4bregion, xdim, ncomponents)
end

function BoundaryOperator(boundarytype4bregion::Array{DataType,1}, data4region, xdim::Int, ncomponents::Int = 1; bonus_quadorder::Int = 0)
    regions4boundarytype = Dict{Type{<:AbstractBoundaryType},Array{Int,1}}()
    quadorder4bregion = ones(Int,length(boundarytype4bregion))*bonus_quadorder
    for j = 1 : length(boundarytype4bregion)
        btype = boundarytype4bregion[j]
        regions4boundarytype[btype]=push!(get(regions4boundarytype, btype, []),j)
        
    end
    return BoundaryOperator(regions4boundarytype, data4region, quadorder4bregion, xdim, ncomponents)
end

function Base.append!(O::BoundaryOperator,region::Int, btype::Type{<:AbstractBoundaryType}; data = Nothing, bonus_quadorder::Int = 0)
    O.regions4boundarytype[btype]=push!(get(O.regions4boundarytype, btype, []),region)
    while length(O.data4bregion) < region
        push!(O.data4bregion, Nothing)
    end
    while length(O.quadorder4bregion) < region
        push!(O.quadorder4bregion, 0)
    end
    O.quadorder4bregion[region] = bonus_quadorder
    O.data4bregion[region] = data
end


function Base.append!(O::BoundaryOperator,regions::Array{Int,1}, btype::Type{<:AbstractBoundaryType}; data = Nothing, bonus_quadorder::Int = 0)
    for j = 1 : length(regions)
        append!(O,regions[j], btype; data = data, bonus_quadorder = bonus_quadorder)
    end
end



# this function assembles all boundary data at once
# first all interpolation Dirichlet boundaries are assembled
# then all hoomogeneous Dirichlet boundaries are set to zero
# then all DirichletBestapprox boundaries are handled (previous data is fixed)
function boundarydata!(
    Target::FEVectorBlock,
    O::BoundaryOperator;
    dirichlet_penalty::Float64 = 1e60,
    verbosity::Int = 0)
    fixed_dofs = []
  
    FE = Target.FES
    xdim = size(FE.xgrid[Coordinates],1) 
    FEType = eltype(typeof(FE))
    ncomponents::Int = FiniteElements.get_ncomponents(FEType)
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
            interpolate!(Target, O.data4bregion[InterDirichletBoundaryRegions[r]]; dofs = bregiondofs)
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
        FEType = eltype(typeof(FE))
        Dboperator = DefaultDirichletBoundaryOperator4FE(FEType)
        b = zeros(Float64,FE.ndofs,1)
        A = FEMatrix{Float64}("MassMatrixBnd", FE)

        if Dboperator == Identity
            function bnd_rhs_function_h1()
                temp = zeros(Float64,ncomponents)
                function closure(result, input, x, region)
                    O.data4bregion[region](temp,x)
                    result[1] = 0.0
                    for j = 1 : ncomponents
                        result[1] += temp[j]*input[j] 
                    end 
                end   
            end   
            action = RegionWiseXFunctionAction(bnd_rhs_function_h1(),1,xdim; bonus_quadorder = bonus_quadorder)
            RHS_bnd = LinearForm(Float64, AbstractAssemblyTypeBFACE, FE, Dboperator, action; regions = BADirichletBoundaryRegions)
            FEAssembly.assemble!(b, RHS_bnd; verbosity = verbosity - 1)
            L2ProductBnd = SymmetricBilinearForm(Float64, AbstractAssemblyTypeBFACE, FE, Dboperator, DoNotChangeAction(ncomponents); regions = BADirichletBoundaryRegions)    
            FEAssembly.assemble!(A[1],L2ProductBnd; verbosity = verbosity - 1)
        elseif Dboperator == NormalFlux
            xFaceNormals = FE.xgrid[FaceNormals]
            xBFaces = FE.xgrid[BFaces]
            function bnd_rhs_function_hdiv()
                temp = zeros(Float64,ncomponents)
                function closure(result, input, x, bface)
                    O.data4bregion[xBFaceRegions[bface]](temp,x)
                    result[1] = 0.0
                    for j = 1 : ncomponents
                        result[1] += temp[j] * xFaceNormals[j,xBFaces[bface]]
                    end 
                    result[1] *= input[1] 
                end   
            end   
            action = ItemWiseXFunctionAction(bnd_rhs_function_hdiv(),1,xdim; bonus_quadorder = bonus_quadorder)
            RHS_bnd = LinearForm(Float64, AbstractAssemblyTypeBFACE, FE, Dboperator, action; regions = BADirichletBoundaryRegions)
            FEAssembly.assemble!(b, RHS_bnd; verbosity = verbosity - 1)
            L2ProductBnd = SymmetricBilinearForm(Float64, AbstractAssemblyTypeBFACE, FE, Dboperator, DoNotChangeAction(1); regions = BADirichletBoundaryRegions)    
            FEAssembly.assemble!(A[1],L2ProductBnd; verbosity = verbosity - 1)
        end    

        # fix already set dofs by other boundary conditions
        for j in fixed_dofs
            A[1][j,j] = dirichlet_penalty
            b[j] = Target[j]*dirichlet_penalty
        end
        append!(fixed_dofs,BAdofs)
        fixed_dofs = Base.unique(fixed_dofs)

        # solve best approximation problem on boundary and write into Target
        Target[fixed_dofs] = A.entries[fixed_dofs,fixed_dofs]\b[fixed_dofs,1]
    end
    
    return fixed_dofs
end    



