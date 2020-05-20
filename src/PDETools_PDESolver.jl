
function assemble!(A::FEMatrixBlock, O::ReactionOperator; verbosity::Int = 0)
    FE1 = A.FETypeX
    FE2 = A.FETypeY
    @assert FE1 == FE2
    L2Product = SymmetricBilinearForm(Float64,AbstractAssemblyTypeCELL, FE1, Identity, O.action)    
    FEAssembly.assemble!(A, L2Product; verbosity = verbosity)
end

function assemble!(A::FEMatrixBlock, O::LaplaceOperator; verbosity::Int = 0)
    FE1 = A.FETypeX
    FE2 = A.FETypeY
    @assert FE1 == FE2
    H1Product = SymmetricBilinearForm(Float64, AbstractAssemblyTypeCELL, FE1, Gradient, O.action)    
    FEAssembly.assemble!(A, H1Product; verbosity = verbosity)
end

function assemble!(A::FEMatrixBlock, O::ConvectionOperator; verbosity::Int = 0)
    FE1 = A.FETypeX
    FE2 = A.FETypeY
    ConvectionForm = BilinearForm(Float64, AbstractAssemblyTypeCELL, FE1, FE2, Gradient, Identity, O.action)  
    FEAssembly.assemble!(A, ConvectionForm; verbosity = verbosity)
end

function assemble!(A::FEMatrixBlock, O::LagrangeMultiplier; verbosity::Int = 0, At::FEMatrixBlock)
    FE1 = A.FETypeX
    FE2 = A.FETypeY
    @assert At.FETypeX == FE2
    @assert At.FETypeY == FE1
    DivPressure = BilinearForm(Float64, AbstractAssemblyTypeCELL, FE1, FE2, O.operator, Identity, MultiplyScalarAction(-1.0,1))   
    FEAssembly.assemble!(At, DivPressure; verbosity = verbosity, transpose_copy = A)
end

function assemble!(b::FEVectorBlock, O::RhsOperator; verbosity::Int = 0)
    FE = b.FEType
    RHS = LinearForm(Float64,AbstractAssemblyTypeCELL, FE, O.operator, O.action)
    FEAssembly.assemble!(b, RHS; verbosity = verbosity)
end

function assemble!(PDE::PDEDescription, FE::Array{<:AbstractFiniteElement,1}; verbosity::Int = 0)

    A = FEMatrix{Float64}("SystemMatrix", FE)
    for j = 1 : length(FE), k = 1 : length(FE), o = 1 : length(PDE.LHSOperators[j,k])
        PDEoperator = PDE.LHSOperators[j,k][o]
        if verbosity > 0
            println("\n  Assembling into matrix block[$j,$k]: $(typeof(PDEoperator))")
            if typeof(PDEoperator) == LagrangeMultiplier
                @time assemble!(A[j,k], PDEoperator; verbosity = verbosity, At = A[k,j])
            else
                @time assemble!(A[j,k], PDEoperator; verbosity = verbosity)
            end    
        else
            if typeof(PDEoperator) == LagrangeMultiplier
                assemble!(A[j,k], PDEoperator; verbosity = verbosity, At = A[k,j])
            else
                assemble!(A[j,k], PDEoperator; verbosity = verbosity)
            end    
        end    
    end

    b = FEVector{Float64}("SystemRhs", FE)
    for j = 1 : length(FE), o = 1 : length(PDE.RHSOperators[j])
        if verbosity > 0
            println("\n  Assembling into rhs block [$j]: $(typeof(PDE.RHSOperators[j][o])) ($(PDE.RHSOperators[j][o].operator))")
            @time assemble!(b[j], PDE.RHSOperators[j][o]; verbosity = verbosity)
        else
            assemble!(b[j], PDE.RHSOperators[j][o]; verbosity = verbosity)
        end    
    end

    return A, b
end


function boundarydata!(
    Target::FEVectorBlock,
    O::BoundaryOperator;
    dirichlet_penalty::Float64 = 1e60,
    verbosity::Int = 0)

    FE = Target.FEType
    xdim = size(FE.xgrid[Coordinates],1) 
    ncomponents = get_ncomponents(typeof(FE))
    xBFaces = FE.xgrid[BFaces]
    xBFaceDofs = FE.BFaceDofs
    nbfaces = length(xBFaces)
    xBFaceRegions = FE.xgrid[BFaceRegions]


    ######################
    # Dirichlet boundary #
    ######################
    fixed_bdofs = []

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
            append!(fixed_bdofs,bregiondofs)
            bregiondofs = []
        end   

        if verbosity > 0
            println("   Int-DBnd = $InterDirichletBoundaryRegions (ndofs = $(length(fixed_bdofs)))")
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
        append!(fixed_bdofs,hom_dofs)
        fixed_bdofs = Base.unique(fixed_bdofs)
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
        Dboperator = DefaultDirichletBoundaryOperator4FE(typeof(FE))
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
            action = ItemWiseXFunctionAction(bnd_rhs_function_hdiv(),1,1; bonus_quadorder = bonus_quadorder)
            RHS_bnd = LinearForm(Float64, AbstractAssemblyTypeBFACE, FE, Dboperator, action; regions = BADirichletBoundaryRegions)
            FEAssembly.assemble!(b, RHS_bnd; verbosity = verbosity - 1)
            L2ProductBnd = SymmetricBilinearForm(Float64, AbstractAssemblyTypeBFACE, FE, Dboperator, DoNotChangeAction(1); regions = BADirichletBoundaryRegions)    
            FEAssembly.assemble!(A[1],L2ProductBnd; verbosity = verbosity - 1)
        end    

        # fix already set dofs by other boundary conditions
        for j in fixed_bdofs
            A[1][j,j] = dirichlet_penalty
            b[j] = Target[j]*dirichlet_penalty
        end
        append!(fixed_bdofs,BAdofs)
        fixed_bdofs = Base.unique(fixed_bdofs)

        # solve best approximation problem on boundary and write into Target
        Target[fixed_bdofs] = A.entries[fixed_bdofs,fixed_bdofs]\b[fixed_bdofs,1]
    end
    
    return fixed_bdofs
end    



function solve!(
    Target::FEVector,
    PDE::PDEDescription;
    dirichlet_penalty = 1e60,
    verbosity::Int = 0)
    FEs = Array{AbstractFiniteElement,1}([])
    for j=1 : length(Target.FEVectorBlocks)
        push!(FEs,Target.FEVectorBlocks[j].FEType)
    end    


    if verbosity > 0
        println("\nSOLVING PDE")
        println("===========")
        println("  name = $(PDE.name)")
        print("   FEs = ")
        for j = 1 : length(Target)
            println("$(Target[j].FEType.name) (ndofs = $(length(Target.entries)))\n         ");
        end
    end

    # ASSEMBLE SYSTEM
    A,b = assemble!(PDE,FEs; verbosity = verbosity - 1)

    # ASSEMBLE BOUNDARY DATA
    fixed_bdofs = []
    for j= 1 : length(Target.FEVectorBlocks)
        if verbosity > 1
            println("\n  Assembling boundary data for block [$j]...")
            @time new_fixed_dofs = boundarydata!(Target[j],PDE.BoundaryOperators[j]; verbosity = verbosity - 1)
            append!(fixed_bdofs, new_fixed_dofs)
        else
            new_fixed_dofs = boundarydata!(Target[j],PDE.BoundaryOperators[j]; verbosity = verbosity - 1)
            append!(fixed_bdofs, new_fixed_dofs)
        end    
    end    

    # penalize fixed dofs
    for j = 1 : length(fixed_bdofs)
        b.entries[fixed_bdofs[j]] = dirichlet_penalty * Target.entries[fixed_bdofs[j]]
        A[1][fixed_bdofs[j],fixed_bdofs[j]] = dirichlet_penalty
    end

    # prepare further global constraints
    for j= 1 : length(FEs), k = 1 : length(PDE.GlobalConstraints[j])
        if typeof(PDE.GlobalConstraints[j][k]) == FixedIntegralMean
            if verbosity > 1
                println("\n  Ensuring fixed integral mean for component $j...")
            end
            b[j][1] = 0.0
            A[j,j][1,1] = dirichlet_penalty
        end
    end

    # solve
    if verbosity > 1
        println("\n  Solving")
        @time Target.entries[:] = A.entries\b.entries
    else
        Target.entries[:] = A.entries\b.entries
    end

    # realize global constraints

    for j= 1 : length(FEs), k = 1 : length(PDE.GlobalConstraints[j])
        if typeof(PDE.GlobalConstraints[j][k]) == FixedIntegralMean
            if verbosity > 0
                println("\n  Moving integral mean for component $j to value $(PDE.GlobalConstraints[j][k].value)")
            end
            # move integral mean
            pmeanIntegrator = ItemIntegrator{Float64,AbstractAssemblyTypeCELL}(Identity, DoNotChangeAction(1), [0])
            meanvalue =  evaluate(pmeanIntegrator,Target[j]; verbosity = verbosity - 1)
            total_area = sum(FEs[j].xgrid[CellVolumes], dims=1)[1]
            meanvalue /= total_area
            for dof=1:FEs[j].ndofs
                Target[j][dof] -= meanvalue + PDE.GlobalConstraints[j][k].value
            end    
        end
    end
end


