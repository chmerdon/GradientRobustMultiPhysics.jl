
mutable struct SolverConfig
    is_nonlinear::Bool      # PDE is nonlinear
    is_timedependent::Bool  # PDE is time_dependent
    LHS_AssemblyTriggers::Array{DataType,2} # assembly triggers for blocks in LHS
    RHS_AssemblyTriggers::Array{DataType,1} # assembly triggers for blocks in RHS
    maxIterations::Int          # maximum number of iterations
    maxResidual::Real           # tolerance for residual
    current_time::Real          # current time in a time-dependent setting
    dirichlet_penalty::Real     # penalty for Dirichlet data
end


# check if PDE is nonlinear or time-dependent and which blocks require recalculation
# and devise some initial solver strategy
function generate_solver(PDE::PDEDescription)
    nonlinear::Bool = false
    timedependent::Bool = false
    block_nonlinear::Bool = false
    block_timedependent::Bool = false
    op_nonlinear::Bool = false
    op_timedependent::Bool = false
    LHS_ATs = Array{DataType,2}(undef,size(PDE.LHSOperators,1),size(PDE.LHSOperators,2))
    for j = 1 : size(PDE.LHSOperators,1), k = 1 : size(PDE.LHSOperators,2)
        block_nonlinear = false
        block_timedependent = false
        for o = 1 : length(PDE.LHSOperators[j,k])
            op_nonlinear, op_timedependent = check_PDEoperator(PDE.LHSOperators[j,k][o])
            if op_nonlinear == true
                block_nonlinear = true
            end
            if op_timedependent == true
                block_timedependent = true
            end
        end
        LHS_ATs[j,k] = AssemblyInitial
        if block_timedependent== true
            timedependent = true
            LHS_ATs[j,k] = AssemblyEachTimeStep
        end
        if block_nonlinear == true
            nonlinear = true
            LHS_ATs[j,k] = AssemblyAlways
        end
    end
    RHS_ATs = Array{DataType,1}(undef,size(PDE.RHSOperators,1))
    for j = 1 : size(PDE.RHSOperators,1)
        block_nonlinear = false
        block_timedependent = false
        for o = 1 : length(PDE.RHSOperators[j])
            op_nonlinear, op_timedependent = check_PDEoperator(PDE.RHSOperators[j][o])
            if op_nonlinear == true
                block_nonlinear = true
            end
            if op_timedependent== true
                block_timedependent = true
            end
        end
        RHS_ATs[j] = AssemblyInitial
        if block_timedependent== true
            timedependent = true
            RHS_ATs[j] = AssemblyEachTimeStep
        end
        if block_nonlinear == true
            nonlinear = true
            RHS_ATs[j] = AssemblyAlways
        end
    end
    return SolverConfig(nonlinear, timedependent, LHS_ATs, RHS_ATs, 10, 1e-10, 0.0, 1e60)
end

function Base.show(io::IO, SC::SolverConfig)

    println("\nSOLVER-CONFIGURATION")
    println("======================")
    println("         nonlinear = $(SC.is_nonlinear)")
    println("     timedependent = $(SC.is_timedependent)")

    println("  AssemblyTriggers = ")
    for j = 1 : size(SC.LHS_AssemblyTriggers,1)
        print("         LHS_AT[$j] : ")
        for k = 1 : size(SC.LHS_AssemblyTriggers,2)
            if SC.LHS_AssemblyTriggers[j,k] == AssemblyInitial
                print(" I ")
            elseif SC.LHS_AssemblyTriggers[j,k] == AssemblyAlways
                print(" A ")
            elseif SC.LHS_AssemblyTriggers[j,k] == AssemblyEachTimeStep
                print(" T ")
            end
        end
        println("")
    end

    for j = 1 : size(SC.RHS_AssemblyTriggers,1)
        print("         RHS_AT[$j] : ")
        if SC.RHS_AssemblyTriggers[j] == AssemblyInitial
            print(" I ")
        elseif SC.RHS_AssemblyTriggers[j] == AssemblyAlways
            print(" A ")
        elseif SC.RHS_AssemblyTriggers[j] == AssemblyEachTimeStep
            print(" T ")
        end
        println("")
    end
    println("                     (I = Once, T = EachTimeStep, A = Always)")

end









function assemble!(A::FEMatrixBlock, CurrentSolution::FEVector, O::ReactionOperator; verbosity::Int = 0)
    FE1 = A.FESX
    FE2 = A.FESY
    @assert FE1 == FE2
    L2Product = SymmetricBilinearForm(Float64,AbstractAssemblyTypeCELL, FE1, Identity, O.action; regions = O.regions)    
    FEAssembly.assemble!(A, L2Product; verbosity = verbosity)
end


function assemble!(A::FEMatrixBlock, CurrentSolution::FEVector, O::DiagonalOperator; verbosity::Int = 0)
    FE1 = A.FESX
    FE2 = A.FESY
    @assert FE1 == FE2
    xCellDofs = FE1.CellDofs
    xCellRegions = FE1.xgrid[CellRegions]
    ncells = num_sources(xCellDofs)
    dof::Int = 0
    for item = 1 : ncells
        for r = 1 : length(O.regions)
            # check if item region is in regions
            if xCellRegions[item] == O.regions[r]
                for k = 1 : num_targets(xCellDofs,item)
                    dof = xCellDofs[k,item]
                    if O.onlynz == true
                        if A[dof,dof] == 0
                            #println(" PEN dof=$dof")
                            A[dof,dof] = O.value
                        end
                    else
                        A[dof,dof] = O.value
                    end    
                end
            end
        end
    end
end


function assemble!(A::FEMatrixBlock, CurrentSolution::FEVector, O::StiffnessOperator; verbosity::Int = 0)
    FE1 = A.FESX
    FE2 = A.FESY
    @assert FE1 == FE2
    H1Product = SymmetricBilinearForm(Float64, AbstractAssemblyTypeCELL, FE1, O.gradient_operator, O.action; regions = O.regions)    
    FEAssembly.assemble!(A, H1Product; verbosity = verbosity)
end

function assemble!(A::FEMatrixBlock, CurrentSolution::FEVector, O::ConvectionOperator; verbosity::Int = 0)
    if O.beta_from == 0
        FE1 = A.FESX
        FE2 = A.FESY
        ConvectionForm = BilinearForm(Float64, AbstractAssemblyTypeCELL, FE1, FE2, Gradient, O.testfunction_operator, O.action; regions = O.regions)  
        FEAssembly.assemble!(A, ConvectionForm; verbosity = verbosity)
    else
        FE1 = A.FESX
        FE2 = A.FESY
        ConvectionForm = TrilinearForm(Float64, AbstractAssemblyTypeCELL, FE1, FE1, FE2, O.testfunction_operator, Gradient, O.testfunction_operator, O.action; regions = O.regions)  
        FEAssembly.assemble!(A, ConvectionForm, CurrentSolution[O.beta_from]; verbosity = verbosity)
    end
end

function assemble!(A::FEMatrixBlock, CurrentSolution::FEVector, O::LagrangeMultiplier; verbosity::Int = 0, At::FEMatrixBlock)
    FE1 = A.FESX
    FE2 = A.FESY
    @assert At.FESX == FE2
    @assert At.FESY == FE1
    DivPressure = BilinearForm(Float64, AbstractAssemblyTypeCELL, FE1, FE2, O.operator, Identity, MultiplyScalarAction(-1.0,1))   
    FEAssembly.assemble!(At, DivPressure; verbosity = verbosity, transpose_copy = A)
end

function assemble!(b::FEVectorBlock, CurrentSolution::FEVector, O::RhsOperator{AT}; verbosity::Int = 0) where {AT<:AbstractAssemblyType}
    FE = b.FES
    RHS = LinearForm(Float64,AT, FE, O.testfunction_operator, O.action; regions = O.regions)
    FEAssembly.assemble!(b, RHS; verbosity = verbosity)
end

function assemble!(
    A::FEMatrix,
    b::FEVector,
    PDE::PDEDescription,
    SC::SolverConfig,
    CurrentSolution::FEVector;
    equations::Array{Int,1} = [],
    min_trigger::Type{<:AbstractAssemblyTrigger} = AssemblyAlways,
    verbosity::Int = 0)

    if length(equations) == 0
        equations = 1:size(PDE.LHSOperators,1)
    end

    for j in equations, k = 1 : size(PDE.LHSOperators,2)
        if SC.LHS_AssemblyTriggers[j,k] <: min_trigger
            fill!(A[j,k],0.0)
            for o = 1 : length(PDE.LHSOperators[j,k])
                PDEoperator = PDE.LHSOperators[j,k][o]
                if verbosity > 0
                    println("\n  Assembling into matrix block[$j,$k]: $(typeof(PDEoperator))")
                    if typeof(PDEoperator) == LagrangeMultiplier
                        @time assemble!(A[j,k], CurrentSolution, PDEoperator; verbosity = verbosity, At = A[k,j])
                    else
                        @time assemble!(A[j,k], CurrentSolution, PDEoperator; verbosity = verbosity)
                    end    
                else
                    if typeof(PDEoperator) == LagrangeMultiplier
                        assemble!(A[j,k], CurrentSolution, PDEoperator; verbosity = verbosity, At = A[k,j])
                    else
                        assemble!(A[j,k], CurrentSolution, PDEoperator; verbosity = verbosity)
                    end    
                end  
            end  
        end
    end

    for j in equations
        if SC.RHS_AssemblyTriggers[j] <: min_trigger
            for o = 1 : length(PDE.RHSOperators[j])
                if verbosity > 0
                    println("\n  Assembling into rhs block [$j]: $(typeof(PDE.RHSOperators[j][o])) ($(PDE.RHSOperators[j][o].testfunction_operator))")
                    @time assemble!(b[j], CurrentSolution, PDE.RHSOperators[j][o]; verbosity = verbosity)
                else
                    assemble!(b[j], CurrentSolution, PDE.RHSOperators[j][o]; verbosity = verbosity)
                end    
            end
        end
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


# check if operator causes nonlinearity or time-dependence
function check_PDEoperator(O::AbstractPDEOperator)
    return false, false
end
function check_PDEoperator(O::ConvectionOperator)
    return O.beta_from != 0, false
end


function apply_constraints!(
    A::FEMatrix,
    b::FEVector,
    PDE::PDEDescription,
    SC::SolverConfig,
    Target::FEVector;
    verbosity::Int = 0)

    fixed_dofs = []

    for j = 1 : length(PDE.GlobalConstraints)
        if typeof(PDE.GlobalConstraints[j]) == FixedIntegralMean
            c = PDE.GlobalConstraints[j].component
            if verbosity > 0
                println("\n  Ensuring fixed integral mean for component $j...")
            end
            b[c][1] = 0.0
            A[c,c][1,1] = SC.dirichlet_penalty
            push!(fixed_dofs,A[c,c].offsetX+1)
        elseif typeof(PDE.GlobalConstraints[j]) == CombineDofs
            c = PDE.GlobalConstraints[j].componentX
            c2 = PDE.GlobalConstraints[j].componentY
            if verbosity > 0 
                println("\n  Combining dofs of component $c and $c2...")
            end
            # add subblock [dofsY,dofsY] of block [c2,c2] to subblock [dofsX,dofsX] of block [c,c]
            # and penalize dofsY dofs
            rows = rowvals(A.entries.cscmatrix)
            targetrow = 0
            sourcerow = 0
            targetcolumn = 0
            sourcecolumn = 0
            for dof = 1 :length(PDE.GlobalConstraints[j].dofsX)


                targetrow = A[c,c].offsetX + PDE.GlobalConstraints[j].dofsX[dof]
                sourcerow = A[c2,c2].offsetX + PDE.GlobalConstraints[j].dofsY[dof]
                #println("copying sourcerow=$sourcerow to targetrow=$targetrow")
                for dof = 1 : length(PDE.GlobalConstraints[j].dofsX)
                    sourcecolumn = PDE.GlobalConstraints[j].dofsY[dof] + A[c2,c2].offsetY
                    for r in nzrange(A.entries.cscmatrix, sourcecolumn)
                        if sourcerow == rows[r]
                            targetcolumn = PDE.GlobalConstraints[j].dofsX[dof] + A[c,c].offsetY
                            A.entries[targetrow, targetcolumn] += 0.5*A.entries.cscmatrix.nzval[r] 
                        end
                    end
                end
                targetcolumn = A[c,c].offsetY + PDE.GlobalConstraints[j].dofsX[dof]
                sourcecolumn = A[c2,c2].offsetY + PDE.GlobalConstraints[j].dofsY[dof]
                #println("copying sourcecolumn=$sourcecolumn to targetcolumn=$targetcolumn")
                for dof = 1 : length(PDE.GlobalConstraints[j].dofsX)
                    sourcerow = PDE.GlobalConstraints[j].dofsY[dof] + A[c2,c2].offsetX
                    for r in nzrange(A.entries.cscmatrix, sourcecolumn)
                        if sourcerow == rows[r]
                            targetrow = PDE.GlobalConstraints[j].dofsX[dof] + A[c,c].offsetX
                            A.entries[targetrow,targetcolumn] += 0.5*A.entries.cscmatrix.nzval[r] 
                        end
                    end
                end

                # penalize Y dofs
                #println(" PEN dof=$sourcecolumn")
                b.entries[sourcecolumn] = 0.0
                A.entries[sourcecolumn,sourcecolumn] = SC.dirichlet_penalty
                push!(fixed_dofs,sourcecolumn)
            end
        end
    end

    return fixed_dofs
end

function realize_constraints!(
    Target::FEVector,
    PDE::PDEDescription,
    SC::SolverConfig;
    verbosity::Int = 0)

    for j = 1 : length(PDE.GlobalConstraints)
        if typeof(PDE.GlobalConstraints[j]) == FixedIntegralMean
            c = PDE.GlobalConstraints[j].component
            if verbosity > 0
                println("\n  Moving integral mean for component $c to value $(PDE.GlobalConstraints[j][k].value)")
            end
            # move integral mean
            pmeanIntegrator = ItemIntegrator{Float64,AbstractAssemblyTypeCELL}(Identity, DoNotChangeAction(1), [0])
            meanvalue =  evaluate(pmeanIntegrator,Target[c]; verbosity = verbosity - 1)
            total_area = sum(Target.FEVectorBlocks[c].FES.xgrid[CellVolumes], dims=1)[1]
            meanvalue /= total_area
            for dof=1:Target.FEVectorBlocks[c].FES.ndofs
                Target[c][dof] -= meanvalue + PDE.GlobalConstraints[j].value
            end    
        elseif typeof(PDE.GlobalConstraints[j]) == CombineDofs
            c = PDE.GlobalConstraints[j].componentX
            c2 = PDE.GlobalConstraints[j].componentY
            if verbosity > 0
                println("\n  Moving entries of combined dofs from component $c to component $c2")
            end
            for dof = 1 : length(PDE.GlobalConstraints[j].dofsX)
                Target[c2][PDE.GlobalConstraints[j].dofsY[dof]] = Target[c][PDE.GlobalConstraints[j].dofsX[dof]]
            end 
            
        end
    end
end


# for linear, stationary PDEs that can be solved in one step
function solve_direct!(Target::FEVector, PDE::PDEDescription, SC::SolverConfig; verbosity::Int = 0)

    FEs = Array{FESpace,1}([])
    for j=1 : length(Target.FEVectorBlocks)
        push!(FEs,Target.FEVectorBlocks[j].FES)
    end    

    # ASSEMBLE SYSTEM
    A = FEMatrix{Float64}("SystemMatrix", FEs)
    b = FEVector{Float64}("SystemRhs", FEs)
    for j = 1:size(PDE.RHSOperators,1)
        assemble!(A,b,PDE,SC,Target; equations = [j], min_trigger = AssemblyInitial, verbosity = verbosity - 1)
    end

    # ASSEMBLE BOUNDARY DATA
    fixed_dofs = []
    for j= 1 : length(Target.FEVectorBlocks)
        if verbosity > 1
            println("\n  Assembling boundary data for block [$j]...")
            @time new_fixed_dofs = boundarydata!(Target[j],PDE.BoundaryOperators[j]; verbosity = verbosity - 1)
            append!(fixed_dofs, new_fixed_dofs)
        else
            new_fixed_dofs = boundarydata!(Target[j],PDE.BoundaryOperators[j]; verbosity = verbosity - 1)
            append!(fixed_dofs, new_fixed_dofs)
        end    
    end    

    # PENALIZE FIXED DOFS
    # (possibly from boundary conditions)
    for j = 1 : length(fixed_dofs)
        b.entries[fixed_dofs[j]] = SC.dirichlet_penalty * Target.entries[fixed_dofs[j]]
        A[1][fixed_dofs[j],fixed_dofs[j]] = SC.dirichlet_penalty
    end

    # PREPARE GLOBALCONSTRAINTS
    # (possibly more penalties)
    flush!(A.entries)
    apply_constraints!(A,b,PDE,SC,Target; verbosity = verbosity - 1)


    # SOLVE
    if verbosity > 1
        println("\n  Solving")
        @time Target.entries[:] = A.entries\b.entries
    else
        Target.entries[:] = A.entries\b.entries
    end

    if verbosity > 0
        # CHECK RESIDUAL
        residual = (A.entries*Target.entries - b.entries).^2
        residual[fixed_dofs] .= 0
        println("\n  residual = $(sqrt(sum(residual, dims = 1)[1]))")
    end


    # REALIZE GLOBAL GLOBALCONSTRAINTS 
    # (possibly changes some entries of Target)
    realize_constraints!(Target,PDE,SC;verbosity = verbosity - 1)
end




# for nonlinear, stationary PDEs that can be solved by fixpoint iteration
function solve_fixpoint!(Target::FEVector, PDE::PDEDescription, SC::SolverConfig; verbosity::Int = 0)

    FEs = Array{FESpace,1}([])
    for j=1 : length(Target.FEVectorBlocks)
        push!(FEs,Target.FEVectorBlocks[j].FES)
    end    

    # ASSEMBLE SYSTEM INIT
    A = FEMatrix{Float64}("SystemMatrix", FEs)
    b = FEVector{Float64}("SystemRhs", FEs)
    for j = 1:size(PDE.RHSOperators,1)
        assemble!(A,b,PDE,SC,Target; equations = [j], min_trigger = AssemblyInitial, verbosity = verbosity - 2)
    end

    # ASSEMBLE BOUNDARY DATA
    fixed_dofs = []
    for j= 1 : length(Target.FEVectorBlocks)
        if verbosity > 1
            println("\n  Assembling boundary data for block [$j]...")
            @time new_fixed_dofs = boundarydata!(Target[j],PDE.BoundaryOperators[j]; verbosity = verbosity - 2)
            append!(fixed_dofs, new_fixed_dofs)
        else
            new_fixed_dofs = boundarydata!(Target[j],PDE.BoundaryOperators[j]; verbosity = verbosity - 2)
            append!(fixed_dofs, new_fixed_dofs)
        end    
    end    

    residual = zeros(Float64,length(b.entries))
    resnorm::Float64 = 0.0

    if verbosity > 0
        println("\n  starting fixpoint iterations")
    end
    for j = 1 : SC.maxIterations

        # PENALIZE FIXED DOFS
        # (possibly from boundary conditions)
        for j = 1 : length(fixed_dofs)
            b.entries[fixed_dofs[j]] = SC.dirichlet_penalty * Target.entries[fixed_dofs[j]]
            A[1][fixed_dofs[j],fixed_dofs[j]] = SC.dirichlet_penalty
        end

        # PREPARE GLOBALCONSTRAINTS
        # (possibly more penalties)
        flush!(A.entries)
        apply_constraints!(A,b,PDE,SC,Target; verbosity = verbosity - 1)

        # SOLVE
        if verbosity > 1
            println("\n  Solving")
            @time Target.entries[:] = A.entries\b.entries
        else
            Target.entries[:] = A.entries\b.entries
        end

        # REASSEMBLE NONLINEAR PARTS
        for j = 1:size(PDE.RHSOperators,1)
            assemble!(A,b,PDE,SC,Target; equations = [j], min_trigger = AssemblyAlways, verbosity = verbosity - 2)
        end

        # CHECK RESIDUAL
        residual = (A.entries*Target.entries - b.entries).^2
        residual[fixed_dofs] .= 0
        resnorm = (sqrt(sum(residual, dims = 1)[1]))
        if verbosity > 0
            println("  iteration = $j | residual = $resnorm")
        end

        if resnorm < SC.maxResidual
            if verbosity > 0
                println("  converged (maxResidual reached)")
            end
            break;
        end
        if j == SC.maxIterations
            if verbosity > 0
                println("  terminated (maxIterations reached)")
                break
            end
        end

    end


    # REALIZE GLOBAL GLOBALCONSTRAINTS 
    # (possibly changes some entries of Target)
    realize_constraints!(Target,PDE,SC;verbosity = verbosity - 1)

end



function solve!(
    Target::FEVector,
    PDE::PDEDescription;
    dirichlet_penalty::Real = 1e60,
    maxResidual::Real = 1e-12,
    maxIterations::Int = 10,
    verbosity::Int = 0)

    SolverConfig = generate_solver(PDE)
    SolverConfig.dirichlet_penalty = dirichlet_penalty

    if verbosity > 0
        println("\nSOLVING PDE")
        println("===========")
        println("  name = $(PDE.name)")

        FEs = Array{FESpace,1}([])
        for j=1 : length(Target.FEVectorBlocks)
            push!(FEs,Target.FEVectorBlocks[j].FES)
        end    
        if verbosity > 0
            print("   FEs = ")
            for j = 1 : length(Target)
                print("$(Target[j].FES.name) (ndofs = $(Target[j].FES.ndofs))\n         ");
            end
        end

        if verbosity > 1
            Base.show(SolverConfig)
        end
    end

    # check if PDE can be solved directly
    if SolverConfig.is_nonlinear == false
        solve_direct!(Target, PDE, SolverConfig; verbosity = verbosity)
    else
        SolverConfig.maxResidual = maxResidual
        SolverConfig.maxIterations = maxIterations
        solve_fixpoint!(Target, PDE, SolverConfig; verbosity = verbosity)
    end
end


