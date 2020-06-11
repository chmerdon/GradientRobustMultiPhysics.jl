
mutable struct SolverConfig
    is_nonlinear::Bool      # PDE is nonlinear
    is_timedependent::Bool  # PDE is time_dependent
    LHS_AssemblyTriggers::Array{DataType,2} # assembly triggers for blocks in LHS
    RHS_AssemblyTriggers::Array{DataType,1} # assembly triggers for blocks in RHS
    subiterations::Array{Array{Int,1},1} # combination of equations (= rows in PDE.LHS) that should be solved together
    maxIterations::Int          # maximum number of iterations
    maxResidual::Real           # tolerance for residual
    current_time::Real          # current time in a time-dependent setting
    dirichlet_penalty::Real     # penalty for Dirichlet data
    verbosity::Int              # verbosity level (tha larger the more messaging)
end



# check if PDE is nonlinear or time-dependent and which blocks require recalculation
# and devise some initial solver strategy
function generate_solver(PDE::PDEDescription; subiterations = "auto", verbosity = 0)
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
        if length(PDE.LHSOperators[j,k]) == 0
            LHS_ATs[j,k] = AssemblyNever
        else
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
        if length(PDE.RHSOperators[j]) == 0
            RHS_ATs[j] = AssemblyNever
        else
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
    end


    if subiterations != "auto"
        for s = 1 : length(subiterations), k = 1 : size(PDE.LHSOperators,1)
            if (k in subiterations[s]) == false
                for j in subiterations[s]
                    if LHS_ATs[j,k] != AssemblyNever
                        LHS_ATs[j,k] = AssemblyAlways
                    end
                end
            end
        end
        return SolverConfig(nonlinear, timedependent, LHS_ATs, RHS_ATs,subiterations, 10, 1e-10, 0.0, 1e60, verbosity)
    else
        return SolverConfig(nonlinear, timedependent, LHS_ATs, RHS_ATs,[1:size(PDE.LHSOperators,1)], 10, 1e-10, 0.0, 1e60, verbosity)
    end

end

function Base.show(io::IO, SC::SolverConfig)

    println("\nSOLVER-CONFIGURATION")
    println("======================")
    println("         nonlinear = $(SC.is_nonlinear)")
    println("     timedependent = $(SC.is_timedependent)")

    println("     subiterations = $(SC.subiterations)")

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
            elseif SC.LHS_AssemblyTriggers[j,k] == AssemblyNever
                print(" N ")
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
    println("                     (I = Once, T = EachTimeStep, A = Always, N = Never)")

end


function assemble!(
    A::FEMatrix,
    b::FEVector,
    PDE::PDEDescription,
    SC::SolverConfig,
    CurrentSolution::FEVector;
    time::Real = 0,
    equations = [],
    columns = [],
    min_trigger::Type{<:AbstractAssemblyTrigger} = AssemblyAlways,
    verbosity::Int = 0)

    if length(equations) == 0
        equations = 1:size(PDE.LHSOperators,1)
    end
    if length(columns) == 0
        columns = 1:size(PDE.LHSOperators,1)
    end

    if verbosity > 0
        println("\n  Entering assembly of equations=$equations (min_trigger = $min_trigger)")
    end

    rhs_block_has_been_erased = Array{Bool,1}(undef,length(equations))

    for j = 1 : length(equations)
        rhs_block_has_been_erased[j] = false
        if SC.RHS_AssemblyTriggers[equations[j]] <: min_trigger
            if verbosity > 0
                println("  Erasing rhs block [$j]")
            end
            fill!(b[j],0.0)
            rhs_block_has_been_erased[j] = true
            for o = 1 : length(PDE.RHSOperators[equations[j]])
                if verbosity > 0
                    try
                        println("  Assembling into rhs block [$j]: $(typeof(PDE.RHSOperators[equations[j]][o])) ($(PDE.RHSOperators[equations[j]][o].testfunction_operator))")
                    catch
                        println("  Assembling into rhs block [$j]: $(typeof(PDE.RHSOperators[equations[j]][o]))")
                    end
                    @time assemble!(b[j], CurrentSolution, PDE.RHSOperators[equations[j]][o]; time = time, verbosity = verbosity)
                else
                    assemble!(b[j], CurrentSolution, PDE.RHSOperators[equations[j]][o]; time = time, verbosity = verbosity)
                end    
            end
        end
    end

    subblock = 0
    for j = 1:length(equations)
        for k = 1 : size(PDE.LHSOperators,2)
            if (k in equations) && (k in columns)
                subblock += 1
                #println("\n  Equation $j, subblock $subblock")
                if SC.LHS_AssemblyTriggers[equations[j],k] <: min_trigger
                    if verbosity > 0
                        println("  Erasing lhs block [$j,$subblock]")
                    end
                    fill!(A[j,subblock],0.0)
                    for o = 1 : length(PDE.LHSOperators[equations[j],k])
                        PDEoperator = PDE.LHSOperators[equations[j],k][o]
                        if verbosity > 0
                            println("  Assembling into matrix block[$j,$subblock]: $(PDEoperator.name)")
                            if typeof(PDEoperator) <: LagrangeMultiplier
                                @time assemble!(A[j,subblock], CurrentSolution, PDEoperator; time = time, verbosity = verbosity, At = A[subblock,j])
                            else
                                @time assemble!(A[j,subblock], CurrentSolution, PDEoperator; time = time, verbosity = verbosity)
                            end    
                        else
                            if typeof(PDEoperator) <: LagrangeMultiplier
                                assemble!(A[j,subblock], CurrentSolution, PDEoperator; time = time, verbosity = verbosity, At = A[subblock,j])
                            else
                                assemble!(A[j,subblock], CurrentSolution, PDEoperator; time = time, verbosity = verbosity)
                            end    
                        end  
                    end  
                end
            elseif k in columns
                if SC.LHS_AssemblyTriggers[equations[j],k] <: min_trigger
                    if (length(PDE.LHSOperators[equations[j],k]) > 0) && (!(SC.RHS_AssemblyTriggers[equations[j]] <: min_trigger))
                        if rhs_block_has_been_erased[j] == false
                            if verbosity > 0
                                println("  Erasing rhs block [$j]")
                            end
                            fill!(b[j],0.0)
                            rhs_block_has_been_erased[j] = true
                        end
                    end
                    for o = 1 : length(PDE.LHSOperators[equations[j],k])
                        PDEoperator = PDE.LHSOperators[equations[j],k][o]
                        if verbosity > 0
                            println("  Assembling lhs block[$j,$k] into rhs block[$j] ($k not in equations): $(PDEoperator.name)") 
                            @time assemble!(b[j], CurrentSolution, PDEoperator; factor = -1.0, time = time, verbosity = verbosity, fixed_component = k)
                        else  
                            assemble!(b[j], CurrentSolution, PDEoperator; factor = -1.0, time = time, verbosity = verbosity, fixed_component = k)
                        end  
                    end
                end
            end
        end
        subblock = 0
    end
end

# for linear, stationary PDEs that can be solved in one step
function solve_direct!(Target::FEVector, PDE::PDEDescription, SC::SolverConfig)

    verbosity = SC.verbosity

    FEs = Array{FESpace,1}([])
    for j=1 : length(Target.FEVectorBlocks)
        push!(FEs,Target.FEVectorBlocks[j].FES)
    end    

    # ASSEMBLE SYSTEM
    A = FEMatrix{Float64}("SystemMatrix", FEs)
    b = FEVector{Float64}("SystemRhs", FEs)
    assemble!(A,b,PDE,SC,Target; equations = Array{Int,1}(1:length(FEs)), min_trigger = AssemblyInitial, verbosity = verbosity - 1)

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

    # PREPARE GLOBALCONSTRAINTS
    flush!(A.entries)
    for j = 1 : length(PDE.GlobalConstraints)
        additional_fixed_dofs = apply_constraint!(A,b,PDE.GlobalConstraints[j],Target; verbosity = verbosity - 1)
        append!(fixed_dofs,additional_fixed_dofs)
    end

    # PENALIZE FIXED DOFS
    # (from boundary conditions and constraints)
    fixed_dofs = unique(fixed_dofs)
    for j = 1 : length(fixed_dofs)
        b.entries[fixed_dofs[j]] = SC.dirichlet_penalty * Target.entries[fixed_dofs[j]]
        A[1][fixed_dofs[j],fixed_dofs[j]] = SC.dirichlet_penalty
    end

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
    for j = 1 : length(PDE.GlobalConstraints)
        realize_constraint!(Target,PDE.GlobalConstraints[j]; verbosity = verbosity - 1)
    end
end


# solve full system iteratively until fixpoint is reached
function solve_fixpoint_full!(Target::FEVector, PDE::PDEDescription, SC::SolverConfig)

    verbosity = SC.verbosity

    FEs = Array{FESpace,1}([])
    for j=1 : length(Target.FEVectorBlocks)
        push!(FEs,Target.FEVectorBlocks[j].FES)
    end    

    # ASSEMBLE SYSTEM INIT
    A = FEMatrix{Float64}("SystemMatrix", FEs)
    b = FEVector{Float64}("SystemRhs", FEs)
    assemble!(A,b,PDE,SC,Target; equations = Array{Int,1}(1:length(FEs)), min_trigger = AssemblyInitial, verbosity = verbosity - 2)

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

        # PREPARE GLOBALCONSTRAINTS
        flush!(A.entries)
        for j = 1 : length(PDE.GlobalConstraints)
            additional_fixed_dofs = apply_constraint!(A,b,PDE.GlobalConstraints[j],Target; verbosity = verbosity - 1)
            append!(fixed_dofs,additional_fixed_dofs)
        end

        # PENALIZE FIXED DOFS
        # (from boundary conditions and constraints)
        fixed_dofs = unique(fixed_dofs)
        for j = 1 : length(fixed_dofs)
            b.entries[fixed_dofs[j]] = SC.dirichlet_penalty * Target.entries[fixed_dofs[j]]
            A[1][fixed_dofs[j],fixed_dofs[j]] = SC.dirichlet_penalty
        end

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
    for j = 1 : length(PDE.GlobalConstraints)
        realize_constraint!(Target,PDE.GlobalConstraints[j]; verbosity = verbosity - 1)
    end

end


# solve system iteratively until fixpoint is reached
# by solving each equation on its own
function solve_fixpoint_subiterations!(Target::FEVector, PDE::PDEDescription, SC::SolverConfig)

    verbosity = SC.verbosity

    FEs = Array{FESpace,1}([])
    for j=1 : length(Target.FEVectorBlocks)
        push!(FEs,Target.FEVectorBlocks[j].FES)
    end    


    # ASSEMBLE SYSTEM INIT
    nsubiterations = length(SC.subiterations)
    eqoffsets = Array{Array{Int,1},1}(undef,nsubiterations)
    A = Array{FEMatrix{Float64},1}(undef,nsubiterations)
    b = Array{FEVector{Float64},1}(undef,nsubiterations)
    x = Array{FEVector{Float64},1}(undef,nsubiterations)
    for i = 1 : nsubiterations
        A[i] = FEMatrix{Float64}("SystemMatrix subiteration $i", FEs[SC.subiterations[i]])
        b[i] = FEVector{Float64}("SystemRhs subiteration $i", FEs[SC.subiterations[i]])
        x[i] = FEVector{Float64}("SystemRhs subiteration $i", FEs[SC.subiterations[i]])
        assemble!(A[i],b[i],PDE,SC,Target; equations = SC.subiterations[i], min_trigger = AssemblyInitial, verbosity = verbosity - 2)
        eqoffsets[i] = zeros(Int,length(SC.subiterations[i]))
        for j= 1 : length(Target.FEVectorBlocks), eq = 1 : length(SC.subiterations[i])
            if j < SC.subiterations[i][eq]
                eqoffsets[i][eq] += FEs[j].ndofs
            end
        end
    end

    # ASSEMBLE BOUNDARY DATA
    fixed_dofs = []
    eqdof = 0
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

    residual = Array{Array{Float64},1}(undef,nsubiterations)
    for s = 1 : nsubiterations
        residual[s] = zeros(Float64,length(b[s].entries))
    end
    resnorm::Float64 = 0.0

    if verbosity > 0
        println("\n  Starting fixpoint iterations with $(length(SC.subiterations)) subiterations")
    end
    for iteration = 1 : SC.maxIterations

        for s = 1 : nsubiterations
            if verbosity > 1
                println("\n  Subiteration $s with equations $(SC.subiterations[s])")
            end

            # TODO: INCLUDE GLOBALCONSTRAINTS

            # PENALIZE FIXED DOFS
            # (from boundary conditions and constraints)
            fixed_dofs = unique(fixed_dofs)
            for j = 1 : length(fixed_dofs)
                for eq = 1 : length(SC.subiterations[s])
                    # check if fixed_dof is necessary for subiteration
                    if fixed_dofs[j] > eqoffsets[s][eq] && fixed_dofs[j] <= eqoffsets[s][eq]+FEs[SC.subiterations[s][eq]].ndofs
                        eqdof = fixed_dofs[j] - eqoffsets[s][eq]
                        b[s].entries[eqdof] = SC.dirichlet_penalty * Target.entries[fixed_dofs[j]]
                        A[s][1][eqdof,eqdof] = SC.dirichlet_penalty
                    end
                end
            end

            # SOLVE
            if verbosity > 1
                println("\n  Solving")
                @time x[s].entries[:] = A[s].entries\b[s].entries
            else
                x[s].entries[:] = A[s].entries\b[s].entries
            end

            # WRITE INTO Target
            for j = 1 : length(SC.subiterations[s])
                for k = 1 : length(Target[SC.subiterations[s][j]])
                    Target[SC.subiterations[s][j]][k] = x[s][j][k]
                end
            end

        end


        # REASSEMBLE NONLINEAR PARTS
        for s = 1 : nsubiterations
            assemble!(A[s],b[s],PDE,SC,Target; equations = SC.subiterations[s], min_trigger = AssemblyAlways, verbosity = verbosity - 2)
        end

        # CHECK RESIDUAL
        resnorm = 0.0
        for s = 1 : nsubiterations
            residual[s] = (A[s].entries*x[s].entries - b[s].entries).^2
            for j = 1 : length(fixed_dofs)
                for eq = 1 : length(SC.subiterations[s])
                    if fixed_dofs[j] > eqoffsets[s][eq] && fixed_dofs[j] <= eqoffsets[s][eq]+FEs[SC.subiterations[s][eq]].ndofs
                        eqdof = fixed_dofs[j] - eqoffsets[s][eq]
                        residual[s][eqdof] = 0
                    end
                end
            end
            resnorm += (sqrt(sum(residual[s], dims = 1)[1]))
            if verbosity > 1
                println("  residual[$s] = $(sqrt(sum(residual[s], dims = 1)[1]))")
            end
        end
        if verbosity > 0
            println("  iteration = $iteration | residual = $resnorm")
        end



        if resnorm < SC.maxResidual
            if verbosity > 0
                println("  converged (maxResidual reached)")
            end
            break
        end
        if iteration == SC.maxIterations
            if verbosity > 0
                println("  terminated (maxIterations reached)")
                break
            end
        end

    end

    # REALIZE GLOBAL GLOBALCONSTRAINTS 
    # (possibly changes some entries of Target)
    for j = 1 : length(PDE.GlobalConstraints)
        realize_constraint!(Target,PDE.GlobalConstraints[j]; verbosity = verbosity - 1)
    end

end

function solve!(
    Target::FEVector,
    PDE::PDEDescription;
    subiterations = "auto",
    dirichlet_penalty::Real = 1e60,
    maxResidual::Real = 1e-12,
    maxIterations::Int = 10,
    verbosity::Int = 0)

    SolverConfig = generate_solver(PDE; subiterations = subiterations, verbosity = verbosity)
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
    end

    SolverConfig.maxResidual = maxResidual
    SolverConfig.maxIterations = maxIterations
    if verbosity > 1
        Base.show(SolverConfig)
    end


    # check if PDE can be solved directly
    if subiterations == "auto"
        if SolverConfig.is_nonlinear == false
            solve_direct!(Target, PDE, SolverConfig)
        else
            solve_fixpoint_full!(Target, PDE, SolverConfig)
        end
    else
        solve_fixpoint_subiterations!(Target, PDE, SolverConfig)
    end
end


#########################
# TIME-DEPENDENT SOLVER #
#########################


abstract type AbstractTimeIntegrationRule end
abstract type BackwardEuler <: AbstractTimeIntegrationRule end

mutable struct TimeControlSolver{TIR<:AbstractTimeIntegrationRule}
    PDE::PDEDescription      # PDE description (operators, data etc.)
    SC::SolverConfig         # solver configurations (subiterations, penalties etc.)
    ctime::Real              # current time
    cstep::Real              # current timestep count
    last_timestep::Real      # last timestep
    which::Array{Int,1}      # which equations shall have a time derivative?
    AM::Array{FEMatrix,1}    # heap for mass matrices for each equation package
    A::Array{FEMatrix,1}     # heap for system matrix for each equation package
    b::Array{FEVector,1}     # heap for system rhs for each equation package
    x::Array{FEVector,1}     # heap for current solution for each equation package
    X::FEVector              # full solution vector
    fixed_dofs::Array{Int,1}            # fixed dof numbes (values are stored in X)
    eqoffsets::Array{Array{Int,1},1}    # offsets for subblocks of each equation package
end


function TimeControlSolver(
    PDE::PDEDescription,
    InitialValues::FEVector,    # contains initial values and stores solution of advance method below
    TIR::Type{<:AbstractTimeIntegrationRule} = BackwardEuler;
    timedependent_equations = [],
    subiterations = "auto",
    start_time::Real = 0,
    verbosity::Int = 0,
    dirichlet_penalty = 1e60)

    # generate solver for time-independent problem
    SC = generate_solver(PDE; subiterations = subiterations, verbosity = verbosity)
    for j = 1 : length(InitialValues.FEVectorBlocks)
        if !(SC.RHS_AssemblyTriggers[j] <: AssemblyEachTimeStep)
            SC.RHS_AssemblyTriggers[j] = AssemblyEachTimeStep
        end
    end
    SC.dirichlet_penalty = dirichlet_penalty
    SC.maxIterations = 1
    SC.maxResidual = 1e-16

    if verbosity > 0

        println("\nPREPARING TIME-DEPDENDENT SOLVER FOR PDE")
        println("========================================")
        println("    name = $(PDE.name)")
        show(SC)
    end

    # initial assembly
    FEs = Array{FESpace,1}([])
    for j=1 : length(InitialValues.FEVectorBlocks)
        push!(FEs,InitialValues.FEVectorBlocks[j].FES)
    end    
    nsubiterations = length(SC.subiterations)
    eqoffsets = Array{Array{Int,1},1}(undef,nsubiterations)
    AM = Array{FEMatrix{Float64},1}(undef,nsubiterations)
    A = Array{FEMatrix{Float64},1}(undef,nsubiterations)
    b = Array{FEVector{Float64},1}(undef,nsubiterations)
    x = Array{FEVector{Float64},1}(undef,nsubiterations)
    for i = 1 : nsubiterations
        A[i] = FEMatrix{Float64}("SystemMatrix subiteration $i", FEs[SC.subiterations[i]])
        AM[i] = FEMatrix{Float64}("MassMatrix subiteration $i", FEs[SC.subiterations[i]])
        b[i] = FEVector{Float64}("SystemRhs subiteration $i", FEs[SC.subiterations[i]])
        x[i] = FEVector{Float64}("SystemRhs subiteration $i", FEs[SC.subiterations[i]])
        assemble!(A[i],b[i],PDE,SC,InitialValues; time = start_time, equations = SC.subiterations[i], min_trigger = AssemblyInitial, verbosity = verbosity - 2)
        eqoffsets[i] = zeros(Int,length(SC.subiterations[i]))
        for j= 1 : length(FEs), eq = 1 : length(SC.subiterations[i])
            if j < SC.subiterations[i][eq]
                eqoffsets[i][eq] += FEs[j].ndofs
            end
        end
        for j = 1 : length(SC.subiterations[i])
            if (SC.subiterations[i][j] in timedependent_equations) == true
                # assembly mass matrix into AM block
                if verbosity > 1
                    println("\n  Assembling mass matrix for block [$j,$j] of subiteration $i...")
                    @time assemble!(AM[i][j,j],InitialValues,ReactionOperator(DoNotChangeAction(get_ncomponents(eltype(FEs[SC.subiterations[i][j]])))); verbosity = verbosity - 2)
                else
                    assemble!(AM[i][j,j],InitialValues,ReactionOperator(DoNotChangeAction(get_ncomponents(eltype(FEs[SC.subiterations[i][j]])))); verbosity = verbosity - 2)
                end
            end
        end
        flush!(AM[i].entries)
    end


    # ASSEMBLE BOUNDARY DATA
    fixed_dofs = []
    for j= 1 : length(InitialValues.FEVectorBlocks)
        if verbosity > 1
            println("\n  Assembling boundary data for block [$j]...")
            @time new_fixed_dofs = boundarydata!(InitialValues[j],PDE.BoundaryOperators[j]; verbosity = verbosity - 2)
            append!(fixed_dofs, new_fixed_dofs)
        else
            new_fixed_dofs = boundarydata!(InitialValues[j],PDE.BoundaryOperators[j]; verbosity = verbosity - 2)
            append!(fixed_dofs, new_fixed_dofs)
        end    
    end    


    return  TimeControlSolver{TIR}(PDE,SC,start_time,0,0,timedependent_equations,AM,A,b,x,InitialValues, fixed_dofs, eqoffsets)
end

function advance(TCS::TimeControlSolver, timestep::Real = 1e-1)

    # update timestep counter
    TCS.cstep += 1

    # unpack
    SC = TCS.SC
    PDE = TCS.PDE
    A = TCS.A 
    AM = TCS.AM
    b = TCS.b
    x = TCS.x
    X = TCS.X

    if SC.verbosity > 1
        println("\n\n\n  Entering timestep $(TCS.cstep)...")
    end


    nsubiterations = length(SC.subiterations)
    change = zeros(Float64,nsubiterations)
    d = 0
    l = 0
    for s = 1 : nsubiterations

        if SC.verbosity > 1
            println("\n\n\n  Entering subiteration $s...")
        end
        # add change in mass matrix to diagonal blocks if needed
        for k = 1 : length(nsubiterations)
            d = SC.subiterations[s][k]
            if (SC.LHS_AssemblyTriggers[d,d] <: AssemblyEachTimeStep) == true || TCS.last_timestep == 0 # if block was reassembled at the end of the last iteration
                if SC.verbosity > 1
                    println("  Adding mass matrix to block [$k,$k] of subiteration $s")
                end
                addblock(A[s][k,k],AM[s][k,k]; factor = 1.0/timestep)
            else
                if (TCS.last_timestep != timestep) # if block was not reassembled, but timestep changed
                    if SC.verbosity > 1
                        println("  Adding mass matrix change to block [$k,$k] of subiteration $s")
                    end
                    addblock(A[s][k,k],AM[s][k,k]; factor = -1.0/TCS.last_timestep + 1.0/timestep)
                end
            end
            if SC.RHS_AssemblyTriggers[d] <: AssemblyEachTimeStep || TCS.last_timestep == 0 # if rhs block was reassembled at the end of the last iteration
                if SC.verbosity > 1
                    println("  Adding time derivative to rhs block [$k] of subiteration $s")
                end
                addblock_matmul(b[s][k],AM[s][k,k],X[d]; factor = 1.0/timestep)
            else
                if SC.verbosity > 1
                    println("  Adding time derivative change to rhs block [$k] of subiteration $s")
                end
                addblock_matmul(b[s][k],AM[s][k,k],x[s][k]; factor = -1.0/TCS.last_timestep) # subtract rhs from last time step
                addblock_matmul(b[s][k],AM[s][k,k],X[d]; factor = 1.0/timestep) # add new rhs from last time step
            end
        end
        flush!(A[s].entries)

        # TODO: update boundary data

        # PENALIZE FIXED DOFS
        # (from boundary conditions and constraints)
        fixed_dofs = TCS.fixed_dofs
        eqoffsets = TCS.eqoffsets
        for j = 1 : length(fixed_dofs)
            for eq = 1 : length(SC.subiterations[s])
                if fixed_dofs[j] > eqoffsets[s][eq] && fixed_dofs[j] <= eqoffsets[s][eq]+X[SC.subiterations[s][eq]].FES.ndofs
                    eqdof = fixed_dofs[j] - eqoffsets[s][eq]
                    b[s].entries[eqdof] = SC.dirichlet_penalty * X.entries[fixed_dofs[j]]
                    A[s][1][eqdof,eqdof] = SC.dirichlet_penalty
                end
            end
        end

        # SOLVE
        if SC.verbosity > 0
            println("\n  Solving equation(s) $(SC.subiterations[s])")
            @time x[s].entries[:] = A[s].entries\b[s].entries
            residual = norm(A[s].entries*x[s].entries - b[s].entries)
            if SC.verbosity > 0
                println("    ... residual = $residual")
            end
        else
            x[s].entries[:] = A[s].entries\b[s].entries
        end

        # WRITE INTO X and COMPUTE CHANGE
        l = 0
        for j = 1 : length(SC.subiterations[s])
            for k = 1 : length(X[SC.subiterations[s][j]])
                change[s] += (X[SC.subiterations[s][j]][k] - x[s][j][k])^2
                X[SC.subiterations[s][j]][k] = x[s][j][k]
            end
            l += length(X[SC.subiterations[s][j]])
        end
        #change[s] /= l^4*timestep^2
        if SC.verbosity > 0
            println("    ... change = $(sqrt(change[s]))")
        end

        # REASSEMBLE NONLINEAR PARTS DEPENDING ON CURRENT SUBITERATES
        # todo: only reassemble blocks that really change because they depend on subiterates
        # i.e. blocks that are in the respective columns and nonlinearterms that depend on the subiterates elsewhere (e.g. Upwind)
        for s2 = 1 : nsubiterations
            assemble!(A[s2],b[s2],PDE,SC,X; equations = SC.subiterations[s2], min_trigger = AssemblyEachTimeStep, verbosity = SC.verbosity - 1)
        end
    end
    TCS.last_timestep = timestep
    TCS.ctime += timestep

    return sqrt(sum(change))
end