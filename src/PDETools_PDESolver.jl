
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


