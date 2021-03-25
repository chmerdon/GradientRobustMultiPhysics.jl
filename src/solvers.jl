#############
# PDESolver #
#############
#
# collection of methods to solve the PDE described in a PDEDescription
# there are several possiblities (in each timestep):
# - direct solve of all equations at once
# - fixpoint iteration of all equations at once
# - fixpoint iteration with subiterations where only subsets of equations are solved together
#
# the strut SolverConfig collects all the information needed to assess if a block must be reassembled
# at some point (e.g. after a time step or after a iteration) and there are triggers that mark a
# block to be nonlinear/time-dependent
#
# also parameter to steer penalties and stopping criterarions are saved in SolverConfig

abstract type AbstractLinearSystem{T} end

mutable struct SolverConfig{T <: Real}
    is_nonlinear::Bool      # PDE is nonlinear
    is_timedependent::Bool  # PDE is time_dependent
    LHS_AssemblyTriggers::Array{DataType,2} # assembly triggers for blocks in LHS
    LHS_dependencies::Array{Array{Int,1},2} # dependencies on components
    RHS_AssemblyTriggers::Array{DataType,1} # assembly triggers for blocks in RHS
    RHS_dependencies::Array{Array{Int,1},1} # dependencies on components
    LHS_AssemblyPatterns::Array{Array{AssemblyPattern,1},2} # last used assembly pattern (and assembly pattern preps to avoid their recomputation)
    RHS_AssemblyPatterns::Array{Array{AssemblyPattern,1},1} # last used assembly pattern (and assembly pattern preps to avoid their recomputation)
    LHS_AssemblyTimes::Array{Array{Float64,1},2}
    RHS_AssemblyTimes::Array{Array{Float64,1},1}
    subiterations::Array{Array{Int,1},1} # combination of equations (= rows in PDE.LHS) that should be solved together
    maxIterations::Int          # maximum number of iterations
    maxResidual::T           # tolerance for residual
    current_time::T          # current time in a time-dependent setting
    dirichlet_penalty::T     # penalty for Dirichlet data
    linsolver::Union{Nothing,Module,Type{<:AbstractLinearSystem}}
    anderson_iterations::Int    # number of Anderson iterations (= 0 normal fixed-point iterations, > 0 Anderson iterations)
    skip_update::Array{Int,1}    # max number of solves the LU decomposition of linsolver is reused
    check_nonlinear_residual::Bool    # check nonlinear residual (only used in time-dependent solver)
    damping::T               # damping factor (only for nonlinear solves, 0 = no damping, must be < 1 !)
    verbosity::Int              # verbosity level (tha larger the more messaging)
end


mutable struct LinearSystemDirectUMFPACK{T,verbosity} <: AbstractLinearSystem{T}
    x::AbstractVector{T}
    A::ExtendableSparseMatrix{T,Int64}
    b::AbstractVector{T}
    ALU::SuiteSparse.UMFPACK.UmfpackLU{T,Int64}     # LU factorization
    LinearSystemDirectUMFPACK{T,verbosity}(x,A,b) where {T,verbosity} = new{T,verbosity}(x,A,b)
end

function createsolver(ST::Type{<:AbstractLinearSystem{T}},x::AbstractVector{T},A::ExtendableSparseMatrix{T,Int64},b::AbstractVector{T}) where {T}
    return ST(x,A,b)
end

function update!(LS::AbstractLinearSystem{T}) where {T}
    # do nothing for an abstract solver
end

function update!(LS::LinearSystemDirectUMFPACK{T,verbosity}) where {T, verbosity}
    #try
    #    if verbosity > 1
    #        println("\n  Updating LU decomposition...")
    #    end
    #    lu!(LS.ALU,LS.A.cscmatrix)
    #catch
        if verbosity > 1
            println("\n  (Re)generating LU decomposition...")
        end
        LS.ALU = lu(LS.A.cscmatrix)
    #end
end

function solve!(LS::LinearSystemDirectUMFPACK{T,verbosity}) where {T, verbosity}
    if verbosity > 1
        println("\n  Solving directly with UMFPACK...")
    end
    ldiv!(LS.x,LS.ALU,LS.b)
end



# check if PDE is nonlinear or time-dependent and which blocks require recalculation
# and devise some initial solver strategy
function generate_solver(PDE::PDEDescription, T::Type{<:Real} = Float64; subiterations = "auto", verbosity = 0)
    nonlinear::Bool = false
    timedependent::Bool = false
    block_nonlinear::Bool = false
    block_timedependent::Bool = false
    op_nonlinear::Bool = false
    op_timedependent::Bool = false
    LHS_ATs = Array{DataType,2}(undef,size(PDE.LHSOperators,1),size(PDE.LHSOperators,2))
    LHS_dep = Array{Array{Int,1},2}(undef,size(PDE.LHSOperators,1),size(PDE.LHSOperators,2))
    current_dep::Array{Int,1} = []
    for j = 1 : size(PDE.LHSOperators,1), k = 1 : size(PDE.LHSOperators,2)
        block_nonlinear = false
        block_timedependent = false
        current_dep = [j,k]
        for o = 1 : length(PDE.LHSOperators[j,k])
            # check nonlinearity and time-dependency of operator
            op_nonlinear, op_timedependent = check_PDEoperator(PDE.LHSOperators[j,k][o])
            # check dependency on components
            for beta = 1:size(PDE.LHSOperators,2)
                if check_dependency(PDE.LHSOperators[j,k][o],beta) == true
                    push!(current_dep,beta)
                end
            end
            if op_nonlinear == true
                block_nonlinear = true
            end
            if op_timedependent == true
                block_timedependent = true
            end
        end
        # check nonlinearity and time-dependency of whole block
        # and assign appropriate AssemblyTrigger
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
        # assign dependencies
        LHS_dep[j,k] = deepcopy(unique(current_dep))
    end
    RHS_ATs = Array{DataType,1}(undef,size(PDE.RHSOperators,1))
    RHS_dep = Array{Array{Int,1},1}(undef,size(PDE.RHSOperators,1))
    for j = 1 : size(PDE.RHSOperators,1)
        block_nonlinear = false
        block_timedependent = false
        current_dep = []
        for o = 1 : length(PDE.RHSOperators[j])
            op_nonlinear, op_timedependent = check_PDEoperator(PDE.RHSOperators[j][o])
            for beta = 1:size(PDE.LHSOperators,2)
                if check_dependency(PDE.RHSOperators[j][o],beta) == true
                    push!(current_dep,beta)
                end
            end
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
        # assign dependencies
        RHS_dep[j] = deepcopy(Base.unique(current_dep))
    end
    LHS_APs = Array{Array{AssemblyPattern,1},2}(undef,size(PDE.LHSOperators,1),size(PDE.LHSOperators,1))
    RHS_APs = Array{Array{AssemblyPattern,1},1}(undef,size(PDE.RHSOperators,1))
    LHS_AssemblyTimes = Array{Array{Float64,1},2}(undef,size(PDE.LHSOperators,1),size(PDE.LHSOperators,2))
    RHS_AssemblyTimes = Array{Array{Float64,1},1}(undef,size(PDE.RHSOperators,1))
    for j = 1 : size(PDE.LHSOperators,1), k = 1 : size(PDE.LHSOperators,2)
        LHS_APs[j,k] = Array{AssemblyPattern,1}(undef, length(PDE.LHSOperators[j,k]))
        LHS_AssemblyTimes[j,k] = zeros(Float64, length(PDE.LHSOperators[j,k]))
        for o = 1 : length(PDE.LHSOperators[j,k])
            LHS_APs[j,k][o] = AssemblyPattern()
        end
    end
    for j = 1 : size(PDE.RHSOperators,1)
        RHS_APs[j] = Array{AssemblyPattern,1}(undef, length(PDE.RHSOperators[j]))
        RHS_AssemblyTimes[j] = zeros(Float64, length(PDE.RHSOperators[j]))
        for o = 1 : length(PDE.RHSOperators[j])
            RHS_APs[j][o] = AssemblyPattern()
        end
    end
    if subiterations != "auto"
        for s = 1 : length(subiterations), k = 1 : size(PDE.LHSOperators,1)
            if (k in subiterations[s]) == false
                for j in subiterations[s]
                    if LHS_ATs[j,k] == AssemblyInitial
                        LHS_ATs[j,k] = AssemblyEachTimeStep
                    end
                end
            end
        end
        return SolverConfig{T}(nonlinear, timedependent, LHS_ATs, LHS_dep, RHS_ATs, RHS_dep, LHS_APs, RHS_APs, LHS_AssemblyTimes, RHS_AssemblyTimes, subiterations, 10, 1e-10, 0.0, 1e60, LinearSystemDirectUMFPACK{T,0}, 0, [1], false, 0,verbosity)
    else
        return SolverConfig{T}(nonlinear, timedependent, LHS_ATs, LHS_dep, RHS_ATs, RHS_dep, LHS_APs, RHS_APs, LHS_AssemblyTimes, RHS_AssemblyTimes, [1:size(PDE.LHSOperators,1)], 10, 1e-10, 0.0, 1e60, LinearSystemDirectUMFPACK{T,0}, 0, [1], false, 0,verbosity)
    end

end

function Base.show(io::IO, SC::SolverConfig)

    println("\nSOLVER-CONFIGURATION")
    println("======================")
    #println("              type = $(SC.parameters[1])")
    println("         nonlinear = $(SC.is_nonlinear)")
    println("     timedependent = $(SC.is_timedependent)")
    println("         linsolver = $(SC.linsolver)")

    println("     subiterations = $(SC.subiterations)")
    println("     maxIterations = $(SC.maxIterations)")
    println("       maxResidual = $(SC.maxResidual)")

    if SC.anderson_iterations > 0
        println("       AndersonIts = $(SC.anderson_iterations)")
    end

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
        elseif SC.RHS_AssemblyTriggers[j] == AssemblyNever
            print(" N ")
        end
        println("")
    end
    println("                     (I = Once, T = EachTimeStep, A = Always, N = Never)")
    println("\n  LHS_dependencies = $(SC.LHS_dependencies)\n")

end



function show_statistics(io::IO, PDE::PDEDescription, SC::SolverConfig)

    subiterations = SC.subiterations

    println("\nACCUMULATED ASSEMBLY TIMES")
    println("==========================")

    for s = 1 : length(subiterations)
        for j = 1 : length(subiterations[s])
            eq = subiterations[s][j]
            for k = 1 : size(SC.LHS_AssemblyTimes,2)
                for o = 1 : size(SC.LHS_AssemblyTimes[eq,k],1)
                    println("  LHS[$eq,$k][$o] ($(PDE.LHSOperators[eq,k][o].name)) = $(SC.LHS_AssemblyTimes[eq,k][o])s")
                end
            end
        end
    end

    for s = 1 : length(subiterations)
        for j = 1 : length(subiterations[s])
            eq = subiterations[s][j]
            for o = 1 : size(SC.RHS_AssemblyTimes[eq],1)
                println("  RHS[$eq][$o] ($(PDE.RHSOperators[eq][o].name)) = $(SC.RHS_AssemblyTimes[eq][o])s")
            end
        end
    end

end



function assemble!(
    A::FEMatrix{T},
    b::FEVector{T},
    PDE::PDEDescription,
    SC::SolverConfig{T},
    CurrentSolution::FEVector;
    time::Real = 0,
    equations = [],
    if_depends_on = [], # block is only assembled if it depends on these components
    min_trigger::Type{<:AbstractAssemblyTrigger} = AssemblyAlways,
    storage_trigger = "same as min_trigger",
    only_rhs::Bool = false,
    verbosity::Int = 0) where {T <: Real}

    if length(equations) == 0
        equations = 1:size(PDE.LHSOperators,1)
    end
    if length(if_depends_on) == 0
        if_depends_on = 1:size(PDE.LHSOperators,1)
    end
    if storage_trigger == "same as min_trigger"
        storage_trigger = min_trigger
    end

    if verbosity > 0
        println("\n  Entering assembly of equations=$equations (time = $time, min_trigger = $min_trigger)")
    end

    # important to flush first in case there is some cached stuff that
    # will not be seen by fill! functions
    flush!(A.entries)

    elapsedtime::Float64 = 0
    # force (re)assembly of stored bilinearforms and RhsOperators
    if (min_trigger == AssemblyInitial) == true && !only_rhs
        for j = 1:length(equations)
            for k = 1 : size(PDE.LHSOperators,2)
                if (storage_trigger <: SC.LHS_AssemblyTriggers[equations[j],k]) == true
                    for o = 1 : length(PDE.LHSOperators[equations[j],k])
                        PDEOperator = PDE.LHSOperators[equations[j],k][o]
                        try
                            if PDEOperator.store_operator == false
                                continue
                            end
                        catch
                            continue
                        end
                        elapsedtime = @elapsed update_storage!(PDEOperator, CurrentSolution, equations[j], k; time = time, verbosity = verbosity)
                        SC.LHS_AssemblyTimes[equations[j],k][o] += elapsedtime
                        if verbosity > 0
                            println("  Assembly time for storage of operator $(PDEOperator.name) = $(elapsedtime)s (total = $(SC.LHS_AssemblyTimes[equations[j],k][o])s)")
                        end
                    end
                end
            end
        end
        for j = 1:length(equations)
            if (storage_trigger <: SC.RHS_AssemblyTriggers[equations[j]]) == true
                for o = 1 : length(PDE.RHSOperators[equations[j]])
                    PDEOperator = PDE.RHSOperators[equations[j]][o]
                    try
                        if PDEOperator.store_operator == false
                            continue
                        end
                    catch
                        continue
                    end
                    elapsedtime = @elapsed update_storage!(PDEOperator, CurrentSolution, equations[j] ; time = time, verbosity = verbosity)
                    SC.RHS_AssemblyTimes[equations[j]][o] += elapsedtime
                    if verbosity > 0
                        println("  Assembly time for storage of operator $(PDEOperator.name) = $(elapsedtime)s (total = $(SC.RHS_AssemblyTimes[equations[j]][o])s)")
                    end
                end
            end
        end
    end

    # (re)assembly right-hand side
    rhs_block_has_been_erased = zeros(Bool,length(equations))
    for j = 1 : length(equations)
        if (min_trigger <: SC.RHS_AssemblyTriggers[equations[j]]) == true
            if verbosity > 0
                println("  Erasing rhs block [$j]")
            end
            fill!(b[j],0.0)
            rhs_block_has_been_erased[j] = true
            for o = 1 : length(PDE.RHSOperators[equations[j]])
                PDEOperator = PDE.RHSOperators[equations[j]][o]
                elapsedtime = @elapsed assemble!(b[j], SC, equations[j], o, PDE.RHSOperators[equations[j]][o], CurrentSolution; time = time, verbosity = verbosity)
                SC.RHS_AssemblyTimes[equations[j]][o] += elapsedtime
                if verbosity > 0
                    println("  Assembly time for operator $(PDEOperator.name) = $(elapsedtime)s (total = $(SC.RHS_AssemblyTimes[equations[j]][o])s)")
                end   
            end
        end
    end

    # (re)assembly left-hand side
    lhs_block_has_been_erased = zeros(Bool,length(equations),length(equations))
    subblock = 0
    for j = 1:length(equations)
        for k = 1 : size(PDE.LHSOperators,2)
            if length(intersect(SC.LHS_dependencies[equations[j],k], if_depends_on)) > 0
                if (k in equations) && !only_rhs
                    subblock += 1
                    #println("\n  Equation $j, subblock $subblock")
                    if (min_trigger <: SC.LHS_AssemblyTriggers[equations[j],k]) == true
                        if verbosity > 0
                            println("  Erasing lhs block [$j,$subblock]")
                        end
                        fill!(A[j,subblock],0)
                        lhs_block_has_been_erased[j, subblock] = true
                        for o = 1 : length(PDE.LHSOperators[equations[j],k])
                            PDEOperator = PDE.LHSOperators[equations[j],k][o]
                            if typeof(PDEOperator) <: LagrangeMultiplier
                                elapsedtime = @elapsed assemble!(A[j,subblock], SC, equations[j],k,o, PDEOperator, CurrentSolution; time = time, verbosity = verbosity, At = A[subblock,j])
                            else
                                elapsedtime = @elapsed assemble!(A[j,subblock], SC, equations[j],k,o, PDEOperator, CurrentSolution; time = time, verbosity = verbosity)
                            end  
                            SC.LHS_AssemblyTimes[equations[j],k][o] += elapsedtime
                            if verbosity > 0
                                println("  Assembly time for operator $(PDEOperator.name) = $(elapsedtime)s (total = $(SC.LHS_AssemblyTimes[equations[j],k][o])s)")
                            end
                        end  
                    end
                elseif !(k in equations)
                    if (min_trigger <: SC.LHS_AssemblyTriggers[equations[j],k]) == true
                        if (length(PDE.LHSOperators[equations[j],k]) > 0) && (!(min_trigger <: SC.RHS_AssemblyTriggers[equations[j]]))
                            if rhs_block_has_been_erased[j] == false
                                if verbosity > 0
                                    println("  Erasing rhs block [$j]")
                                end
                                fill!(b[j],0)
                                rhs_block_has_been_erased[j] = true
                            end
                        end
                        for o = 1 : length(PDE.LHSOperators[equations[j],k])
                            PDEOperator = PDE.LHSOperators[equations[j],k][o]
                            if verbosity > 0
                                println("  Assembling lhs block[$j,$k] into rhs block[$j] ($k not in equations): $(PDEOperator.name)") 
                                elapsedtime = @elapsed assemble!(b[j], SC, equations[j],k,o, PDEOperator, CurrentSolution; factor = -1.0, time = time, verbosity = verbosity, fixed_component = k)
                            else  
                                elapsedtime = @elapsed assemble!(b[j], SC, equations[j],k,o, PDEOperator, CurrentSolution; factor = -1.0, time = time, verbosity = verbosity, fixed_component = k)
                            end  
                            SC.LHS_AssemblyTimes[equations[j],k][o] += elapsedtime
                            if verbosity > 0
                                println("  Assembly time for operator $(PDEOperator.name) = $(elapsedtime)s (total = $(SC.LHS_AssemblyTimes[equations[j],k][o])s)")
                            end
                        end
                    end
                end
            end
        end
        subblock = 0
    end

    return lhs_block_has_been_erased, rhs_block_has_been_erased
end

# for linear, stationary PDEs that can be solved in one step
function solve_direct!(Target::FEVector{T}, PDE::PDEDescription, SC::SolverConfig{T}; time::Real = 0) where {T <: Real}

    verbosity = SC.verbosity

    FEs = Array{FESpace,1}([])
    for j=1 : length(Target.FEVectorBlocks)
        push!(FEs,Target.FEVectorBlocks[j].FES)
    end    

    # ASSEMBLE SYSTEM
    A = FEMatrix{T}("SystemMatrix", FEs)
    b = FEVector{T}("SystemRhs", FEs)
    assemble!(A,b,PDE,SC,Target; equations = Array{Int,1}(1:length(FEs)), min_trigger = AssemblyInitial, time = time, verbosity = verbosity - 2)

    # ASSEMBLE BOUNDARY DATA
    fixed_dofs = []
    for j= 1 : length(Target.FEVectorBlocks)
        if verbosity > 2
            println("\n  Assembling boundary data for block [$j]...")
            @time new_fixed_dofs = boundarydata!(Target[j],PDE.BoundaryOperators[j]; time = time, verbosity = verbosity - 2)
        else
            new_fixed_dofs = boundarydata!(Target[j],PDE.BoundaryOperators[j]; time = time, verbosity = verbosity - 2)
        end    
        new_fixed_dofs .+= Target[j].offset
        append!(fixed_dofs, new_fixed_dofs)
    end    

    # PREPARE GLOBALCONSTRAINTS
    flush!(A.entries)
    for j = 1 : length(PDE.GlobalConstraints)
        additional_fixed_dofs = apply_constraint!(A,b,PDE.GlobalConstraints[j],Target; verbosity = verbosity - 2)
        append!(fixed_dofs,additional_fixed_dofs)
    end

    # PENALIZE FIXED DOFS
    # (from boundary conditions and constraints)
    fixed_dofs = Base.unique(fixed_dofs)
    for j = 1 : length(fixed_dofs)
        b.entries[fixed_dofs[j]] = SC.dirichlet_penalty * Target.entries[fixed_dofs[j]]
        A[1][fixed_dofs[j],fixed_dofs[j]] = SC.dirichlet_penalty
    end

    # SOLVE
    LS = createsolver(SC.linsolver,Target.entries,A.entries,b.entries)
    flush!(A.entries)
    update!(LS)
    solve!(LS)

    residuals = zeros(T, length(Target.FEVectorBlocks))
    # CHECK RESIDUAL
    residual = A.entries*Target.entries - b.entries
    residual[fixed_dofs] .= 0
    for j = 1 : length(Target.FEVectorBlocks)
        for k = 1 : Target.FEVectorBlocks[j].FES.ndofs
            residuals[j] += residual[k + Target.FEVectorBlocks[j].offset].^2
        end
    end
    resnorm = sum(residuals)
    residuals = sqrt.(residuals)
    if verbosity > 0
        println("\n  residuals = $residuals")
    end

    # REALIZE GLOBAL GLOBALCONSTRAINTS 
    # (possibly changes some entries of Target)
    for j = 1 : length(PDE.GlobalConstraints)
        realize_constraint!(Target,PDE.GlobalConstraints[j]; verbosity = verbosity - 2)
    end

    if verbosity > 1
        show_statistics(stdout,PDE,SC)
    end
    return sqrt(resnorm)
end


# solve full system iteratively until fixpoint is reached
function solve_fixpoint_full!(Target::FEVector{T}, PDE::PDEDescription, SC::SolverConfig{T}; time::Real = 0) where {T <: Real}

    verbosity = SC.verbosity

    anderson_iterations = SC.anderson_iterations

    FEs = Array{FESpace,1}([])
    for j=1 : length(Target.FEVectorBlocks)
        push!(FEs,Target.FEVectorBlocks[j].FES)
    end    

    # ASSEMBLE SYSTEM INIT
    A = FEMatrix{T}("SystemMatrix", FEs)
    b = FEVector{T}("SystemRhs", FEs)
    assembly_time = @elapsed assemble!(A,b,PDE,SC,Target; time = time, equations = Array{Int,1}(1:length(FEs)), min_trigger = AssemblyInitial, verbosity = verbosity - 2)

    # ASSEMBLE BOUNDARY DATA
    fixed_dofs = []
    assembly_time += @elapsed for j= 1 : length(Target.FEVectorBlocks)
        if verbosity > 2
            println("\n  Assembling boundary data for block [$j]...")
            @time new_fixed_dofs = boundarydata!(Target[j],PDE.BoundaryOperators[j]; time = time, verbosity = verbosity - 2) .+ Target[j].offset
            append!(fixed_dofs, new_fixed_dofs)
        else
            new_fixed_dofs = boundarydata!(Target[j],PDE.BoundaryOperators[j]; time = time, verbosity = verbosity - 2) .+ Target[j].offset
            append!(fixed_dofs, new_fixed_dofs)
        end    
    end    

    # ANDERSON ITERATIONS
    if anderson_iterations > 0
        # we need to save the last iterates
        LastAndersonIterates = Array{FEVector,1}(undef, anderson_iterations+1) # actual iterates u_k
        LastAndersonIteratesTilde = Array{FEVector,1}(undef, anderson_iterations+1) # auxiliary iterates \tilde u_k
        PDE.LHSOperators[1,1][1].store_operator = true # use the first operator to compute norm in which Anderson iterations are optimised
        AIONormOperator = PDE.LHSOperators[1,1][1].storage
        AIOMatrix = zeros(T, anderson_iterations+2, anderson_iterations+2)
        AIORhs = zeros(T, anderson_iterations+2)
        AIORhs[anderson_iterations+2] = 1 # sum of all coefficients should equal 1
        AIOalpha = zeros(T, anderson_iterations+2)
        for j = 1 : anderson_iterations+1
            LastAndersonIterates[j] = deepcopy(Target)
            LastAndersonIteratesTilde[j] = deepcopy(Target)
            AIOMatrix[j,anderson_iterations+2] = 1
            AIOMatrix[anderson_iterations+2,j] = 1
        end
    end
    if SC.damping > 0
        LastIterate = deepcopy(Target)
    end

    residual = zeros(T,length(b.entries))
    linresnorm::T = 0.0
    resnorm::T = 0.0

    ## INIT SOLVER
    LS = createsolver(SC.linsolver,Target.entries,A.entries,b.entries)

    if verbosity > 1
        @printf("\n  initial assembly time = %.2e (s)\n",assembly_time)
        @printf("\n  ITERATION |  LSRESIDUAL  |  NLRESIDUAL  | TIME ASSEMBLY/SOLVE/TOTAL (s)")
        @printf("\n  -----------------------------------------------------------------------")
    end
    for j = 1 : SC.maxIterations
        time_total = @elapsed begin

        # PREPARE GLOBALCONSTRAINTS
        flush!(A.entries)
        for j = 1 : length(PDE.GlobalConstraints)
            additional_fixed_dofs = apply_constraint!(A,b,PDE.GlobalConstraints[j],Target; verbosity = verbosity - 2)
            append!(fixed_dofs,additional_fixed_dofs)
        end

        # PENALIZE FIXED DOFS
        # (from boundary conditions and constraints)
        fixed_dofs = Base.unique(fixed_dofs)
        for j = 1 : length(fixed_dofs)
            b.entries[fixed_dofs[j]] = SC.dirichlet_penalty * Target.entries[fixed_dofs[j]]
            A[1][fixed_dofs[j],fixed_dofs[j]] = SC.dirichlet_penalty
        end

        # SOLVE
        time_solver = @elapsed begin
            flush!(A.entries)
            if j == 1 || (j % SC.skip_update[1] == 0 && SC.skip_update[1] != -1)
                update!(LS)
            end
            solve!(LS)
        end

        # CHECK LINEAR RESIDUAL
        if verbosity > 1
            residual = A.entries*Target.entries - b.entries
            residual[fixed_dofs] .= 0
            linresnorm = (sqrt(sum(residual.^2, dims = 1)[1]))
        end

        # POSTPRCOESS : ANDERSON ITERATE
        if anderson_iterations > 0
            depth = min(anderson_iterations,j-1)

            # move last tilde iterates to the front in memory
            for j = 1 : anderson_iterations
                LastAndersonIteratesTilde[j] = deepcopy(LastAndersonIteratesTilde[j+1])
            end
            # save fixpoint iterate as new tilde iterate
            LastAndersonIteratesTilde[anderson_iterations+1] = deepcopy(Target)

            if depth > 0
                # fill matrix
                for j = 1 : depth+1, k = 1 : depth+1
                    AIOMatrix[j,k] = lrmatmul(LastAndersonIteratesTilde[anderson_iterations+2-j][1][:] .- LastAndersonIterates[anderson_iterations+2-j][1][:],
                    AIONormOperator, LastAndersonIteratesTilde[anderson_iterations+2-k][1][:] .- LastAndersonIterates[anderson_iterations+2-k][1][:])
                end
                # solve for alpha coefficients
                ind = union(1:depth+1,anderson_iterations+2)
                AIOalpha[ind] = AIOMatrix[ind,ind]\AIORhs[ind]

                # move last iterates to the front in memory
                for j = 1 : anderson_iterations
                    LastAndersonIterates[j] = deepcopy(LastAndersonIterates[j+1])
                end

                #println("alpha = $(AIOalpha)")
                # compute next iterates
                fill!(Target[1],0.0)
                for a = 1 : depth+1
                    for j = 1 : FEs[1].ndofs
                        Target[1][j] += AIOalpha[a] * LastAndersonIteratesTilde[anderson_iterations+2-a][1][j]
                    end
                end
                LastAndersonIterates[anderson_iterations+1] = deepcopy(Target)
            else
                LastAndersonIterates[anderson_iterations+1] = deepcopy(Target)
            end
        end

        # POSTPROCESS : DAMPING
        if SC.damping > 0
            for j = 1 : length(Target.entries)
                Target.entries[j] = SC.damping*LastIterate.entries[j] + (1-SC.damping)*Target.entries[j]
            end
            LastIterate = deepcopy(Target)
        end


        # REASSEMBLE NONLINEAR PARTS
        time_reassembly = @elapsed for j = 1:size(PDE.RHSOperators,1)
            assemble!(A,b,PDE,SC,Target; time = time, equations = [j], min_trigger = AssemblyAlways, verbosity = verbosity - 2)
        end

        # CHECK NONLINEAR RESIDUAL
        residual = A.entries*Target.entries - b.entries
        residual[fixed_dofs] .= 0
        resnorm = (sqrt(sum(residual.^2, dims = 1)[1]))

        end #elapsed

        if verbosity > 1
            @printf("\n     %4d  ", j)
            @printf(" | %e", linresnorm)
            @printf(" | %e", resnorm)
            @printf(" | %.2e/%.2e/%.2e",time_reassembly,time_solver, time_total)
        end

        if resnorm < SC.maxResidual
            if verbosity > 0
                println("  converged after $j iterations (maxResidual reached)")
            end
            break;
        end
        if j == SC.maxIterations
            if verbosity > 0
                println("  terminated after $j iterations (maxIterations reached)")
                break
            end
        end

    end

    # REALIZE GLOBAL GLOBALCONSTRAINTS 
    # (possibly changes some entries of Target)
    for j = 1 : length(PDE.GlobalConstraints)
        realize_constraint!(Target,PDE.GlobalConstraints[j]; verbosity = verbosity - 2)
    end

    if verbosity > 2
        show_statistics(stdout,PDE,SC)
    end

    return resnorm
end


# solve system iteratively until fixpoint is reached
# by solving each equation on its own
function solve_fixpoint_subiterations!(Target::FEVector{T}, PDE::PDEDescription, SC::SolverConfig{T}; time = 0) where {T <: Real}

    verbosity = SC.verbosity

    FEs = Array{FESpace,1}([])
    for j=1 : length(Target.FEVectorBlocks)
        push!(FEs,Target.FEVectorBlocks[j].FES)
    end    

    assembly_time = @elapsed begin
        # ASSEMBLE SYSTEM INIT
        nsubiterations = length(SC.subiterations)
        eqoffsets = Array{Array{Int,1},1}(undef,nsubiterations)
        A = Array{FEMatrix{T},1}(undef,nsubiterations)
        b = Array{FEVector{T},1}(undef,nsubiterations)
        x = Array{FEVector{T},1}(undef,nsubiterations)
        for i = 1 : nsubiterations
            A[i] = FEMatrix{T}("SystemMatrix subiteration $i", FEs[SC.subiterations[i]])
            b[i] = FEVector{T}("SystemRhs subiteration $i", FEs[SC.subiterations[i]])
            x[i] = FEVector{T}("SystemRhs subiteration $i", FEs[SC.subiterations[i]])
            assemble!(A[i],b[i],PDE,SC,Target; time = time, equations = SC.subiterations[i], min_trigger = AssemblyInitial, verbosity = verbosity - 2)
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
        for j = 1 : length(Target.FEVectorBlocks)
            if verbosity > 2
                println("\n  Assembling boundary data for block [$j]...")
                @time new_fixed_dofs = boundarydata!(Target[j],PDE.BoundaryOperators[j]; time = time, verbosity = verbosity - 2) .+ Target[j].offset
                append!(fixed_dofs, new_fixed_dofs)
            else
                new_fixed_dofs = boundarydata!(Target[j],PDE.BoundaryOperators[j]; time = time, verbosity = verbosity - 2) .+ Target[j].offset
                append!(fixed_dofs, new_fixed_dofs)
            end    
        end    
    end
    
    residual = Array{FEVector{T},1}(undef,nsubiterations)
    for s = 1 : nsubiterations
        residual[s] = FEVector{T}("residual subiteration $s", FEs[SC.subiterations[s]])
    end
    linresnorm = zeros(T,nsubiterations)
    resnorm = zeros(T,nsubiterations)


    ## INIT SOLVERS
    LS = Array{AbstractLinearSystem,1}(undef,nsubiterations)
    for s = 1 : nsubiterations
        LS[s] = createsolver(SC.linsolver,x[s].entries,A[s].entries,b[s].entries)
    end


    if verbosity > 1
        @printf("\n  initial assembly time = %.2e (s)\n",assembly_time)
        @printf("\n  ITERATION |  LSRESIDUAL  |  NLRESIDUAL  | TIME ASSEMBLY/SOLVE/TOTAL (s)")
        @printf("\n  -----------------------------------------------------------------------")
    end
    for iteration = 1 : SC.maxIterations

        time_reassembly = 0
        time_solver = 0
        time_total = @elapsed for s = 1 : nsubiterations
            if verbosity > 2
                println("\n  Subiteration $s with equations $(SC.subiterations[s])")
            end

            # PREPARE GLOBALCONSTRAINTS
            # known bug: this will only work if no components in front of the constrained component(s)
            # are missing in the subiteration
            for j = 1 : length(PDE.GlobalConstraints)
                if PDE.GlobalConstraints[j].component in SC.subiterations[s]
                   additional_fixed_dofs = apply_constraint!(A[s],b[s],PDE.GlobalConstraints[j],Target; current_equations = SC.subiterations[s], verbosity = SC.verbosity - 2)
                   append!(fixed_dofs, additional_fixed_dofs)
                end
            end

            # PENALIZE FIXED DOFS
            # (from boundary conditions and constraints)
            for j = 1 : length(fixed_dofs)
                for eq = 1 : length(SC.subiterations[s])
                    # check if fixed_dof is necessary for subiteration
                    if fixed_dofs[j] > eqoffsets[s][eq] && fixed_dofs[j] <= eqoffsets[s][eq]+FEs[SC.subiterations[s][eq]].ndofs
                        eqdof = fixed_dofs[j] - eqoffsets[s][eq]
                        #println("fixing dof $eqdof (global $(fixed_dofs[j])) of unknown $eq with value $(Target.entries[fixed_dofs[j]])")
                        b[s][eq][eqdof] = SC.dirichlet_penalty * Target.entries[fixed_dofs[j]]
                        A[s][eq,eq][eqdof,eqdof] = SC.dirichlet_penalty
                    end
                end
            end

            # SOLVE
            time_solver = @elapsed begin
                flush!(A[s].entries)
                if iteration == 1 || (iteration % SC.skip_update[s] == 0 && SC.skip_update[s] != -1)
                    update!(LS[s])
                end
                solve!(LS[s])
            end

            # CHECK LINEAR RESIDUAL
            if verbosity > 1
                residual[s].entries[:] = A[s].entries*x[s].entries - b[s].entries
                for j = 1 : length(fixed_dofs)
                    for eq = 1 : length(SC.subiterations[s])
                        # check if fixed_dof is necessary for subiteration
                        if fixed_dofs[j] > eqoffsets[s][eq] && fixed_dofs[j] <= eqoffsets[s][eq]+FEs[SC.subiterations[s][eq]].ndofs
                            eqdof = fixed_dofs[j] - eqoffsets[s][eq]
                            residual[s][eq][eqdof] = 0
                        end
                    end
                end
                linresnorm[s] = (sqrt(sum(residual[s].entries.^2, dims = 1)[1]))
            end

            # WRITE INTO Target
            for j = 1 : length(SC.subiterations[s])
                for k = 1 : length(Target[SC.subiterations[s][j]])
                    Target[SC.subiterations[s][j]][k] = x[s][j][k]
                end
            end

            # REASSEMBLE PARTS FOR NEXT SUBITERATION
            time_reassembly += @elapsed begin
                next_eq = (s == nsubiterations) ? 1 : s+1
                assemble!(A[next_eq],b[next_eq],PDE,SC,Target; time = time, equations = SC.subiterations[next_eq], min_trigger = AssemblyEachTimeStep, verbosity = SC.verbosity - 2)
            end

        end

        # CHECK NONLINEAR RESIDUAL
        for s = 1 : nsubiterations
            residual[s].entries[:] = A[s].entries*x[s].entries - b[s].entries
            for j = 1 : length(fixed_dofs)
                for eq = 1 : length(SC.subiterations[s])
                    if fixed_dofs[j] > eqoffsets[s][eq] && fixed_dofs[j] <= eqoffsets[s][eq]+FEs[SC.subiterations[s][eq]].ndofs
                        eqdof = fixed_dofs[j] - eqoffsets[s][eq]
                        residual[s][eq][eqdof] = 0
                    end
                end
            end
            resnorm[s] = (sqrt(sum(residual[s].entries.^2, dims = 1)[1]))
        end

        if verbosity > 1
            @printf("\n     %4d  ", iteration)
            @printf(" | %e", sqrt(sum(linresnorm.^2)))
            @printf(" | %e", sqrt(sum(resnorm.^2)))
            @printf(" | %.2e/%.2e/%.2e",time_reassembly,time_solver, time_total)
        end

        if iteration == SC.maxIterations
            if verbosity > 0
                println("  terminated (maxIterations reached)")
                break
            end
        end

        if sqrt(sum(resnorm.^2)) < SC.maxResidual
            if verbosity > 0
                println("  converged (maxResidual of $(SC.maxResidual) reached)")
            end
            break
        end
    end

    # REALIZE GLOBALCONSTRAINTS 
    # (possibly changes some entries of Target)
    for j = 1 : length(PDE.GlobalConstraints)
        realize_constraint!(Target,PDE.GlobalConstraints[j]; verbosity = verbosity - 2)
    end

    if verbosity > 2
        show_statistics(stdout,PDE,SC)
    end

    return sqrt(sum(resnorm.^2))
end

"""
````
function solve!(
    Target::FEVector,
    PDE::PDEDescription;
    subiterations = "auto",
    dirichlet_penalty::Real = 1e60,
    time::Real = 0,
    maxResidual::Real = 1e-12,
    maxIterations::Int = 10,
    linsolver = "UMFPACK",          # or Type{<:AbstractLinearSystem}
    skip_update = [1],
    anderson_iterations = 0, #  0 = Picard iteration, >0 Anderson iteration (experimental feature)
    verbosity::Int = 0)
````

Solves a given PDE (provided as a PDEDescription) and writes the solution into the FEVector Target. The ansatz spaces are taken from the of this vector.

Further optional arguments:
- subiterations :  specifies subsets of equations that are solved together in the given order; "auto" tries to solve the whole system at once
- dirichlet_penalty : Dirichlet data is enforced by penalties on the diagonal of the matrix.
- time : If time-dependent data is involved, the time can be fixed to some value here.
- linsolver : specifies the linear solver that is used to solver the linear system of equation in each fixpoint iteration, "UMFPACK" will use the default direct solver
- skip_update : specifies after how much iterations the lu decomposition should be recomputed (-1 = only once if e.g. the matrix stays the same for each fixpoint iteration)

Depending on the subiterations and detected/configured nonlinearties the whole system is either solved directly in one step
or via a fixed-point iteration.

"""
function solve!(
    Target::FEVector{T},
    PDE::PDEDescription;
    subiterations = "auto",
    dirichlet_penalty::Real = 1e60,
    time::Real = 0,
    maxResidual::Real = 1e-12,
    maxIterations::Int = 10,
    linsolver = "UMFPACK",
    skip_update = [1],
    damping = 0,
    anderson_iterations = 0, #  0 = Picard iteration, >0 Anderson iteration
    verbosity::Int = 0) where {T <: Real}

    SolverConfig = generate_solver(PDE, T; subiterations = subiterations, verbosity = verbosity)
    SolverConfig.dirichlet_penalty = dirichlet_penalty
    SolverConfig.anderson_iterations = anderson_iterations
    if linsolver == "UMFPACK"
        SolverConfig.linsolver = LinearSystemDirectUMFPACK{T,verbosity-1}
    else
        SolverConfig.linsolver = linsolver
    end
    SolverConfig.skip_update = skip_update
    SolverConfig.damping = damping
    SolverConfig.maxResidual = maxResidual
    SolverConfig.maxIterations = maxIterations

    if verbosity >= 0
        if SolverConfig.is_timedependent
            @info "Solving $(PDE.name) (at fixed time $time)"
        else
            @info "Solving $(PDE.name)"
        end
        for s = 1 : length(SolverConfig.subiterations), o = 1 : length(SolverConfig.subiterations[s])
            d = SolverConfig.subiterations[s][o]
            @info "\tSubIt/Eq $s/$d : $(PDE.unknown_names[d]) >> $(Target[d].name) ($(Target[d].FES.name), ndofs = $(Target[d].FES.ndofs))"
        end
        if verbosity > 2
            @show SolverConfig
        end
    end



    # check if PDE can be solved directly
    if subiterations == "auto"
        if SolverConfig.is_nonlinear == false
            residual = solve_direct!(Target, PDE, SolverConfig; time = time)
        else
            residual = solve_fixpoint_full!(Target, PDE, SolverConfig; time = time)
        end
    else
        residual = solve_fixpoint_subiterations!(Target, PDE, SolverConfig; time = time)
    end

    if verbosity >= 0
        if residual > SolverConfig.maxResidual
            @warn "\tfinished solve with residual = $residual\n\t\t(warning because residual is larger than $maxResidual)"
        else
            @info "\tfinished solve with residual = $residual"
        end
    end
    return residual
end

#########################
# TIME-DEPENDENT SOLVER #
#########################


abstract type AbstractTimeIntegrationRule end
abstract type BackwardEuler <: AbstractTimeIntegrationRule end

mutable struct TimeControlSolver{TIR<:AbstractTimeIntegrationRule}
    PDE::PDEDescription      # PDE description (operators, data etc.)
    SC::SolverConfig         # solver configurations (subiterations, penalties etc.)
    LS::Array{AbstractLinearSystem,1} # array for linear solvers of all subiterations
    ctime::Real              # current time
    cstep::Real              # current timestep count
    last_timestep::Real      # last timestep
    AM::Array{FEMatrix,1}    # heap for mass matrices for each equation package
    which::Array{Int,1}      # which equations shall have a time derivative?
    nonlinear_dt::Array{Bool,1} # if true mass matrix is recomputed in each iteration
    dt_operators::Array{DataType,1} # operators associated with the time derivative
    dt_actions::Array{<:AbstractAction,1} # actions associated with the time derivative
    A::Array{FEMatrix,1}     # heap for system matrix for each equation package
    b::Array{FEVector,1}     # heap for system rhs for each equation package
    x::Array{FEVector,1}     # heap for current solution for each equation package
    res::Array{FEVector,1}   # residual vector
    X::FEVector              # full solution vector
    LastIterate::FEVector    # helper variable if nonlinear_iterations > 1
    fixed_dofs::Array{Int,1}            # fixed dof numbes (values are stored in X)
    eqoffsets::Array{Array{Int,1},1}    # offsets for subblocks of each equation package
    ALU::Array{Any,1}  # LU decompositions of matrices A
end



function assemble_massmatrix4subiteration!(TCS::TimeControlSolver, i::Int; verbosity::Int = 0, force::Bool = false)
    for j = 1 : length(TCS.SC.subiterations[i])
        if (TCS.SC.subiterations[i][j] in TCS.which) == true 

            # assembly mass matrix into AM block
            pos = findall(x->x==TCS.SC.subiterations[i][j], TCS.which)[1]

            if TCS.nonlinear_dt[pos] == true || force == true
                if verbosity > 1
                    println("\n  Assembling mass matrix for block [$j,$j] of subiteration $i with action of type $(typeof(TCS.dt_actions[pos]))...")
                end
                A = TCS.AM[i][j,j]
                FE1 = A.FESX
                FE2 = A.FESY
                operator1 = TCS.dt_operators[pos]
                operator2 = TCS.dt_operators[pos]
                BLF = SymmetricBilinearForm(Float64, ON_CELLS, [FE1, FE2], [operator1, operator2], TCS.dt_actions[pos])    
                time = @elapsed assemble!(A, BLF; verbosity = verbosity - 1, skip_preps = false)
                if verbosity > 1
                    println("   mass matrix assembly time = $time")
                end
            end
        end
    end
    flush!(TCS.AM[i].entries)
end


"""
````
function TimeControlSolver(
    PDE::PDEDescription,
    Solution::FEVector,    # contains initial values and stores solution of advance method below
    TIR::Type{<:AbstractTimeIntegrationRule} = BackwardEuler;
    timedependent_equations = [],
    subiterations = "auto",
    start_time::Real = 0,
    verbosity::Int = 0,
    dt_testfunction_operator = [],
    dt_action = [],
    nonlinear_dt::Bool = false,
    dirichlet_penalty = 1e60)
````

Creates a time-dependent solver that can be advanced in time with advance!.
The FEVector Solution stores the initial state but also the solutions of each timestep that are computed via any of the advance! methods.
The argument TIR carries the time integration rule (currently there is only BackwardEuler).

Further optional arguments (that are not listed in the description of solve!):
- timedependent_equations : contains the equation numbers (=rows in PDEDescription) that are time-dependent and should get a time derivative (currently only time derivatives of order 1)
- start_time : initial time
- dt_test_function_operator : operator applied to testfunctions in time derivative
- dt_action : additional actions that are applied to the ansatz function in the time derivative (to include parameters etc.)

"""
function TimeControlSolver(
    PDE::PDEDescription,
    InitialValues::FEVector,    # contains initial values and stores solution of advance! methods
    TIR::Type{<:AbstractTimeIntegrationRule} = BackwardEuler;
    timedependent_equations = [],
    subiterations = "auto",
    start_time::Real = 0,
    verbosity::Int = 0,
    dt_testfunction_operator = [],
    dt_action = [],
    nonlinear_dt = [],
    nonlinear_iterations::Int = 1,
    check_nonlinear_residual::Bool = false,
    maxResidual = 1e-12,
    linsolver = "UMFPACK",
    skip_update = [1],
    dirichlet_penalty = 1e60)

    # generate solver for time-independent problem
    SC = generate_solver(PDE; subiterations = subiterations, verbosity = verbosity)
    for j = 1 : length(InitialValues.FEVectorBlocks)
        if (SC.RHS_AssemblyTriggers[j] <: AssemblyEachTimeStep)
            SC.RHS_AssemblyTriggers[j] = AssemblyEachTimeStep
        end
    end
    SC.dirichlet_penalty = dirichlet_penalty
    SC.maxIterations = nonlinear_iterations
    SC.maxResidual = maxResidual
    SC.check_nonlinear_residual = check_nonlinear_residual
    if linsolver == "UMFPACK"
        SC.linsolver = LinearSystemDirectUMFPACK{Float64,verbosity-1}
    else
        SC.linsolver = linsolver
    end
    SC.skip_update = skip_update



    if verbosity >= 0
        @info "Preparing time control solver for $(PDE.name)"
        for s = 1 : length(SC.subiterations), o = 1 : length(SC.subiterations[s])
            d = SC.subiterations[s][o]
            @info "\tSubIt/Eq $s/$d : $(PDE.unknown_names[d]) >> $(InitialValues[d].name) ($(InitialValues[d].FES.name), ndofs = $(InitialValues[d].FES.ndofs)), timedependent = $(d in timedependent_equations ? "yes" : "no")"
        end
        if verbosity > 2
            @show SolverConfig
        end
    end

    # allocate matrices etc
    FEs = Array{FESpace,1}([])
    for j = 1 : length(InitialValues.FEVectorBlocks)
        push!(FEs,InitialValues.FEVectorBlocks[j].FES)
    end    
    nsubiterations = length(SC.subiterations)
    eqoffsets = Array{Array{Int,1},1}(undef,nsubiterations)
    AM = Array{FEMatrix{Float64},1}(undef,nsubiterations)
    A = Array{FEMatrix{Float64},1}(undef,nsubiterations)
    b = Array{FEVector{Float64},1}(undef,nsubiterations)
    x = Array{FEVector{Float64},1}(undef,nsubiterations)
    res = Array{FEVector{Float64},1}(undef,nsubiterations)
    for i = 1 : nsubiterations
        A[i] = FEMatrix{Float64}("SystemMatrix subiteration $i", FEs[SC.subiterations[i]])
        AM[i] = FEMatrix{Float64}("MassMatrix subiteration $i", FEs[SC.subiterations[i]])
        b[i] = FEVector{Float64}("SystemRhs subiteration $i", FEs[SC.subiterations[i]])
        x[i] = FEVector{Float64}("Solution subiteration $i", FEs[SC.subiterations[i]])
        res[i] = FEVector{Float64}("Residual subiteration $i", FEs[SC.subiterations[i]])
        assemble!(A[i],b[i],PDE,SC,InitialValues; time = start_time, equations = SC.subiterations[i], min_trigger = AssemblyInitial, verbosity = verbosity - 2)
        eqoffsets[i] = zeros(Int,length(SC.subiterations[i]))
        for j= 1 : length(FEs), eq = 1 : length(SC.subiterations[i])
            if j < SC.subiterations[i][eq]
                eqoffsets[i][eq] += FEs[j].ndofs
            end
        end
    end


    # ASSEMBLE BOUNDARY DATA
    # will be overwritten in solve if time-dependent
    # but fixed_dofs will remain throughout
    fixed_dofs = []
    for j = 1 : length(InitialValues.FEVectorBlocks)
        if verbosity > 2
            println("\n  Assembling boundary data for block [$j]...")
            @time new_fixed_dofs = boundarydata!(InitialValues[j],PDE.BoundaryOperators[j]; verbosity = verbosity - 2) .+ InitialValues[j].offset
            append!(fixed_dofs, new_fixed_dofs)
        else
            new_fixed_dofs = boundarydata!(InitialValues[j],PDE.BoundaryOperators[j]; verbosity = verbosity - 2) .+ InitialValues[j].offset
            append!(fixed_dofs, new_fixed_dofs)
        end    
    end    

    for s = 1 : nsubiterations

        # PREPARE GLOBALCONSTRAINTS
        # known bug: this will only work if no components in front of the constrained component(s)
        # are missing in the subiteration
        for j = 1 : length(PDE.GlobalConstraints)
            if PDE.GlobalConstraints[j].component in SC.subiterations[s]
                additional_fixed_dofs = apply_constraint!(A[s],b[s],PDE.GlobalConstraints[j],InitialValues; current_equations = SC.subiterations[s], verbosity = verbosity - 2)
                append!(fixed_dofs, additional_fixed_dofs)
            end
        end

        # COPY INITIAL VALUES TO SUB-PROBLEM SOLUTIONS
        for j = 1 : length(x[s].entries), k = 1 : length(SC.subiterations[s])
            d = SC.subiterations[s][k]
            x[s][k][:] = InitialValues[d][:]
        end

        # prepare and configure mass matrices
        for j = 1 : length(SC.subiterations[s])
            d = SC.subiterations[s][j] # equation of subiteration
            if (d in timedependent_equations) == true 
                pos = findall(x->x==d, timedependent_equations)[1] # position in timedependent_equations
                if length(nonlinear_dt) < pos
                    push!(nonlinear_dt, false)
                end
                if length(dt_action) < pos
                    push!(dt_action, DoNotChangeAction(get_ncomponents(eltype(FEs[d]))))
                end
                if length(dt_testfunction_operator) < pos
                    push!(dt_testfunction_operator, Identity)
                end
                # set diagonal equations block and rhs block to nonlinear
                if nonlinear_dt[pos] == true
                    SC.LHS_AssemblyTriggers[d,d] = AssemblyEachTimeStep
                    SC.RHS_AssemblyTriggers[d] = AssemblyEachTimeStep
                end
            end
        end
    end

    dt_action = Array{AbstractAction,1}(dt_action)
    dt_testfunction_operator = Array{DataType,1}(dt_testfunction_operator)

    # INIT LINEAR SOLVERS
    LS = Array{AbstractLinearSystem,1}(undef,nsubiterations)
    for s = 1 : nsubiterations
        if length(SC.skip_update) < s
            push!(SC.skip_update, 1)
        end
        LS[s] = createsolver(SC.linsolver,x[s].entries,A[s].entries,b[s].entries)
    end

    # if nonlinear iterations are performed we need to remember the iterate from last timestep
    if nonlinear_iterations > 1
        # two vectors to store intermediate approximations
        LastIterate = deepcopy(InitialValues)
    else
        # same vector, only one is needed
        LastIterate = InitialValues
    end

    # generate TimeControlSolver
    TCS = TimeControlSolver{TIR}(PDE,SC,LS,start_time,0,0,AM,timedependent_equations,nonlinear_dt,dt_testfunction_operator,dt_action,A,b,x,res,InitialValues, LastIterate, fixed_dofs, eqoffsets, Array{SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64},1}(undef,length(SC.subiterations)))

    # trigger initial assembly of all time derivative mass matrices
    for i = 1 : nsubiterations
        assemble_massmatrix4subiteration!(TCS, i; force = true, verbosity = verbosity - 1)
    end


    return TCS
end

"""
````
function TimeControlSolver(
    advance!(TCS::TimeControlSolver, timestep::Real = 1e-1)
````

Advances a TimeControlSolver one step in time with the given timestep.

"""
function advance!(TCS::TimeControlSolver, timestep::Real = 1e-1)
    # update timestep counter
    TCS.cstep += 1
    TCS.ctime += timestep

    # unpack
    SC = TCS.SC
    PDE = TCS.PDE
    A = TCS.A 
    AM = TCS.AM
    b = TCS.b
    x = TCS.x
    res = TCS.res
    X = TCS.X
    LS = TCS.LS
    LastIterate = TCS.LastIterate
    fixed_dofs = TCS.fixed_dofs
    eqoffsets = TCS.eqoffsets
    T = eltype(res[1].entries)
    statistics = zeros(Float64,length(X),4)

    # save current solution if nonlinear iterations are needed
    # X will then contain the current nonlinear iterate
    # LastIterate contains the solution from last time step (for reassembly of time derivatives)
    if SC.maxIterations > 1
        for j = 1 : length(X.entries)
            LastIterate.entries[j] = X.entries[j]
        end
    end


    ## LOOP OVER ALL SUBITERATIONS
    for s = 1 : length(SC.subiterations)


        # UPDATE SYSTEM
        if SC.skip_update[s] != -1 || TCS.cstep == 1 # update matrix
            fill!(A[s].entries,0)
            fill!(b[s].entries,0)
            assemble!(A[s],b[s],PDE,SC,LastIterate; equations = SC.subiterations[s], time = TCS.ctime, min_trigger = AssemblyInitial, verbosity = SC.verbosity - 2, storage_trigger = AssemblyEachTimeStep)

            ## update mass matrix and add time derivative
            assemble_massmatrix4subiteration!(TCS, s; force = false, verbosity = SC.verbosity - 2)
            for k = 1 : length(SC.subiterations[s])
                d = SC.subiterations[s][k]
                addblock!(A[s][k,k],AM[s][k,k]; factor = 1.0/timestep)
                addblock_matmul!(b[s][k],AM[s][k,k],X[d]; factor = 1.0/timestep)
            end
            flush!(A[s].entries)
        else # only update rhs
            fill!(b[s].entries,0)
            assemble!(A[s],b[s],PDE,SC,LastIterate; equations = SC.subiterations[s], time = TCS.ctime, min_trigger = AssemblyInitial, verbosity = SC.verbosity - 2, storage_trigger = AssemblyEachTimeStep, only_rhs = true)

            ## add time derivative
            for k = 1 : length(SC.subiterations[s])
                d = SC.subiterations[s][k]
                addblock_matmul!(b[s][k],AM[s][k,k],X[d]; factor = 1.0/timestep)
            end
        end

        # ASSEMBLE (TIME-DEPENDENT) BOUNDARY DATA
        for k = 1 : length(SC.subiterations[s])
            d = SC.subiterations[s][k]
            if any(PDE.BoundaryOperators[d].timedependent) == true
                if SC.verbosity > 2
                    println("\n  Assembling boundary data for block [$d]...")
                    boundarydata!(x[s][k],PDE.BoundaryOperators[d]; time = TCS.ctime, verbosity = SC.verbosity - 2)
                else
                    boundarydata!(x[s][k],PDE.BoundaryOperators[d]; time = TCS.ctime, verbosity = SC.verbosity - 2)
                end    
            end
        end    

        ## START (NONLINEAR) ITERATION(S)
        for iteration = 1 : SC.maxIterations
            statistics[s,4] = iteration.^2 # will be square-rooted later

            # PENALIZE FIXED DOFS (IN CASE THE MATRIX CHANGED)
            # (from boundary conditions and global constraints)
            eqdof = 0
            for j = 1 : length(fixed_dofs)
                for eq = 1 : length(SC.subiterations[s])
                    if fixed_dofs[j] > eqoffsets[s][eq] && fixed_dofs[j] <= eqoffsets[s][eq]+X[SC.subiterations[s][eq]].FES.ndofs
                        eqdof = fixed_dofs[j] - eqoffsets[s][eq]
                        b[s][eq][eqdof] = SC.dirichlet_penalty * x[s][eq][eqdof]
                        A[s][eq,eq][eqdof,eqdof] = SC.dirichlet_penalty
                    end
                end
            end

            ## SOLVE for x[s]
            time_solver = @elapsed begin
                flush!(A[s].entries)
                if TCS.cstep == 1 || (TCS.cstep % SC.skip_update[s] == 0 && SC.skip_update[s] != -1)
                    update!(LS[s])
                end
                solve!(LS[s])
            end

            ## CHECK LINEAR RESIDUAL
            res[s].entries[:] = A[s].entries*x[s].entries - b[s].entries
            for j = 1 : length(fixed_dofs)
                for eq = 1 : length(SC.subiterations[s])
                    if fixed_dofs[j] > eqoffsets[s][eq] && fixed_dofs[j] <= eqoffsets[s][eq]+X[SC.subiterations[s][eq]].FES.ndofs
                        eqdof = fixed_dofs[j] - eqoffsets[s][eq]
                        res[s][eq][eqdof] = 0
                    end
                end
            end
            statistics[SC.subiterations[s][:],1] .= 0
            for j = 1 : length(SC.subiterations[s])
                statistics[SC.subiterations[s][j],1] += sum(res[s][j][:].^2)
            end
            linresnorm = norm(res[s].entries)
            if SC.verbosity > 1
                @printf("\n                       ")
                @printf("| %.4e ", linresnorm)
            end

            ## REASSEMBLE NONLINEAR PARTS
            if SC.maxIterations > 1 || SC.check_nonlinear_residual
                lhs_erased, rhs_erased = assemble!(A[s],b[s],PDE,SC,x[s]; equations = SC.subiterations[s], min_trigger = AssemblyAlways, verbosity = SC.verbosity - 2, time = TCS.ctime)

                ## REPAIR TIME DERIVATIVE IF NEEDED
                for k = 1 : length(SC.subiterations[s])
                    d = SC.subiterations[s][k]
                    if lhs_erased[k,k]
                        addblock!(A[s][k,k],AM[s][k,k]; factor = 1.0/timestep)
                    end
                    if rhs_erased[k]
                        addblock_matmul!(b[s][k],AM[s][k,k],LastIterate[d]; factor = 1.0/timestep)
                    end
                end

                # CHECK NONLINEAR RESIDUAL
                if sum(lhs_erased[:]) + sum(rhs_erased) > 0
                    flush!(A[s].entries)
                    res[s].entries[:] = A[s].entries*x[s].entries - b[s].entries
                    for j = 1 : length(fixed_dofs)
                        for eq = 1 : length(SC.subiterations[s])
                            if fixed_dofs[j] > eqoffsets[s][eq] && fixed_dofs[j] <= eqoffsets[s][eq]+X[SC.subiterations[s][eq]].FES.ndofs
                                eqdof = fixed_dofs[j] - eqoffsets[s][eq]
                                res[s][eq][eqdof] = 0
                            end
                        end
                    end
                    statistics[SC.subiterations[s][:],3] .= 0
                    for j = 1 : length(SC.subiterations[s])
                        statistics[SC.subiterations[s][j],3] += sum(res[s][j][:].^2)
                    end
                    resnorm = norm(res[s].entries)
                    if SC.verbosity > 1
                        @printf("| %.4e (%d) |", resnorm, iteration)
                    end
                else
                    statistics[SC.subiterations[s][:],3] .= statistics[SC.subiterations[s][:],1]
                end
            else
                statistics[SC.subiterations[s][:],3] .= 1e99 # nonlinear residual has not been checked !
            end
        end

        # WRITE x[s] INTO X and COMPUTE CHANGE
        for j = 1 : length(SC.subiterations[s])
            for k = 1 : length(LastIterate[SC.subiterations[s][j]])
                statistics[SC.subiterations[s][j],2] += (LastIterate[SC.subiterations[s][j]][k] - x[s][j][k])^2
                X[SC.subiterations[s][j]][k] = x[s][j][k]
            end
        end
    end
    
    # REALIZE GLOBAL GLOBALCONSTRAINTS 
    # known bug: this will only work if no components in front of the constrained component(s)
    # are missing in the subiteration
    for j = 1 : length(PDE.GlobalConstraints)
        realize_constraint!(X,PDE.GlobalConstraints[j]; verbosity = SC.verbosity - 2)
    end

    TCS.last_timestep = timestep

    return sqrt.(statistics)
end

function center_string(S::String, L::Int = 8)
    if length(S) > L
        S = S[1:L]
    end
    while length(S) < L-1
        S = " " * S * " "
    end
    if length(S) < L
        S = " " * S
    end
    return S
end

"""
````
advance_until_stationarity!(TCS::TimeControlSolver, timestep; stationarity_threshold = 1e-11, maxTimeSteps = 100, do_after_each_timestep = nothing)
````

Advances a TimeControlSolver in time with the given (initial) timestep until stationarity is detected (change of variables below threshold) or a maximal number of time steps is exceeded.
The function do_after_timestep is called after each timestep and can be used to print/save data (and maybe timestep control in future).
"""
function advance_until_stationarity!(TCS::TimeControlSolver, timestep; stationarity_threshold = 1e-11, maxTimeSteps = 100, do_after_each_timestep = nothing)
    statistics = zeros(Float64,length(TCS.X),3)
    if TCS.SC.verbosity >= 0
        @info "Advancing in time until stationarity..."
    end
    if TCS.SC.verbosity > 0
        if TCS.SC.maxIterations > 1 || TCS.SC.check_nonlinear_residual
            @printf("\n    STEP  |    TIME    | LSRESIDUAL |   NLRESIDUAL   |   CHANGE ")
            for j = 1 : size(statistics,1)
                @printf("        ")
            end
            if do_after_each_timestep != nothing
                do_after_each_timestep(0, statistics)
            end
            @printf("\n          |            |  (total)   |    (total)     |")
        else
            @printf("\n    STEP  |    TIME    | LSRESIDUAL |   CHANGE ")
            for j = 1 : size(statistics,1)
                @printf("        ")
            end
            if do_after_each_timestep != nothing
                do_after_each_timestep(0, statistics)
            end
            @printf("\n          |            |  (total)   |")
        end
        for j = 1 : size(statistics,1)
            @printf(" %s ",center_string(TCS.PDE.unknown_names[j],10))
        end
    end
    for iteration = 1 : maxTimeSteps
        statistics = advance!(TCS, timestep)
        if TCS.SC.verbosity > 0
            @printf("\n    %4d  ",iteration)
            @printf("| %.4e ",TCS.ctime)
            @printf("| %.4e |",sqrt(sum(statistics[:,1].^2)))
            if TCS.SC.maxIterations > 1 || TCS.SC.check_nonlinear_residual
                @printf(" %.4e (%d) |",sqrt(sum(statistics[:,3].^2)), statistics[1,4])
            end
            for j = 1 : size(statistics,1)
                @printf(" %.4e ",statistics[j,2])
            end
        end
        if do_after_each_timestep != nothing
            do_after_each_timestep(TCS.cstep, statistics)
        end
        if sum(statistics[:,2]) < stationarity_threshold
            println("\n  stationarity detected after $iteration timesteps")
            break;
        end
        if iteration == maxTimeSteps 
            println("  terminated (maxTimeSteps reached)")
        end
    end

    if TCS.SC.verbosity > 2
        show_statistics(stdout,TCS.PDE,TCS.SC)
    end
end


"""
````
advance_until_time!(TCS::TimeControlSolver, timestep, finaltime; finaltime_tolerance = 1e-15, do_after_each_timestep = nothing)
````

Advances a TimeControlSolver in time with the given (initial) timestep until the specified finaltime is reached (up to the specified tolerance).
The function do_after_timestep is called after each timestep and can be used to print/save data (and maybe timestep control in future).
"""
function advance_until_time!(TCS::TimeControlSolver, timestep, finaltime; finaltime_tolerance = 1e-15, do_after_each_timestep = nothing)
    statistics = zeros(Float64,length(TCS.X),3)
    if TCS.SC.verbosity >= 0
        @info "Advancing in time from $(TCS.ctime) until $finaltime"
    end
    if TCS.SC.verbosity > 0
        if TCS.SC.maxIterations > 1 || TCS.SC.check_nonlinear_residual
            @printf("\n    STEP  |    TIME    | LSRESIDUAL |   NLRESIDUAL   |   CHANGE ")
            for j = 1 : size(statistics,1)
                @printf("        ")
            end
            if do_after_each_timestep != nothing
                do_after_each_timestep(0, statistics)
            end
            @printf("\n          |            |  (total)   |    (total)     |")
        else
            @printf("\n    STEP  |    TIME    | LSRESIDUAL |   CHANGE ")
            for j = 1 : size(statistics,1)
                @printf("        ")
            end
            if do_after_each_timestep != nothing
                do_after_each_timestep(0, statistics)
            end
            @printf("\n          |            |  (total)   |")
        end
        for j = 1 : size(statistics,1)
            @printf(" %s ",center_string(TCS.PDE.unknown_names[j],10))
        end
    end
    while TCS.ctime < finaltime - finaltime_tolerance
        statistics = advance!(TCS, timestep)
        if TCS.SC.verbosity > 0
            @printf("\n    %4d  ",TCS.cstep)
            @printf("| %.4e ",TCS.ctime)
            @printf("| %.4e |",sqrt(sum(statistics[:,1].^2)))
            if TCS.SC.maxIterations > 1 || TCS.SC.check_nonlinear_residual
                @printf(" %.4e (%d) |",sqrt(sum(statistics[:,3].^2)), statistics[1,4])
            end
            for j = 1 : size(statistics,1)
                @printf(" %.4e ",statistics[j,2])
            end
        end
        if do_after_each_timestep != nothing
            do_after_each_timestep(TCS.cstep, statistics)
        end
    end
    if TCS.SC.verbosity > 0
        @printf("\n\n  arrived at time T = %.4e...\n",TCS.ctime)
    end

    if TCS.SC.verbosity > 2
        show_statistics(stdout,TCS.PDE,TCS.SC)
    end
end


"""
````
advance_until_time!(DiffEQ::Module, TCS::TimeControlSolver, timestep, finaltime; solver = nothing, abstol = 1e-1, reltol = 1e-1, dtmin = 0, adaptive::Bool = true)
````

Advances a TimeControlSolver in time with the given (initial) timestep until the specified finaltime is reached (up to the specified tolerance)
with the given exterior time integration module. The only valid Module here is DifferentialEquations.jl and the optional arguments are passed to it.
If solver == nothing the solver Rosenbrock23(autodiff = false) will be chosen. For more choices please consult the documentation of DifferentialEquations.jl.

Also note that this is a highly experimental feature and will not work for general TimeControlSolvers configuration (e.g. in the case of several subiterations or, it seems,
saddle point problems). Also have a look at corressponding the example in the advanced examples section.
"""
function advance_until_time!(DiffEQ::Module, sys::TimeControlSolver, timestep, finaltime; solver = nothing, abstol = 1e-1, reltol = 1e-1, dtmin = 0, adaptive::Bool = true)
    if solver == nothing 
        solver = DiffEQ.Rosenbrock23(autodiff = false)
    end

    if sys.SC.verbosity >= 0
        @info "Advancing in time from $(sys.ctime) until $finaltime using $DiffEQ with solver = $solver"
    end

    ## generate ODE problem
    f = DiffEQ.ODEFunction(eval_rhs!, jac=eval_jacobian!, jac_prototype=jac_prototype(sys), mass_matrix=mass_matrix(sys))
    prob = DiffEQ.ODEProblem(f,sys.X.entries, (sys.ctime,finaltime),sys)

    ## solve ODE problem
    sol = DiffEQ.solve(prob,solver, abstol=abstol, reltol=reltol, dt = timestep, dtmin = dtmin, initializealg=DiffEQ.NoInit(), adaptive = adaptive)

    ## pickup solution at final time
    sys.X.entries .= sol[:,end]
end