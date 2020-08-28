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


mutable struct SolverConfig
    is_nonlinear::Bool      # PDE is nonlinear
    is_timedependent::Bool  # PDE is time_dependent
    LHS_AssemblyTriggers::Array{DataType,2} # assembly triggers for blocks in LHS
    LHS_dependencies::Array{Array{Int,1},2} # dependencies on components
    RHS_AssemblyTriggers::Array{DataType,1} # assembly triggers for blocks in RHS
    RHS_dependencies::Array{Array{Int,1},1} # dependencies on components
    subiterations::Array{Array{Int,1},1} # combination of equations (= rows in PDE.LHS) that should be solved together
    maxIterations::Int          # maximum number of iterations
    maxResidual::Real           # tolerance for residual
    current_time::Real          # current time in a time-dependent setting
    dirichlet_penalty::Real     # penalty for Dirichlet data
    anderson_iterations::Int    # number of Anderson iterations (= 0 normal fixed-point iterations, > 0 Anderson iterations)
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
        return SolverConfig(nonlinear, timedependent, LHS_ATs, LHS_dep, RHS_ATs, RHS_dep, subiterations, 10, 1e-10, 0.0, 1e60, 0, verbosity)
    else
        return SolverConfig(nonlinear, timedependent, LHS_ATs, LHS_dep, RHS_ATs, RHS_dep, [1:size(PDE.LHSOperators,1)], 10, 1e-10, 0.0, 1e60, 0, verbosity)
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
        elseif SC.RHS_AssemblyTriggers[j] == AssemblyNever
            print(" N ")
        end
        println("")
    end
    println("                     (I = Once, T = EachTimeStep, A = Always, N = Never)")
    println("\n LHS_dependencies = $(SC.LHS_dependencies)\n")

end


function assemble!(
    A::FEMatrix,
    b::FEVector,
    PDE::PDEDescription,
    SC::SolverConfig,
    CurrentSolution::FEVector;
    time::Real = 0,
    equations = [],
    if_depends_on = [], # block is only assembled if it depends on these components
    min_trigger::Type{<:AbstractAssemblyTrigger} = AssemblyAlways,
    verbosity::Int = 0)

    if length(equations) == 0
        equations = 1:size(PDE.LHSOperators,1)
    end
    if length(if_depends_on) == 0
        if_depends_on = 1:size(PDE.LHSOperators,1)
    end

    if verbosity > 0
        println("\n  Entering assembly of equations=$equations (min_trigger = $min_trigger)")
    end

    # important to flush first in case there is some cached stuff that
    # will not be seen by fill! functions
    flush!(A.entries)

    # force (re)assembly of stored bilinearforms and RhsOperators
    if (min_trigger == AssemblyInitial) == true
        for j = 1:length(equations)
            for k = 1 : size(PDE.LHSOperators,2)
                for o = 1 : length(PDE.LHSOperators[equations[j],k])
                    PDEoperator = PDE.LHSOperators[equations[j],k][o]
                    try
                        if PDEoperator.store_operator == true
                            if verbosity > 0
                                println("  Updating storage of operator $(PDEoperator.name)")
                            end
                        else
                            break
                        end
                    catch
                        break
                    end
                    update_storage!(PDEoperator, CurrentSolution, j, k ; time = time, verbosity = verbosity)
                end
            end
        end
        for j = 1:length(equations)
            for o = 1 : length(PDE.RHSOperators[equations[j]])
                PDEoperator = PDE.RHSOperators[equations[j]][o]
                try
                    if PDEoperator.store_operator == true
                        if verbosity > 0
                            println("  Updating storage of operator $(PDEoperator.name)")
                        end
                    else
                        break
                    end
                catch
                    break
                end
                update_storage!(PDEoperator, CurrentSolution, j ; time = time, verbosity = verbosity)
            end
        end
    end

    # (re)assembly right-hand side
    rhs_block_has_been_erased = Array{Bool,1}(undef,length(equations))
    for j = 1 : length(equations)
        rhs_block_has_been_erased[j] = false
        if (min_trigger <: SC.RHS_AssemblyTriggers[equations[j]]) == true
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

    # (re)assembly left-hand side
    subblock = 0
    for j = 1:length(equations)
        for k = 1 : size(PDE.LHSOperators,2)
            if length(intersect(SC.LHS_dependencies[equations[j],k], if_depends_on)) > 0
                if (k in equations)
                    subblock += 1
                    #println("\n  Equation $j, subblock $subblock")
                    if (min_trigger <: SC.LHS_AssemblyTriggers[equations[j],k]) == true
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
                else
                    if (min_trigger <: SC.LHS_AssemblyTriggers[equations[j],k]) == true
                        if (length(PDE.LHSOperators[equations[j],k]) > 0) && (!(min_trigger <: SC.RHS_AssemblyTriggers[equations[j]]))
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
        end
        subblock = 0
    end
end

# for linear, stationary PDEs that can be solved in one step
function solve_direct!(Target::FEVector, PDE::PDEDescription, SC::SolverConfig; time::Real = 0)

    verbosity = SC.verbosity

    FEs = Array{FESpace,1}([])
    for j=1 : length(Target.FEVectorBlocks)
        push!(FEs,Target.FEVectorBlocks[j].FES)
    end    

    # ASSEMBLE SYSTEM
    A = FEMatrix{Float64}("SystemMatrix", FEs)
    b = FEVector{Float64}("SystemRhs", FEs)
    assemble!(A,b,PDE,SC,Target; equations = Array{Int,1}(1:length(FEs)), min_trigger = AssemblyInitial, time = time, verbosity = verbosity - 1)

    # ASSEMBLE BOUNDARY DATA
    fixed_dofs = []
    for j= 1 : length(Target.FEVectorBlocks)
        if verbosity > 1
            println("\n  Assembling boundary data for block [$j]...")
            @time new_fixed_dofs = boundarydata!(Target[j],PDE.BoundaryOperators[j]; time = time, verbosity = verbosity - 1)
        else
            new_fixed_dofs = boundarydata!(Target[j],PDE.BoundaryOperators[j]; time = time, verbosity = verbosity - 1)
        end    
        new_fixed_dofs .+= Target[j].offset
        append!(fixed_dofs, new_fixed_dofs)
    end    

    # PREPARE GLOBALCONSTRAINTS
    flush!(A.entries)
    for j = 1 : length(PDE.GlobalConstraints)
        additional_fixed_dofs = apply_constraint!(A,b,PDE.GlobalConstraints[j],Target; verbosity = verbosity - 1)
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
function solve_fixpoint_full!(Target::FEVector, PDE::PDEDescription, SC::SolverConfig; time::Real = 0)

    verbosity = SC.verbosity
    anderson_iterations = SC.anderson_iterations

    FEs = Array{FESpace,1}([])
    for j=1 : length(Target.FEVectorBlocks)
        push!(FEs,Target.FEVectorBlocks[j].FES)
    end    

    # ASSEMBLE SYSTEM INIT
    A = FEMatrix{Float64}("SystemMatrix", FEs)
    b = FEVector{Float64}("SystemRhs", FEs)
    assemble!(A,b,PDE,SC,Target; time = time, equations = Array{Int,1}(1:length(FEs)), min_trigger = AssemblyInitial, verbosity = verbosity - 2)

    # ASSEMBLE BOUNDARY DATA
    fixed_dofs = []
    for j= 1 : length(Target.FEVectorBlocks)
        if verbosity > 1
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
        AIOMatrix = zeros(Float64, anderson_iterations+2, anderson_iterations+2)
        AIORhs = zeros(Float64, anderson_iterations+2)
        AIORhs[anderson_iterations+2] = 1 # sum of all coefficients should equal 1
        AIOalpha = zeros(Float64, anderson_iterations+2)
        for j = 1 : anderson_iterations+1
            LastAndersonIterates[j] = deepcopy(Target)
            LastAndersonIteratesTilde[j] = deepcopy(Target)
            AIOMatrix[j,anderson_iterations+2] = 1
            AIOMatrix[anderson_iterations+2,j] = 1
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
        fixed_dofs = Base.unique(fixed_dofs)
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


        # POSTPROCESS NEW ANDERSON ITERATE
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


        # REASSEMBLE NONLINEAR PARTS
        for j = 1:size(PDE.RHSOperators,1)
            assemble!(A,b,PDE,SC,Target; time = time, equations = [j], min_trigger = AssemblyAlways, verbosity = verbosity - 2)
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
function solve_fixpoint_subiterations!(Target::FEVector, PDE::PDEDescription, SC::SolverConfig; time::Real = 0)

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
    for j= 1 : length(Target.FEVectorBlocks)
        if verbosity > 1
            println("\n  Assembling boundary data for block [$j]...")
            @time new_fixed_dofs = boundarydata!(Target[j],PDE.BoundaryOperators[j]; time = time, verbosity = verbosity - 2) .+ Target[j].offset
            append!(fixed_dofs, new_fixed_dofs)
        else
            new_fixed_dofs = boundarydata!(Target[j],PDE.BoundaryOperators[j]; time = time, verbosity = verbosity - 2) .+ Target[j].offset
            append!(fixed_dofs, new_fixed_dofs)
        end    
    end    
    
    residual = Array{FEVector{Float64},1}(undef,nsubiterations)
    for s = 1 : nsubiterations
        residual[s] = FEVector{Float64}("residual subiteration $s", FEs[SC.subiterations[s]])
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

            # PREPARE GLOBALCONSTRAINTS
            # known bug: this will only work if no components in front of the constrained component(s)
            # are missing in the subiteration
            for j = 1 : length(PDE.GlobalConstraints)
                if PDE.GlobalConstraints[j].component in SC.subiterations[s]
                   additional_fixed_dofs = apply_constraint!(A[s],b[s],PDE.GlobalConstraints[j],Target; verbosity = SC.verbosity - 1)
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

            # REASSEMBLE PARTS FOR NEXT SUBITERATION
            next_eq = (s == nsubiterations) ? 1 : s+1
            assemble!(A[next_eq],b[next_eq],PDE,SC,Target; time = time, equations = SC.subiterations[next_eq], min_trigger = AssemblyEachTimeStep, verbosity = SC.verbosity - 1)

        end

        # CHECK RESIDUAL
        resnorm = 0.0
        for s = 1 : nsubiterations
            residual[s].entries[:] = (A[s].entries*x[s].entries - b[s].entries).^2
            for j = 1 : length(fixed_dofs)
                for eq = 1 : length(SC.subiterations[s])
                    if fixed_dofs[j] > eqoffsets[s][eq] && fixed_dofs[j] <= eqoffsets[s][eq]+FEs[SC.subiterations[s][eq]].ndofs
                        eqdof = fixed_dofs[j] - eqoffsets[s][eq]
                        residual[s][eq][eqdof] = 0
                    end
                end
            end
            resnorm += (sqrt(sum(residual[s].entries, dims = 1)[1]))
            if verbosity > 1
                println("  residual[$s] = $(sqrt(sum(residual[s].entries, dims = 1)[1]))")
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

"""
$(TYPEDSIGNATURES)

Solves a given PDE (provided as a PDEDescription) and writes the solution into the FEVector Target. The ansatz spaces are taken from the of this vector.
The field subiterations specifies subsets of equations that are solved together in the given order; "auto" tries to solve the whole system at once, but also might resort to a fixed-point iteration if a nonlinearity is detected.
"""
function solve!(
    Target::FEVector,
    PDE::PDEDescription;
    subiterations = "auto",
    dirichlet_penalty::Real = 1e60,
    time::Real = 0,
    maxResidual::Real = 1e-12,
    maxIterations::Int = 10,
    AndersonIterations = 0, #  0 = Picard iteration, >0 Anderson iteration
    verbosity::Int = 0)

    SolverConfig = generate_solver(PDE; subiterations = subiterations, verbosity = verbosity)
    SolverConfig.dirichlet_penalty = dirichlet_penalty
    SolverConfig.anderson_iterations = AndersonIterations

    if verbosity > 0
        println("\nSOLVING PDE")
        println("===========")
        println("  name = $(PDE.name)")
        println("  time = $(time)")
        if AndersonIterations > 0
            println("  anderson = $AndersonIterations")
        end

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
            solve_direct!(Target, PDE, SolverConfig; time = time)
        else
            solve_fixpoint_full!(Target, PDE, SolverConfig; time = time)
        end
    else
        solve_fixpoint_subiterations!(Target, PDE, SolverConfig; time = time)
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
    AM::Array{FEMatrix,1}    # heap for mass matrices for each equation package
    which::Array{Int,1}      # which equations shall have a time derivative?
    nonlinear_dt::Array{Bool,1} # if true mass matrix is recomputed in each iteration
    dt_operators::Array{DataType,1} # operators associated with the time derivative
    dt_actions::Array{<:AbstractAction,1} # actions associated with the time derivative
    A::Array{FEMatrix,1}     # heap for system matrix for each equation package
    b::Array{FEVector,1}     # heap for system rhs for each equation package
    x::Array{FEVector,1}     # heap for current solution for each equation package
    X::FEVector              # full solution vector
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
                    @time assemble!(TCS.AM[i][j,j],TCS.X,ReactionOperator(TCS.dt_actions[pos]; identity_operator = TCS.dt_operators[pos]); verbosity = verbosity - 2)
                else
                    assemble!(TCS.AM[i][j,j],TCS.X,ReactionOperator(TCS.dt_actions[pos]; identity_operator = TCS.dt_operators[pos]); verbosity = verbosity - 2)
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
The FEVector Solution stores the initial state but also the solutions of each timestep.
The field timedependent_equations contains the equation numbers (=rows in PDEDescription) that are time-dependent.
The field subiterations specifies subsets of equations that are solved together in the given order; "auto" tries to solve the whole system at once.
"""
function TimeControlSolver(
    PDE::PDEDescription,
    InitialValues::FEVector,    # contains initial values and stores solution of advance method below
    TIR::Type{<:AbstractTimeIntegrationRule} = BackwardEuler;
    timedependent_equations = [],
    subiterations = "auto",
    start_time::Real = 0,
    verbosity::Int = 0,
    dt_testfunction_operator = [],
    dt_action = [],
    nonlinear_dt = [],
    dirichlet_penalty = 1e60)

    # generate solver for time-independent problem
    SC = generate_solver(PDE; subiterations = subiterations, verbosity = verbosity)
    for j = 1 : length(InitialValues.FEVectorBlocks)
        if (SC.RHS_AssemblyTriggers[j] <: AssemblyEachTimeStep)
            SC.RHS_AssemblyTriggers[j] = AssemblyEachTimeStep
        end
    end
    SC.dirichlet_penalty = dirichlet_penalty
    SC.maxIterations = 1
    SC.maxResidual = 1e-16

    if verbosity > 0

        println("\nPREPARING TIME-DEPENDENT SOLVER FOR PDE")
        println("=======================================")
        println("    name = $(PDE.name)")
        Base.show(SC)
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
    for i = 1 : nsubiterations
        A[i] = FEMatrix{Float64}("SystemMatrix subiteration $i", FEs[SC.subiterations[i]])
        AM[i] = FEMatrix{Float64}("MassMatrix subiteration $i", FEs[SC.subiterations[i]])
        b[i] = FEVector{Float64}("SystemRhs subiteration $i", FEs[SC.subiterations[i]])
        x[i] = FEVector{Float64}("Solution subiteration $i", FEs[SC.subiterations[i]])
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
    fixed_dofs = []
    for j = 1 : length(InitialValues.FEVectorBlocks)
        if verbosity > 1
            println("\n  Assembling boundary data for block [$j]...")
            @time new_fixed_dofs = boundarydata!(InitialValues[j],PDE.BoundaryOperators[j]; verbosity = verbosity - 2) .+ InitialValues[j].offset
            append!(fixed_dofs, new_fixed_dofs)
        else
            new_fixed_dofs = boundarydata!(InitialValues[j],PDE.BoundaryOperators[j]; verbosity = verbosity - 2) .+ InitialValues[j].offset
            append!(fixed_dofs, new_fixed_dofs)
        end    
    end    

    # prepare and configure mass matrices
    for i = 1 : nsubiterations # subiterations
        for j = 1 : length(SC.subiterations[i])
            d = SC.subiterations[i][j] # equation of subiteration
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

    # generate TimeControlSolver
    TCS = TimeControlSolver{TIR}(PDE,SC,start_time,0,0,AM,timedependent_equations,nonlinear_dt,dt_testfunction_operator,dt_action,A,b,x,InitialValues, fixed_dofs, eqoffsets, Array{SuiteSparse.UMFPACK.UmfpackLU{Float64,Int32},1}(undef,length(SC.subiterations)))

    # trigger initial assembly of all time derivative mass matrices
    for i = 1 : nsubiterations
        assemble_massmatrix4subiteration!(TCS, i; force = true, verbosity = verbosity - 1)
    end

    return TCS
end

"""
$(TYPEDSIGNATURES)

Advances a TimeControlSolver in time with the given timestep.
"""
function advance!(TCS::TimeControlSolver, timestep::Real = 1e-1; reuse_matrix = [])

    if reuse_matrix == []
        reuse_matrix = zeros(Bool,length(TCS.SC.subiterations))
    end

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
    X = TCS.X

    if SC.verbosity > 2
        println("\n\n\n  Entering timestep $(TCS.cstep)...")
    end

    if (TCS.last_timestep != timestep) && any(reuse_matrix) && (TCS.cstep > 1)
        println("   WARNING: reuse_matrix active, but timestep changed! Results may be wrong!")
    end


    nsubiterations = length(SC.subiterations)
    change = zeros(Float64,length(X))
    d = 0
    l = 0
    for s = 1 : nsubiterations

        if SC.verbosity > 2
            println("\n\n\n  Entering subiteration $s...")
        end

        # (re)assemble mass matrices if needed
        assemble_massmatrix4subiteration!(TCS, s; verbosity = SC.verbosity - 1)

        # add change in mass matrix to diagonal blocks if needed
        for k = 1 : length(SC.subiterations[s])
            d = SC.subiterations[s][k]
            if (reuse_matrix[s] == false) || (TCS.cstep == 1) ## if user claims that matrix should not change this is skipped
                if (AssemblyEachTimeStep <: SC.LHS_AssemblyTriggers[d,d]) == true || TCS.last_timestep == 0 # if block was reassembled at the end of the last iteration
                    if SC.verbosity > 2
                        println("  Adding mass matrix to block [$k,$k] of subiteration $s")
                    end
                    addblock!(A[s][k,k],AM[s][k,k]; factor = 1.0/timestep)
                else
                    if (TCS.last_timestep != timestep) # if block was not reassembled, but timestep changed
                        if SC.verbosity > 2
                            println("  Adding mass matrix change to block [$k,$k] of subiteration $s")
                        end
                        addblock!(A[s][k,k],AM[s][k,k]; factor = -1.0/TCS.last_timestep + 1.0/timestep)
                    end
                end
            end
            if  (AssemblyEachTimeStep <: SC.RHS_AssemblyTriggers[d]) || TCS.last_timestep == 0 # if rhs block was reassembled at the end of the last iteration
                if SC.verbosity > 2
                    println("  Adding time derivative to rhs block [$k] of subiteration $s")
                end
                addblock_matmul!(b[s][k],AM[s][k,k],X[d]; factor = 1.0/timestep)
            else
                if (TCS.last_timestep != timestep) # if block was not reassembled, but timestep changed
                    if SC.verbosity > 2
                        println("  Adding time derivative change to rhs block [$k] of subiteration $s")
                    end
                    addblock_matmul!(b[s][k],AM[s][k,k],x[s][k]; factor = -1.0/TCS.last_timestep) # subtract rhs from last time step
                    addblock_matmul!(b[s][k],AM[s][k,k],X[d]; factor = 1.0/timestep) # add new rhs from last time step
                end
            end
        end
        flush!(A[s].entries)


        # UPDATE TIME-DEPENDENT RHS-DATA
        # rhs evaluated at last time step (ctime - timestep) is already assembled
        # and it depends on the timestep_rule how to update it
        # atm only BackwardEuler is supported so ew have to replace the right-hand side by its evaluation at current ctime
        for k = 1 : length(SC.subiterations[s])
            d = SC.subiterations[s][k]
            for o = 1 : length(PDE.RHSOperators[d])
                if typeof(PDE.RHSOperators[d][o]) <: RhsOperator
                    if PDE.RHSOperators[d][o].timedependent
                        if SC.verbosity > 1
                            println("  Updating time-dependent rhs data of equation $d")
                        end
                        assemble!(b[s][k], X, PDE.RHSOperators[d][o]; factor = -1, time = TCS.ctime - timestep)
                        assemble!(b[s][k], X, PDE.RHSOperators[d][o]; factor = +1, time = TCS.ctime)
                    end
                end    
            end
        end  

        # ASSEMBLE TIME-DEPENDENT BOUNDARY DATA at current (already updated) time ctime
        for k = 1 : length(SC.subiterations[s])
            d = SC.subiterations[s][k]
            if any(PDE.BoundaryOperators[d].timedependent) == true
                if SC.verbosity > 1
                    println("\n  Assembling boundary data for block [$d] at time $(TCS.ctime)...")
                    @time boundarydata!(X[d],PDE.BoundaryOperators[d]; time = TCS.ctime, verbosity = SC.verbosity - 2)
                else
                    boundarydata!(X[d],PDE.BoundaryOperators[d]; time = TCS.ctime, verbosity = SC.verbosity - 2)
                end    
            else
                # nothing todo as all boundary data for block d is time-independent
            end
        end    

        fixed_dofs = TCS.fixed_dofs

        # PREPARE GLOBALCONSTRAINTS
        # known bug: this will only work if no components in front of the constrained component(s)
        # are missing in the subiteration
        for j = 1 : length(PDE.GlobalConstraints)
            if PDE.GlobalConstraints[j].component in SC.subiterations[s]
                additional_fixed_dofs = apply_constraint!(A[s],b[s],PDE.GlobalConstraints[j],X; verbosity = SC.verbosity - 1)
                append!(fixed_dofs, additional_fixed_dofs)
            end
        end

        # PENALIZE FIXED DOFS
        # (from boundary conditions and constraints)
        eqdof = 0
        eqoffsets = TCS.eqoffsets
        for j = 1 : length(fixed_dofs)
            for eq = 1 : length(SC.subiterations[s])
                if fixed_dofs[j] > eqoffsets[s][eq] && fixed_dofs[j] <= eqoffsets[s][eq]+X[SC.subiterations[s][eq]].FES.ndofs
                    eqdof = fixed_dofs[j] - eqoffsets[s][eq]
                    b[s][eq][eqdof] = SC.dirichlet_penalty * X.entries[fixed_dofs[j]]
                    A[s][eq,eq][eqdof,eqdof] = SC.dirichlet_penalty
                end
            end
        end

        if (reuse_matrix[s] == false) || (TCS.cstep == 1)
            flush!(A[s].entries)
            TCS.ALU[s] = lu(A[s].entries.cscmatrix)
        end

        # SOLVE
        if SC.verbosity > 1
            println("\n  Solving equation(s) $(SC.subiterations[s])")
            @time x[s].entries[:] = TCS.ALU[s]\b[s].entries
            residual = A[s].entries*x[s].entries - b[s].entries
            for j = 1 : length(fixed_dofs)
                for eq = 1 : length(SC.subiterations[s])
                    if fixed_dofs[j] > eqoffsets[s][eq] && fixed_dofs[j] <= eqoffsets[s][eq]+X[SC.subiterations[s][eq]].FES.ndofs
                        eqdof = fixed_dofs[j] - eqoffsets[s][eq]
                        residual[eqdof] = 0
                    end
                end
            end
            if SC.verbosity > 1
                println("    ... residual = $(norm(residual))")
            end
        else
            x[s].entries[:] = TCS.ALU[s]\b[s].entries
        end
        

        # REALIZE GLOBAL GLOBALCONSTRAINTS 
        # known bug: this will only work if no components in front of the constrained component(s)
        # are missing in the subiteration
        for j = 1 : length(PDE.GlobalConstraints)
            if PDE.GlobalConstraints[j].component in SC.subiterations[s]
                realize_constraint!(x[s],PDE.GlobalConstraints[j]; verbosity = SC.verbosity - 1)
            end
        end

        # WRITE INTO X and COMPUTE CHANGE
        l = 0
        for j = 1 : length(SC.subiterations[s])
            for k = 1 : length(X[SC.subiterations[s][j]])
                change[SC.subiterations[s][j]] += (X[SC.subiterations[s][j]][k] - x[s][j][k])^2
                X[SC.subiterations[s][j]][k] = x[s][j][k]
            end
            l += length(X[SC.subiterations[s][j]])
        end
        #change[s] /= l^4*timestep^2
        if SC.verbosity > 1
            println("    ... change = $(sqrt(change[s]))")
        end


        # REASSEMBLE PARTS FOR NEXT SUBITERATION
        next_eq = (s == nsubiterations) ? 1 : s+1
        assemble!(A[next_eq],b[next_eq],PDE,SC,X; equations = SC.subiterations[next_eq], min_trigger = AssemblyEachTimeStep, verbosity = SC.verbosity - 1, time = TCS.ctime + timestep)
    end
    
    TCS.last_timestep = timestep

    return sqrt.(change)
end