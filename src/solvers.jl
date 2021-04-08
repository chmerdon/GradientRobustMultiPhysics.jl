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
    user_params::Union{Dict{Symbol,Any},Nothing} # dictionary with user parameters
end

#
# Default context information with help info.
#
default_solver_kwargs()=Dict{Symbol,Tuple{Any,String,Bool,Bool}}(
    :subiterations => ("auto", "an array of equation subsets (each an array) that should be solved together in each fixpoint iteration",true,true),
    :timedependent_equations => ([], "array of the equations that should get a time derivative (only for TimeControlSolver)",false,true),
    :target_residual => (1e-10, "stop fixpoint iterations if the (nonlinear) residual is smaller than this number",true,true),
    :maxiterations => ("auto", "maximal number of nonlinear iterations (TimeControlSolver runs that many in each time step)",true,true),
    :check_nonlinear_residual => ("auto", "check the nonlinear residual in last nonlinear iteration (causes one more reassembly of nonlinear terms)",true,true),
    :time => (0, "time at which time-dependent data functions are evaluated or initial time for TimeControlSolver",true,true),
    :skip_update => ([1], "matrix update (for the j-th sub-iteration) will be performed each skip_update[j] iteration; -1 means only in the first iteration",true,true),
    :damping => (0, "damp the new iteration with this part of the old iteration",true,false),
    :anderson_iterations => (0, "use Anderson acceleration with this many previous iterates (to hopefully speed up/enable convergence of fixpoint iterations)",true,false),
    :anderson_metric => ("l2", "String that encodes the desired convergence metric for the Anderson acceleration (possible values: ''l2'' or ''L2'')",true,false),
    :anderson_unknowns => ([1], "an array of unknown numbers that should be included in the Anderson acceleration",true,false),
    :fixed_penalty => (1e60, "penalty that is used for the values of fixed degrees of freedom (e.g. by Dirichlet boundary data or global constraints)",true,true),
    :linsolver => ("UMFPACK", "String that encodes the linear solver (or type name of self-defined solver, see corressponding example)",true,true),
    :show_solver_config => (false, "show the complete solver configuration before starting to solve",true,true),
    :show_iteration_details => (true, "show details (residuals etc.) of each iteration",true,true),
    :show_statistics => (false, "show some statistics like assembly times",true,true)
)

#
# Print default dict for solver parameters into docstrings
#
function _myprint(dict::Dict{Symbol,Tuple{Any,String,Bool,Bool}},instationary::Bool)
    lines_out=IOBuffer()
    for (k,v) in dict
        if (instationary && v[4]) || (!instationary && v[3])
            println(lines_out,"  - $(k): $(v[2]). Default: $(v[1])\n")
        end
    end
    String(take!(lines_out))
end

#
# Update solver params from dict
#
function _update_solver_params!(user_params,kwargs)
    for (k,v) in kwargs
        user_params[Symbol(k)]=v
    end
    return nothing
end


mutable struct LinearSystemDirectUMFPACK{T} <: AbstractLinearSystem{T}
    x::AbstractVector{T}
    A::ExtendableSparseMatrix{T,Int64}
    b::AbstractVector{T}
    ALU::SuiteSparse.UMFPACK.UmfpackLU{T,Int64}     # LU factorization
    LinearSystemDirectUMFPACK{T}(x,A,b) where {T} = new{T}(x,A,b)
end

function createsolver(ST::Type{<:AbstractLinearSystem{T}},x::AbstractVector{T},A::ExtendableSparseMatrix{T,Int64},b::AbstractVector{T}) where {T}
    return ST(x,A,b)
end

function update!(LS::AbstractLinearSystem{T}) where {T}
    # do nothing for an abstract solver
end

function update!(LS::LinearSystemDirectUMFPACK{T}) where {T}
    if isdefined(LS,:ALU)
        try
            @logmsg MoreInfo "Updating LU decomposition..."
            lu!(LS.ALU,LS.A.cscmatrix; check = false)
        catch
            @logmsg MoreInfo "Computing LU decomposition (pattern changed)..."
            GC()
        end
    else
        @logmsg MoreInfo "Computing LU decomposition..."
        LS.ALU = lu(LS.A.cscmatrix)
    end
end

function solve!(LS::LinearSystemDirectUMFPACK{T}) where {T}
    @logmsg MoreInfo "Solving directly with UMFPACK..."
    ldiv!(LS.x,LS.ALU,LS.b)
end

function set_nonzero_pattern!(A::FEMatrix, AT::Type{<:AbstractAssemblyType} = ON_CELLS)
    @debug "Setting nonzero pattern for FEMatrix..."
    for block = 1 : length(A)
        apply_nonzero_pattern!(A[block],AT)
    end
end

# check if PDE is nonlinear or time-dependent and which blocks require recalculation
# and devise some initial solver strategy
function generate_solver(PDE::PDEDescription, user_params, T::Type{<:Real} = Float64)

    ## declare subiterations
    if user_params[:subiterations] == "auto"
        user_params[:subiterations] = [1:size(PDE.LHSOperators,1)]
    end
    ## declare linear solver
    if user_params[:linsolver] == "UMFPACK"
        user_params[:linsolver] = LinearSystemDirectUMFPACK{T}
    end
    while length(user_params[:skip_update]) < length(user_params[:subiterations])
        push!(user_params[:skip_update], 1)
    end

    ## check if subiterations are nonlinear or time-dependent
    nonlinear::Bool = false
    timedependent::Bool = false
    block_nonlinear::Bool = false
    block_timedependent::Bool = false
    op_nonlinear::Bool = false
    op_timedependent::Bool = false
    LHS_ATs = Array{DataType,2}(undef,size(PDE.LHSOperators,1),size(PDE.LHSOperators,2))
    LHS_dep = Array{Array{Int,1},2}(undef,size(PDE.LHSOperators,1),size(PDE.LHSOperators,2))
    current_dep::Array{Int,1} = []
    all_equations = []
    subiterations = user_params[:subiterations]
    for s = 1 : length(subiterations)
        append!(all_equations, subiterations[s])
    end
    for j = 1 : size(PDE.LHSOperators,1), k = 1 : size(PDE.LHSOperators,2)
        LHS_ATs[j,k] = AssemblyNever
        if !(j in all_equations)
            continue
        end
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
        RHS_ATs[j] = AssemblyNever
        if !(j in all_equations)
            continue
        end
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
    ## correct ATs for blocks not actively solved in the subiterations
    for s = 1 : length(subiterations), k = 1 : size(PDE.LHSOperators,1)
        if (k in subiterations[s]) == false
            for j in subiterations[s]
                if LHS_ATs[j,k] == AssemblyInitial
                    LHS_ATs[j,k] = AssemblyEachTimeStep
                end
            end
        end
    end

    ## declare linear solver
    if user_params[:maxiterations] == "auto"
        user_params[:maxiterations] = nonlinear ? 10 : 1
    end
    if user_params[:check_nonlinear_residual] == "auto"
        user_params[:check_nonlinear_residual] = nonlinear ? true : false
    end

    return SolverConfig{T}(nonlinear, timedependent, LHS_ATs, LHS_dep, RHS_ATs, RHS_dep, LHS_APs, RHS_APs, LHS_AssemblyTimes, RHS_AssemblyTimes, user_params)
end

function Base.show(io::IO, SC::SolverConfig)

    println(io, "\nSOLVER-CONFIGURATION")
    println(io, "======================")
    println(io, "  nonlinear = $(SC.is_nonlinear)")
    println(io, "  timedependent = $(SC.is_timedependent)")

    for (k,v) in SC.user_params
        println(io, "  $(k) = $(v)")
    end

    println(io, "  AssemblyTriggers = ")
    for j = 1 : size(SC.LHS_AssemblyTriggers,1)
        print(io, "         LHS_AT[$j] : ")
        for k = 1 : size(SC.LHS_AssemblyTriggers,2)
            if SC.LHS_AssemblyTriggers[j,k] == AssemblyInitial
                print(io, " I ")
            elseif SC.LHS_AssemblyTriggers[j,k] == AssemblyAlways
                print(io, " A ")
            elseif SC.LHS_AssemblyTriggers[j,k] == AssemblyEachTimeStep
                print(io, " T ")
            elseif SC.LHS_AssemblyTriggers[j,k] == AssemblyNever
                print(io, " N ")
            end
        end
        println(io, "")
    end

    for j = 1 : size(SC.RHS_AssemblyTriggers,1)
        print(io, "         RHS_AT[$j] : ")
        if SC.RHS_AssemblyTriggers[j] == AssemblyInitial
            print(io, " I ")
        elseif SC.RHS_AssemblyTriggers[j] == AssemblyAlways
            print(io, " A ")
        elseif SC.RHS_AssemblyTriggers[j] == AssemblyEachTimeStep
            print(io, " T ")
        elseif SC.RHS_AssemblyTriggers[j] == AssemblyNever
            print(io, " N ")
        end
        println(io, "")
    end
    println(io, "                     (I = Once, T = EachTimeStep, A = Always, N = Never)")
    println(io, "\n  LHS_dependencies = $(SC.LHS_dependencies)\n")
end

function show_statistics(PDE::PDEDescription, SC::SolverConfig)

    subiterations = SC.user_params[:subiterations]

    info_msg = "\n\tACCUMULATED ASSEMBLY TIMES"
    info_msg *= "\n\t=========================="

    for s = 1 : length(subiterations)
        for j = 1 : length(subiterations[s])
            eq = subiterations[s][j]
            for k = 1 : size(SC.LHS_AssemblyTimes,2)
                for o = 1 : size(SC.LHS_AssemblyTimes[eq,k],1)
                    info_msg *= "\n\tLHS[$eq,$k][$o] ($(PDE.LHSOperators[eq,k][o].name)) = $(SC.LHS_AssemblyTimes[eq,k][o])s"
                end
            end
        end
    end

    for s = 1 : length(subiterations)
        for j = 1 : length(subiterations[s])
            eq = subiterations[s][j]
            for o = 1 : size(SC.RHS_AssemblyTimes[eq],1)
                info_msg *= "\n\tRHS[$eq][$o] ($(PDE.RHSOperators[eq][o].name)) = $(SC.RHS_AssemblyTimes[eq][o])s"
            end
        end
    end
    @info info_msg
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
    only_rhs::Bool = false) where {T <: Real}

    if length(equations) == 0
        equations = 1:size(PDE.LHSOperators,1)
    end
    if length(if_depends_on) == 0
        if_depends_on = 1:size(PDE.LHSOperators,1)
    end
    if storage_trigger == "same as min_trigger"
        storage_trigger = min_trigger
    end

    @logmsg DeepInfo "Entering assembly of equations=$equations (time = $time, min_trigger = $min_trigger)"

    # important to flush first in case there is some cached stuff that
    # will not be seen by fill! functions
    flush!(A.entries)

    elapsedtime::Float64 = 0
    # force (re)assembly of stored bilinearforms and RhsOperators
    @logmsg DeepInfo "Locking for triggered storage updates..."
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
                        elapsedtime = @elapsed update_storage!(PDEOperator, CurrentSolution, equations[j], k; time = time)
                        SC.LHS_AssemblyTimes[equations[j],k][o] += elapsedtime
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
                    elapsedtime = @elapsed update_storage!(PDEOperator, CurrentSolution, equations[j] ; time = time)
                    SC.RHS_AssemblyTimes[equations[j]][o] += elapsedtime
                end
            end
        end
    end

    # (re)assembly right-hand side
    rhs_block_has_been_erased = zeros(Bool,length(equations))
    for j = 1 : length(equations)
        if (min_trigger <: SC.RHS_AssemblyTriggers[equations[j]]) && (length(PDE.RHSOperators[equations[j]]) > 0)
            @debug "Erasing rhs block [$j]"
            fill!(b[j],0.0)
            rhs_block_has_been_erased[j] = true
            @logmsg DeepInfo "Locking what to assemble in rhs block [$(equations[j])]..."
            for o = 1 : length(PDE.RHSOperators[equations[j]])
                PDEOperator = PDE.RHSOperators[equations[j]][o]
                elapsedtime = @elapsed assemble!(b[j], SC, equations[j], o, PDE.RHSOperators[equations[j]][o], CurrentSolution; time = time)
                SC.RHS_AssemblyTimes[equations[j]][o] += elapsedtime
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
                    if (min_trigger <: SC.LHS_AssemblyTriggers[equations[j],k]) && (length(PDE.LHSOperators[equations[j],k]) > 0)
                        @debug "Erasing lhs block [$j,$subblock]"
                        fill!(A[j,subblock],0)
                        lhs_block_has_been_erased[j, subblock] = true
                        @logmsg DeepInfo "Locking what to assemble in lhs block [$(equations[j]),$k]..."
                        for o = 1 : length(PDE.LHSOperators[equations[j],k])
                            PDEOperator = PDE.LHSOperators[equations[j],k][o]
                            if typeof(PDEOperator) <: LagrangeMultiplier
                                elapsedtime = @elapsed assemble!(A[j,subblock], SC, equations[j],k,o, PDEOperator, CurrentSolution; time = time, At = A[subblock,j])
                            else
                                elapsedtime = @elapsed assemble!(A[j,subblock], SC, equations[j],k,o, PDEOperator, CurrentSolution; time = time)
                            end  
                            SC.LHS_AssemblyTimes[equations[j],k][o] += elapsedtime
                        end  
                    end
                elseif !(k in equations)
                    if (min_trigger <: SC.LHS_AssemblyTriggers[equations[j],k]) == true
                        if (length(PDE.LHSOperators[equations[j],k]) > 0) && (!(min_trigger <: SC.RHS_AssemblyTriggers[equations[j]]))
                            if rhs_block_has_been_erased[j] == false
                                @debug "Erasing rhs block [$j]"
                                fill!(b[j],0)
                                rhs_block_has_been_erased[j] = true
                            end
                        end
                        for o = 1 : length(PDE.LHSOperators[equations[j],k])
                            PDEOperator = PDE.LHSOperators[equations[j],k][o]
                            @debug "Assembling lhs block[$j,$k] into rhs block[$j] ($k not in equations): $(PDEOperator.name)"
                            elapsedtime = @elapsed assemble!(b[j], SC, equations[j],k,o, PDEOperator, CurrentSolution; factor = -1.0, time = time, fixed_component = k)
                            SC.LHS_AssemblyTimes[equations[j],k][o] += elapsedtime
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
function solve_direct!(Target::FEVector{T}, PDE::PDEDescription, SC::SolverConfig{T}; time::Real = 0, show_details::Bool = false) where {T <: Real}

    FEs = Array{FESpace,1}([])
    for j=1 : length(Target.FEVectorBlocks)
        push!(FEs,Target.FEVectorBlocks[j].FES)
    end    

    # ASSEMBLE SYSTEM
    A = FEMatrix{T}("SystemMatrix", FEs)
    b = FEVector{T}("SystemRhs", FEs)
    assemble!(A,b,PDE,SC,Target; equations = Array{Int,1}(1:length(FEs)), min_trigger = AssemblyInitial, time = time)

    # ASSEMBLE BOUNDARY DATA
    fixed_dofs = []
    for j= 1 : length(Target.FEVectorBlocks)
        new_fixed_dofs = boundarydata!(Target[j],PDE.BoundaryOperators[j]; time = time)
        new_fixed_dofs .+= Target[j].offset
        append!(fixed_dofs, new_fixed_dofs)
    end    

    # PREPARE GLOBALCONSTRAINTS
    flush!(A.entries)
    for j = 1 : length(PDE.GlobalConstraints)
        additional_fixed_dofs = apply_constraint!(A,b,PDE.GlobalConstraints[j],Target)
        append!(fixed_dofs,additional_fixed_dofs)
    end

    # PENALIZE FIXED DOFS
    # (from boundary conditions and constraints)
    fixed_dofs = Base.unique(fixed_dofs)
    fixed_penalty = SC.user_params[:fixed_penalty]
    for j = 1 : length(fixed_dofs)
        b.entries[fixed_dofs[j]] = fixed_penalty * Target.entries[fixed_dofs[j]]
        A[1][fixed_dofs[j],fixed_dofs[j]] = fixed_penalty
    end

    # SOLVE
    LS = createsolver(SC.user_params[:linsolver], Target.entries, A.entries, b.entries)
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

    # REALIZE GLOBAL GLOBALCONSTRAINTS 
    # (possibly changes some entries of Target)
    for j = 1 : length(PDE.GlobalConstraints)
        realize_constraint!(Target,PDE.GlobalConstraints[j])
    end

    if SC.user_params[:show_statistics]
        show_statistics(PDE,SC)
    end
    return sqrt(resnorm)
end


# solve full system iteratively until fixpoint is reached
function solve_fixpoint_full!(Target::FEVector{T}, PDE::PDEDescription, SC::SolverConfig{T}; time::Real = 0) where {T <: Real}

    ## get relevant solver parameters
    anderson_iterations = SC.user_params[:anderson_iterations]
    anderson_metric = SC.user_params[:anderson_metric]
    anderson_unknowns = SC.user_params[:anderson_unknowns]
    fixed_penalty = SC.user_params[:fixed_penalty]
    damping = SC.user_params[:damping]
    maxiterations = SC.user_params[:maxiterations]
    target_residual = SC.user_params[:target_residual]
    skip_update = SC.user_params[:skip_update]

    # ASSEMBLE SYSTEM INIT
    FEs = Array{FESpace,1}([])
    for j = 1 : length(Target.FEVectorBlocks)
        push!(FEs,Target.FEVectorBlocks[j].FES)
    end    
    A = FEMatrix{T}("SystemMatrix", FEs)
    b = FEVector{T}("SystemRhs", FEs)
    set_nonzero_pattern!(A)
    assembly_time = @elapsed assemble!(A,b,PDE,SC,Target; time = time, equations = Array{Int,1}(1:length(FEs)), min_trigger = AssemblyInitial)

    # ASSEMBLE BOUNDARY DATA
    fixed_dofs = []
    assembly_time += @elapsed for j= 1 : length(Target.FEVectorBlocks)
        new_fixed_dofs = boundarydata!(Target[j],PDE.BoundaryOperators[j]; time = time) .+ Target[j].offset
        append!(fixed_dofs, new_fixed_dofs)
    end    

    # ANDERSON ITERATIONS
    anderson_time = 0
    if anderson_iterations > 0
        anderson_time += @elapsed begin    
            @logmsg MoreInfo "Preparing Anderson acceleration for unknown(s) = $anderson_unknowns with convergence metric $anderson_metric and $anderson_iterations iterations"
            # we need to save the last iterates
            LastAndersonIterates = Array{FEVector,1}(undef, anderson_iterations+1) # actual iterates u_k
            LastAndersonIteratesTilde = Array{FEVector,1}(undef, anderson_iterations+1) # auxiliary iterates \tilde u_k
            PDE.LHSOperators[1,1][1].store_operator = true # use the first operator to compute norm in which Anderson iterations are optimised

            ## assemble convergence metric
            AIONormOperator = FEMatrix{T}("AA-Metric",FEs)
            for u in anderson_unknowns
                if anderson_metric == "L2"
                    AA_METRIC = SymmetricBilinearForm(Float64, ON_CELLS, [FEs[u], FEs[u]], [Identity, Identity], MultiplyScalarAction(1,get_ncomponents(typeof(FEs[u]).parameters[1])))
                    assemble!(AIONormOperator[u], AA_METRIC)
                elseif anderson_metric == "l2"
                    for j = 1 : FEs[u].ndofs
                        AIONormOperator[u][j,j] = 1
                    end
                end
            end
            AIONormOperator = AIONormOperator.entries
            flush!(AIONormOperator)
            
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
    end
    if damping > 0
        LastIterate = deepcopy(Target)
    end

    residual = zeros(T,length(b.entries))
    linresnorm::T = 0.0
    resnorm::T = 0.0

    ## INIT SOLVER
    LS = createsolver(SC.user_params[:linsolver],Target.entries,A.entries,b.entries)

    if SC.user_params[:show_statistics]
        @info "initial assembly time = $(assembly_time)s"
    end
    if SC.user_params[:show_iteration_details]
        if SC.user_params[:show_statistics]
            @printf("\n\tITERATION |  LSRESIDUAL  |  NLRESIDUAL  | TIME ASSEMBLY/SOLVE/TOTAL (s)")
            @printf("\n\t-----------------------------------------------------------------------\n")
        else
            @printf("\n\tITERATION |  LSRESIDUAL  |  NLRESIDUAL")
            @printf("\n\t--------------------------------------\n")
        end
    end
    for j = 1 : maxiterations
        time_total = @elapsed begin

        # PREPARE GLOBALCONSTRAINTS
        flush!(A.entries)
        for j = 1 : length(PDE.GlobalConstraints)
            additional_fixed_dofs = apply_constraint!(A,b,PDE.GlobalConstraints[j],Target)
            append!(fixed_dofs,additional_fixed_dofs)
        end

        # PENALIZE FIXED DOFS
        # (from boundary conditions and constraints)
        fixed_dofs = Base.unique(fixed_dofs)
        for j = 1 : length(fixed_dofs)
            b.entries[fixed_dofs[j]] = fixed_penalty * Target.entries[fixed_dofs[j]]
            A[1][fixed_dofs[j],fixed_dofs[j]] = fixed_penalty
        end

        # SOLVE
        time_solver = @elapsed begin
            flush!(A.entries)
            if j == 1 || (j % skip_update[1] == 0 && skip_update[1] != -1)
                update!(LS)
            end
            solve!(LS)
        end

        # CHECK LINEAR RESIDUAL
        if SC.user_params[:show_iteration_details]
            residual = A.entries*Target.entries - b.entries
            residual[fixed_dofs] .= 0
            linresnorm = (sqrt(sum(residual.^2, dims = 1)[1]))
        end

        # POSTPRCOESS : ANDERSON ITERATE
        if anderson_iterations > 0
            anderson_time += @elapsed begin
                depth = min(anderson_iterations,j-1)

                # move last tilde iterates to the front in memory
                for j = 1 : anderson_iterations
                    LastAndersonIteratesTilde[j].entries .= LastAndersonIteratesTilde[j+1].entries
                end
                # save fixpoint iterate as new tilde iterate
                LastAndersonIteratesTilde[anderson_iterations+1].entries .= Target.entries

                if depth > 0
                    # fill matrix
                    for j = 1 : depth+1, k = 1 : depth+1
                        AIOMatrix[j,k] = lrmatmul(LastAndersonIteratesTilde[anderson_iterations+2-j].entries .- LastAndersonIterates[anderson_iterations+2-j].entries,
                                                AIONormOperator,
                                                LastAndersonIteratesTilde[anderson_iterations+2-k].entries .- LastAndersonIterates[anderson_iterations+2-k].entries)
                    end
                    # solve for alpha coefficients
                    ind = union(1:depth+1,anderson_iterations+2)
                    AIOalpha[ind] = AIOMatrix[ind,ind]\AIORhs[ind]

                    # move last iterates to the front in memory
                    for j = 1 : anderson_iterations
                        LastAndersonIterates[j].entries .= LastAndersonIterates[j+1].entries
                    end

                    # compute next iterates
                    fill!(Target[1],0.0)
                    for a = 1 : depth+1
                        for u in anderson_unknowns
                            for j = 1 : FEs[u].ndofs
                                Target[u][j] += AIOalpha[a] * LastAndersonIteratesTilde[anderson_iterations+2-a][u][j]
                            end
                        end
                    end
                    LastAndersonIterates[anderson_iterations+1].entries .= Target.entries
                else
                    LastAndersonIterates[anderson_iterations+1].entries .= Target.entries
                end
            end
        end

        # POSTPROCESS : DAMPING
        if damping > 0
            for j = 1 : length(Target.entries)
                Target.entries[j] = damping*LastIterate.entries[j] + (1-SC.damping)*Target.entries[j]
            end
            LastIterate.entries .= Target.entries
        end


        # REASSEMBLE NONLINEAR PARTS
        time_reassembly = @elapsed for j = 1:size(PDE.RHSOperators,1)
            assemble!(A,b,PDE,SC,Target; time = time, equations = [j], min_trigger = AssemblyAlways)
        end

        # CHECK NONLINEAR RESIDUAL
        residual = A.entries*Target.entries - b.entries
        residual[fixed_dofs] .= 0
        resnorm = (sqrt(sum(residual.^2, dims = 1)[1]))

        end #elapsed

        if SC.user_params[:show_iteration_details]
            @printf("\t   %4d  ", j)
            @printf(" | %e", linresnorm)
            @printf(" | %e", resnorm)
            if SC.user_params[:show_statistics]
                @printf(" | %.2e/%.2e/%.2e",time_reassembly,time_solver, time_total)
            end
            @printf("\n")
        end

        if resnorm < target_residual
            break
        end

        if j == maxiterations
            @warn "maxiterations reached!"
            break
        end
    end

    if SC.user_params[:show_iteration_details]
        @printf("\n")
    end

    # REALIZE GLOBAL GLOBALCONSTRAINTS 
    # (possibly changes some entries of Target)
    for j = 1 : length(PDE.GlobalConstraints)
        realize_constraint!(Target,PDE.GlobalConstraints[j])
    end

    if SC.user_params[:show_statistics]
        @info "anderson acceleration took $(anderson_time)s"
        show_statistics(PDE,SC)
    end

    return resnorm
end


# solve system iteratively until fixpoint is reached
# by consecutively solving subiterations
function solve_fixpoint_subiterations!(Target::FEVector{T}, PDE::PDEDescription, SC::SolverConfig{T}; time = 0) where {T <: Real}

    ## get relevant solver parameters
    subiterations = SC.user_params[:subiterations]
    anderson_iterations = SC.user_params[:anderson_iterations]
    fixed_penalty = SC.user_params[:fixed_penalty]
    damping = SC.user_params[:damping]
    maxiterations = SC.user_params[:maxiterations]
    target_residual = SC.user_params[:target_residual]
    skip_update = SC.user_params[:skip_update]

    if anderson_iterations > 0
        @warn "Sorry, Anderson acceleration not yet available for this solver mode."
    end

    # ASSEMBLE SYSTEM INIT
    assembly_time = @elapsed begin
        FEs = Array{FESpace,1}([])
        for j=1 : length(Target.FEVectorBlocks)
            push!(FEs,Target.FEVectorBlocks[j].FES)
        end    
        nsubiterations = length(subiterations)
        eqoffsets = Array{Array{Int,1},1}(undef,nsubiterations)
        A = Array{FEMatrix{T},1}(undef,nsubiterations)
        b = Array{FEVector{T},1}(undef,nsubiterations)
        x = Array{FEVector{T},1}(undef,nsubiterations)
        for i = 1 : nsubiterations
            A[i] = FEMatrix{T}("SystemMatrix subiteration $i", FEs[subiterations[i]])
            b[i] = FEVector{T}("SystemRhs subiteration $i", FEs[subiterations[i]])
            x[i] = FEVector{T}("SystemRhs subiteration $i", FEs[subiterations[i]])
            set_nonzero_pattern!(A[i])
            assemble!(A[i],b[i],PDE,SC,Target; time = time, equations = subiterations[i], min_trigger = AssemblyInitial)
            eqoffsets[i] = zeros(Int,length(subiterations[i]))
            for j= 1 : length(Target.FEVectorBlocks), eq = 1 : length(subiterations[i])
                if j < subiterations[i][eq]
                    eqoffsets[i][eq] += FEs[j].ndofs
                end
            end
        end

        # ASSEMBLE BOUNDARY DATA
        fixed_dofs = []
        eqdof = 0
        for j = 1 : length(Target.FEVectorBlocks)
            new_fixed_dofs = boundarydata!(Target[j],PDE.BoundaryOperators[j]; time = time) .+ Target[j].offset
            append!(fixed_dofs, new_fixed_dofs)
        end    
    end
    
    residual = Array{FEVector{T},1}(undef,nsubiterations)
    for s = 1 : nsubiterations
        residual[s] = FEVector{T}("residual subiteration $s", FEs[subiterations[s]])
    end
    linresnorm = zeros(T,nsubiterations)
    resnorm = zeros(T,nsubiterations)


    ## INIT SOLVERS
    LS = Array{AbstractLinearSystem,1}(undef,nsubiterations)
    for s = 1 : nsubiterations
        LS[s] = createsolver(SC.user_params[:linsolver],x[s].entries,A[s].entries,b[s].entries)
    end
    if damping > 0
        LastIterate = deepcopy(Target)
    end


    if SC.user_params[:show_statistics]
        @info "initial assembly time = $(assembly_time)s"
    end
    if SC.user_params[:show_iteration_details]
        if SC.user_params[:show_statistics]
            @printf("\n\tITERATION |  LSRESIDUAL  |  NLRESIDUAL  | TIME ASSEMBLY/SOLVE/TOTAL (s)")
            @printf("\n\t-----------------------------------------------------------------------\n")
        else
            @printf("\n\tITERATION |  LSRESIDUAL  |  NLRESIDUAL")
            @printf("\n\t--------------------------------------\n")
        end
    end
    for iteration = 1 : maxiterations

        time_reassembly = 0
        time_solver = 0
        time_total = @elapsed for s = 1 : nsubiterations

            # PREPARE GLOBALCONSTRAINTS
            # known bug: this will only work if no components in front of the constrained component(s)
            # are missing in the subiteration
            for j = 1 : length(PDE.GlobalConstraints)
                if PDE.GlobalConstraints[j].component in subiterations[s]
                   additional_fixed_dofs = apply_constraint!(A[s],b[s],PDE.GlobalConstraints[j],Target; current_equations = subiterations[s])
                   append!(fixed_dofs, additional_fixed_dofs)
                end
            end

            # PENALIZE FIXED DOFS
            # (from boundary conditions and constraints)
            for j = 1 : length(fixed_dofs)
                for eq = 1 : length(subiterations[s])
                    # check if fixed_dof is necessary for subiteration
                    if fixed_dofs[j] > eqoffsets[s][eq] && fixed_dofs[j] <= eqoffsets[s][eq]+FEs[subiterations[s][eq]].ndofs
                        eqdof = fixed_dofs[j] - eqoffsets[s][eq]
                        #println("fixing dof $eqdof (global $(fixed_dofs[j])) of unknown $eq with value $(Target.entries[fixed_dofs[j]])")
                        b[s][eq][eqdof] = fixed_penalty * Target.entries[fixed_dofs[j]]
                        A[s][eq,eq][eqdof,eqdof] = fixed_penalty
                    end
                end
            end

            # SOLVE
            time_solver = @elapsed begin
                flush!(A[s].entries)
                if iteration == 1 || (iteration % skip_update[s] == 0 && skip_update[s] != -1)
                    update!(LS[s])
                end
                solve!(LS[s])
            end

            # CHECK LINEAR RESIDUAL
            if SC.user_params[:show_iteration_details]
                residual[s].entries[:] = A[s].entries*x[s].entries - b[s].entries
                for j = 1 : length(fixed_dofs)
                    for eq = 1 : length(subiterations[s])
                        # check if fixed_dof is necessary for subiteration
                        if fixed_dofs[j] > eqoffsets[s][eq] && fixed_dofs[j] <= eqoffsets[s][eq]+FEs[subiterations[s][eq]].ndofs
                            eqdof = fixed_dofs[j] - eqoffsets[s][eq]
                            residual[s][eq][eqdof] = 0
                        end
                    end
                end
                linresnorm[s] = (sqrt(sum(residual[s].entries.^2, dims = 1)[1]))
            end

            # WRITE INTO Target
            if damping > 0
                for j = 1 : length(subiterations[s])
                    for k = 1 : length(Target[subiterations[s][j]])
                        Target[subiterations[s][j]][k] = (1-SC.damping)*x[s][j][k] + damping*LastIterate[subiterations[s][j]][k] 
                    end
                end
                LastIterate.entries .= Target.entries
            else
                for j = 1 : length(subiterations[s])
                    for k = 1 : length(Target[subiterations[s][j]])
                        Target[subiterations[s][j]][k] = x[s][j][k]
                    end
                end
            end

            # REASSEMBLE PARTS FOR NEXT SUBITERATION
            time_reassembly += @elapsed begin
                next_eq = (s == nsubiterations) ? 1 : s+1
                assemble!(A[next_eq],b[next_eq],PDE,SC,Target; time = time, equations = subiterations[next_eq], min_trigger = AssemblyEachTimeStep)
            end

        end

        # CHECK NONLINEAR RESIDUAL
        for s = 1 : nsubiterations
            residual[s].entries[:] = A[s].entries*x[s].entries - b[s].entries
            for j = 1 : length(fixed_dofs)
                for eq = 1 : length(subiterations[s])
                    if fixed_dofs[j] > eqoffsets[s][eq] && fixed_dofs[j] <= eqoffsets[s][eq]+FEs[subiterations[s][eq]].ndofs
                        eqdof = fixed_dofs[j] - eqoffsets[s][eq]
                        residual[s][eq][eqdof] = 0
                    end
                end
            end
            resnorm[s] = (sqrt(sum(residual[s].entries.^2, dims = 1)[1]))
        end

        if SC.user_params[:show_iteration_details]
            @printf("\t   %4d  ", iteration)
            @printf(" | %e", sqrt(sum(linresnorm.^2)))
            @printf(" | %e", sqrt(sum(resnorm.^2)))
            if SC.user_params[:show_statistics]
                @printf(" | %.2e/%.2e/%.2e",time_reassembly,time_solver, time_total)
            end
            @printf("\n")
        end

        if sqrt(sum(resnorm.^2)) < target_residual
            break
        end

        if iteration == maxiterations
            @warn "maxiterations reached!"
            break
        end
    end

    if SC.user_params[:show_iteration_details]
        @printf("\n")
    end

    # REALIZE GLOBALCONSTRAINTS 
    # (possibly changes some entries of Target)
    for j = 1 : length(PDE.GlobalConstraints)
        realize_constraint!(Target,PDE.GlobalConstraints[j])
    end

    if SC.user_params[:show_statistics]
        show_statistics(PDE,SC)
    end

    return sqrt(sum(resnorm.^2))
end

"""
````
function solve!(
    Target::FEVector,
    PDE::PDEDescription;
    kwargs)
````

Solves a given PDE (provided as a PDEDescription) and writes the solution into the FEVector Target (which knows the discrete ansatz spaces).

Keyword arguments:
$(_myprint(default_solver_kwargs(),false))

Depending on the subiterations and detected/configured nonlinearities the whole system is either solved directly in one step or via a fixed-point iteration.

"""
function solve!(
    Target::FEVector{T},
    PDE::PDEDescription;
    kwargs...) where {T <: Real}

    ## generate solver configurations
    user_params=Dict{Symbol,Any}( k => v[1] for (k,v) in default_solver_kwargs())
    _update_solver_params!(user_params,kwargs)
    SC = generate_solver(PDE, user_params, T)

    ## logging stuff
    if SC.is_timedependent
        moreinfo_string = "----- Solving $(PDE.name) (at fixed time $time) -----"
    else
        moreinfo_string = "----- Solving $(PDE.name) -----"
    end
    nsolvedofs = 0
    subiterations = user_params[:subiterations]
    for s = 1 : length(subiterations), o = 1 : length(subiterations[s])
        d = subiterations[s][o]
        moreinfo_string *= "\n\tEquation ($s.$d) $(PDE.equation_names[d]) : $(PDE.unknown_names[d]) >> $(Target[d].name) ($(Target[d].FES.name), ndofs = $(Target[d].FES.ndofs))"
        nsolvedofs += Target[d].FES.ndofs
    end
    @info moreinfo_string
    if user_params[:show_solver_config]
        @show SC
    else
        @debug SC
    end

    ## choose solver strategy
    totaltime = @elapsed begin
        if user_params[:subiterations] == [1:size(PDE.LHSOperators,1)] # full problem is solved
            if SC.is_nonlinear == false
                residual = solve_direct!(Target, PDE, SC; time = user_params[:time])
            else
                residual = solve_fixpoint_full!(Target, PDE, SC; time = user_params[:time])
            end
        else # only part(s) of the problem is solved
            residual = solve_fixpoint_subiterations!(Target, PDE, SC; time = user_params[:time])
        end
    end

    if user_params[:show_statistics]
        @info "totaltime = $(totaltime)s"
    end
    ## report and check final residual
    if !user_params[:show_iteration_details]
        @info "reached residual = $residual"
    end
    if residual > user_params[:target_residual]
        @warn "residual was larger than desired target_residual = $(user_params[:target_residual])!"
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
    LastIterate::FEVector    # helper variable if maxiterations > 1
    fixed_dofs::Array{Int,1}            # fixed dof numbes (values are stored in X)
    eqoffsets::Array{Array{Int,1},1}    # offsets for subblocks of each equation package
    ALU::Array{Any,1}  # LU decompositions of matrices A
end



function assemble_massmatrix4subiteration!(TCS::TimeControlSolver, i::Int; force::Bool = false)
    subiterations = TCS.SC.user_params[:subiterations]
    for j = 1 : length(subiterations[i])
        if (subiterations[i][j] in TCS.which) == true 

            # assembly mass matrix into AM block
            pos = findall(x->x==subiterations[i][j], TCS.which)[1]

            if TCS.nonlinear_dt[pos] == true || force == true
                @logmsg DeepInfo "Assembling mass matrix for block [$j,$j] of subiteration $i with action of type $(typeof(TCS.dt_actions[pos]))"
                A = TCS.AM[i][j,j]
                FE1 = A.FESX
                FE2 = A.FESY
                operator1 = TCS.dt_operators[pos]
                operator2 = TCS.dt_operators[pos]
                BLF = SymmetricBilinearForm(Float64, ON_CELLS, [FE1, FE2], [operator1, operator2], TCS.dt_actions[pos])    
                time = @elapsed assemble!(A, BLF, skip_preps = false)
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
    dt_testfunction_operator = [],
    dt_action = [],
    nonlinear_dt = [],
    kwargs...)
````

Creates a time-dependent solver that can be advanced in time with advance!.
The FEVector Solution stores the initial state but also the solution at the current time.
The argument TIR carries the time integration rule (currently there is only BackwardEuler).

Keyword arguments:
$(_myprint(default_solver_kwargs(),true))

Further (very experimental) optional arguments for TimeControlSolver are:
- dt_test_function_operator : (array of) operators applied to testfunctions in time derivative (default: Identity)
- dt_action : (array of) actions that are applied to the ansatz function in the time derivative (to include parameters etc.)
- nonlinear_dt : (array of) booleans to decide which time derivatives should be recomputed in each iteration/timestep

"""
function TimeControlSolver(
    PDE::PDEDescription,
    InitialValues::FEVector,    # contains initial values and stores solution of advance! methods
    TIR::Type{<:AbstractTimeIntegrationRule} = BackwardEuler;
    dt_testfunction_operator = [],
    dt_action = [],
    nonlinear_dt = [],
    kwargs...)

    ## generate solver configurations
    user_params=Dict{Symbol,Any}( k => v[1] for (k,v) in default_solver_kwargs())
    _update_solver_params!(user_params,kwargs)
    SC = generate_solver(PDE, user_params, Float64)
    start_time = user_params[:time]

    # generate solver for time-independent problem
    for j = 1 : length(InitialValues.FEVectorBlocks)
        if (SC.RHS_AssemblyTriggers[j] <: AssemblyEachTimeStep)
            SC.RHS_AssemblyTriggers[j] = AssemblyEachTimeStep
        end
    end

    ## logging stuff
    moreinfo_string = "----- Preparing time control solver for $(PDE.name) -----"
    nsolvedofs = 0
    subiterations = SC.user_params[:subiterations]
    nsubiterations = length(subiterations)
    timedependent_equations = SC.user_params[:timedependent_equations]
    for s = 1 : nsubiterations, o = 1 : length(subiterations[s])
        d = subiterations[s][o]
        moreinfo_string *= "\n\tEquation ($s.$d) $(PDE.equation_names[d]) : $(PDE.unknown_names[d]) >> $(InitialValues[d].name) ($(InitialValues[d].FES.name), ndofs = $(InitialValues[d].FES.ndofs)), timedependent = $(d in timedependent_equations ? "yes" : "no")"
        nsolvedofs += InitialValues[d].FES.ndofs
    end
    @info moreinfo_string
    if user_params[:show_solver_config]
        @show SC
    else
        @debug SC
    end

    # allocate matrices etc
    FEs = Array{FESpace,1}([])
    for j = 1 : length(InitialValues.FEVectorBlocks)
        push!(FEs,InitialValues.FEVectorBlocks[j].FES)
    end    
    eqoffsets = Array{Array{Int,1},1}(undef,nsubiterations)
    AM = Array{FEMatrix{Float64},1}(undef,nsubiterations)
    A = Array{FEMatrix{Float64},1}(undef,nsubiterations)
    b = Array{FEVector{Float64},1}(undef,nsubiterations)
    x = Array{FEVector{Float64},1}(undef,nsubiterations)
    res = Array{FEVector{Float64},1}(undef,nsubiterations)
    for i = 1 : nsubiterations
        A[i] = FEMatrix{Float64}("SystemMatrix subiteration $i", FEs[subiterations[i]])
        AM[i] = FEMatrix{Float64}("MassMatrix subiteration $i", FEs[subiterations[i]])
        b[i] = FEVector{Float64}("SystemRhs subiteration $i", FEs[subiterations[i]])
        x[i] = FEVector{Float64}("Solution subiteration $i", FEs[subiterations[i]])
        res[i] = FEVector{Float64}("Residual subiteration $i", FEs[subiterations[i]])
        set_nonzero_pattern!(A[i])
        assemble!(A[i],b[i],PDE,SC,InitialValues; time = start_time, equations = subiterations[i], min_trigger = AssemblyInitial)
        eqoffsets[i] = zeros(Int,length(subiterations[i]))
        for j= 1 : length(FEs), eq = 1 : length(subiterations[i])
            if j < subiterations[i][eq]
                eqoffsets[i][eq] += FEs[j].ndofs
            end
        end
    end


    # ASSEMBLE BOUNDARY DATA
    # will be overwritten in solve if time-dependent
    # but fixed_dofs will remain throughout
    fixed_dofs = []
    for j = 1 : length(InitialValues.FEVectorBlocks)
        new_fixed_dofs = boundarydata!(InitialValues[j],PDE.BoundaryOperators[j]) .+ InitialValues[j].offset
        append!(fixed_dofs, new_fixed_dofs)
    end    

    for s = 1 : nsubiterations

        # PREPARE GLOBALCONSTRAINTS
        # known bug: this will only work if no components in front of the constrained component(s)
        # are missing in the subiteration
        for j = 1 : length(PDE.GlobalConstraints)
            if PDE.GlobalConstraints[j].component in subiterations[s]
                additional_fixed_dofs = apply_constraint!(A[s],b[s],PDE.GlobalConstraints[j],InitialValues; current_equations = subiterations[s])
                append!(fixed_dofs, additional_fixed_dofs)
            end
        end

        # COPY INITIAL VALUES TO SUB-PROBLEM SOLUTIONS
        for j = 1 : length(x[s].entries), k = 1 : length(subiterations[s])
            d = subiterations[s][k]
            x[s][k][:] = InitialValues[d][:]
        end

        # prepare and configure mass matrices
        for j = 1 : length(subiterations[s])
            d = subiterations[s][j] # equation of subiteration
            if (d in timedependent_equations) == true 
                pos = findall(x->x==d, timedependent_equations)[1] # position in timedependent_equations
                if length(nonlinear_dt) < pos
                    push!(nonlinear_dt, false)
                end
                if length(dt_action) < pos
                    push!(dt_action, NoAction())
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
        LS[s] = createsolver(SC.user_params[:linsolver],x[s].entries,A[s].entries,b[s].entries)
    end

    # if nonlinear iterations are performed we need to remember the iterate from last timestep
    if user_params[:maxiterations] > 1
        # two vectors to store intermediate approximations
        LastIterate = deepcopy(InitialValues)
    else
        # same vector, only one is needed
        LastIterate = InitialValues
    end

    # generate TimeControlSolver
    TCS = TimeControlSolver{TIR}(PDE,SC,LS,start_time,0,0,AM,timedependent_equations,nonlinear_dt,dt_testfunction_operator,dt_action,A,b,x,res,InitialValues, LastIterate, fixed_dofs, eqoffsets, Array{SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64},1}(undef,length(subiterations)))

    # trigger initial assembly of all time derivative mass matrices
    for i = 1 : nsubiterations
        assemble_massmatrix4subiteration!(TCS, i; force = true)
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
    eqdof = 0
    eqoffsets = TCS.eqoffsets
    T = eltype(res[1].entries)
    statistics = zeros(Float64,length(X),4)
    linresnorm = 1e30
    resnorm = 1e30

    ## get relevant solver parameters
    subiterations = SC.user_params[:subiterations]
    anderson_iterations = SC.user_params[:anderson_iterations]
    fixed_penalty = SC.user_params[:fixed_penalty]
    damping = SC.user_params[:damping]
    maxiterations = SC.user_params[:maxiterations]
    check_nonlinear_residual = SC.user_params[:check_nonlinear_residual]
    target_residual = SC.user_params[:target_residual]
    skip_update = SC.user_params[:skip_update]

    # save current solution if nonlinear iterations are needed
    # X will then contain the current nonlinear iterate
    # LastIterate contains the solution from last time step (for reassembly of time derivatives)
    if maxiterations > 1
        for j = 1 : length(X.entries)
            LastIterate.entries[j] = X.entries[j]
        end
    end


    ## LOOP OVER ALL SUBITERATIONS
    for s = 1 : length(subiterations)


        # UPDATE SYSTEM
        if skip_update[s] != -1 || TCS.cstep == 1 # update matrix
            fill!(A[s].entries.cscmatrix.nzval,0)
            fill!(b[s].entries,0)
            assemble!(A[s],b[s],PDE,SC,LastIterate; equations = subiterations[s], time = TCS.ctime, min_trigger = AssemblyInitial, storage_trigger = AssemblyEachTimeStep)

            ## update mass matrix and add time derivative
            assemble_massmatrix4subiteration!(TCS, s; force = false)
            for k = 1 : length(subiterations[s])
                d = subiterations[s][k]
                addblock!(A[s][k,k],AM[s][k,k]; factor = 1.0/timestep)
                addblock_matmul!(b[s][k],AM[s][k,k],X[d]; factor = 1.0/timestep)
            end
            flush!(A[s].entries)
        else # only update rhs
            fill!(b[s].entries,0)
            assemble!(A[s],b[s],PDE,SC,LastIterate; equations = subiterations[s], time = TCS.ctime, min_trigger = AssemblyInitial, storage_trigger = AssemblyEachTimeStep, only_rhs = true)

            ## add time derivative
            for k = 1 : length(subiterations[s])
                d = subiterations[s][k]
                addblock_matmul!(b[s][k],AM[s][k,k],X[d]; factor = 1.0/timestep)
            end
        end

        # ASSEMBLE (TIME-DEPENDENT) BOUNDARY DATA
        for k = 1 : length(subiterations[s])
            d = subiterations[s][k]
            if any(PDE.BoundaryOperators[d].timedependent) == true
                boundarydata!(x[s][k],PDE.BoundaryOperators[d]; time = TCS.ctime)
            end
        end    

        ## START (NONLINEAR) ITERATION(S)
        for iteration = 1 : maxiterations
            statistics[s,4] = iteration.^2 # will be square-rooted later

            # PENALIZE FIXED DOFS (IN CASE THE MATRIX CHANGED)
            # (from boundary conditions and global constraints)
            for j = 1 : length(fixed_dofs)
                for eq = 1 : length(subiterations[s])
                    if fixed_dofs[j] > eqoffsets[s][eq] && fixed_dofs[j] <= eqoffsets[s][eq]+X[subiterations[s][eq]].FES.ndofs
                        eqdof = fixed_dofs[j] - eqoffsets[s][eq]
                        b[s][eq][eqdof] = fixed_penalty * x[s][eq][eqdof]
                        A[s][eq,eq][eqdof,eqdof] = fixed_penalty
                    end
                end
            end

            ## SOLVE for x[s]
            time_solver = @elapsed begin
                flush!(A[s].entries)
                if TCS.cstep == 1 || (TCS.cstep % skip_update[s] == 0 && skip_update[s] != -1)
                    update!(LS[s])
                end
                solve!(LS[s])
            end

            ## CHECK LINEAR RESIDUAL
            res[s].entries[:] = A[s].entries*x[s].entries - b[s].entries
            for j = 1 : length(fixed_dofs)
                for eq = 1 : length(subiterations[s])
                    if fixed_dofs[j] > eqoffsets[s][eq] && fixed_dofs[j] <= eqoffsets[s][eq]+X[subiterations[s][eq]].FES.ndofs
                        eqdof = fixed_dofs[j] - eqoffsets[s][eq]
                        res[s][eq][eqdof] = 0
                    end
                end
            end
            statistics[subiterations[s][:],1] .= 0
            for j = 1 : length(subiterations[s])
                statistics[subiterations[s][j],1] += sum(res[s][j][:].^2)
            end
            linresnorm = norm(res[s].entries)

            # WRITE x[s] INTO X and COMPUTE CHANGE
            for j = 1 : length(subiterations[s])
                for k = 1 : length(LastIterate[subiterations[s][j]])
                    statistics[subiterations[s][j],2] += (LastIterate[subiterations[s][j]][k] - x[s][j][k])^2
                    X[subiterations[s][j]][k] = x[s][j][k]
                end
            end

            ## REASSEMBLE NONLINEAR PARTS
            if maxiterations > 1 || check_nonlinear_residual
                lhs_erased, rhs_erased = assemble!(A[s],b[s],PDE,SC,X; equations = subiterations[s], min_trigger = AssemblyAlways, time = TCS.ctime)

                ## REPAIR TIME DERIVATIVE IF NEEDED
                for k = 1 : length(subiterations[s])
                    d = subiterations[s][k]
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
                        for eq = 1 : length(subiterations[s])
                            if fixed_dofs[j] > eqoffsets[s][eq] && fixed_dofs[j] <= eqoffsets[s][eq]+X[subiterations[s][eq]].FES.ndofs
                                eqdof = fixed_dofs[j] - eqoffsets[s][eq]
                                res[s][eq][eqdof] = 0
                            end
                        end
                    end
                    statistics[subiterations[s][:],3] .= 0
                    for j = 1 : length(subiterations[s])
                        statistics[subiterations[s][j],3] += sum(res[s][j][:].^2)
                    end
                    resnorm = norm(res[s].entries)
                else
                    statistics[subiterations[s][:],3] .= statistics[subiterations[s][:],1]
                end
            else
                statistics[subiterations[s][:],3] .= 1e99 # nonlinear residual has not been checked !
            end

            if sqrt(sum(resnorm.^2)) < target_residual
                break
            end
        end

    end
    
    # REALIZE GLOBAL GLOBALCONSTRAINTS 
    # known bug: this will only work if no components in front of the constrained component(s)
    # are missing in the subiteration
    for j = 1 : length(PDE.GlobalConstraints)
        realize_constraint!(X,PDE.GlobalConstraints[j])
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
function advance_until_stationarity!(TCS::TimeControlSolver, timestep; stationarity_threshold = 1e-11, maxTimeSteps = 100, do_after_each_timestep = nothing, show_details = true)
    statistics = zeros(Float64,length(TCS.X),3)
    maxiterations = TCS.SC.user_params[:maxiterations]
    check_nonlinear_residual = TCS.SC.user_params[:check_nonlinear_residual]
    @info "Advancing in time until stationarity..."
    if show_details
        if maxiterations > 1 || check_nonlinear_residual
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
        @printf("\n")
    end
    for iteration = 1 : maxTimeSteps
        statistics = advance!(TCS, timestep)
        if show_details
            @printf("    %4d  ",iteration)
            @printf("| %.4e ",TCS.ctime)
            @printf("| %.4e |",sqrt(sum(statistics[:,1].^2)))
            if maxiterations > 1 || check_nonlinear_residual
                @printf(" %.4e (%d) |",sqrt(sum(statistics[:,3].^2)), statistics[1,4])
            end
            for j = 1 : size(statistics,1)
                @printf(" %.4e ",statistics[j,2])
            end
            @printf("\n")
        end
        if do_after_each_timestep != nothing
            do_after_each_timestep(TCS.cstep, statistics)
        end
        if sum(statistics[:,2]) < stationarity_threshold
            @info "stationarity detected after $iteration timesteps"
            break;
        end
        if iteration == maxTimeSteps 
            @warn "maxTimeSteps reached"
        end
    end

    if show_details
        show_statistics(TCS.PDE,TCS.SC)
    end
end


"""
````
advance_until_time!(TCS::TimeControlSolver, timestep, finaltime; finaltime_tolerance = 1e-15, do_after_each_timestep = nothing)
````

Advances a TimeControlSolver in time with the given (initial) timestep until the specified finaltime is reached (up to the specified tolerance).
The function do_after_timestep is called after each timestep and can be used to print/save data (and maybe timestep control in future).
"""
function advance_until_time!(TCS::TimeControlSolver, timestep, finaltime; finaltime_tolerance = 1e-15, do_after_each_timestep = nothing, show_details = true)
    statistics = zeros(Float64,length(TCS.X),3)
    maxiterations = TCS.SC.user_params[:maxiterations]
    check_nonlinear_residual = TCS.SC.user_params[:check_nonlinear_residual]
    @info "Advancing in time from $(TCS.ctime) until $finaltime"
    if show_details
        if maxiterations > 1 || check_nonlinear_residual
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
        @printf("\n")
    end
    while TCS.ctime < finaltime - finaltime_tolerance
        statistics = advance!(TCS, timestep)
        if show_details
            @printf("    %4d  ",TCS.cstep)
            @printf("| %.4e ",TCS.ctime)
            @printf("| %.4e |",sqrt(sum(statistics[:,1].^2)))
            if maxiterations > 1 || check_nonlinear_residual
                @printf(" %.4e (%d) |",sqrt(sum(statistics[:,3].^2)), statistics[1,4])
            end
            for j = 1 : size(statistics,1)
                @printf(" %.4e ",statistics[j,2])
            end
            @printf("\n")
        end
        if do_after_each_timestep != nothing
            do_after_each_timestep(TCS.cstep, statistics)
        end
    end

    if show_details
        show_statistics(TCS.PDE,TCS.SC)
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

    @info "Advancing in time from $(sys.ctime) until $finaltime using $DiffEQ with solver = $solver"

    ## generate ODE problem
    f = DiffEQ.ODEFunction(eval_rhs!, jac=eval_jacobian!, jac_prototype=jac_prototype(sys), mass_matrix=mass_matrix(sys))
    prob = DiffEQ.ODEProblem(f,sys.X.entries, (sys.ctime,finaltime),sys)

    ## solve ODE problem
    sol = DiffEQ.solve(prob,solver, abstol=abstol, reltol=reltol, dt = timestep, dtmin = dtmin, initializealg=DiffEQ.NoInit(), adaptive = adaptive)

    ## pickup solution at final time
    sys.X.entries .= sol[:,end]
end