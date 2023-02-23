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

mutable struct SolverConfig{T, TvG, TiG}
    is_nonlinear::Bool      # (subiterations pf) PDE were detected to be nonlinear
    is_timedependent::Bool  # (subiterations pf) PDE were detected to be time_dependent
    LHS_AssemblyTriggers::Array{DataType,2} # assembly triggers for blocks in LHS
    LHS_dependencies::Array{Array{Int,1},2} # dependencies on components
    RHS_AssemblyTriggers::Array{DataType,1} # assembly triggers for blocks in RHS
    RHS_dependencies::Array{Array{Int,1},1} # dependencies on components
    LHS_AssemblyPatterns::Array{Array{AssemblyPattern,1},2} # last used assembly pattern (and assembly pattern preps to avoid their recomputation)
    RHS_AssemblyPatterns::Array{Array{AssemblyPattern,1},1} # last used assembly pattern (and assembly pattern preps to avoid their recomputation)
    LHS_AssemblyTimes::Array{Array{Float64,1},2}
    RHS_AssemblyTimes::Array{Array{Float64,1},1}
    LHS_TotalLoopAllocations::Array{Array{Int,1},2}
    RHS_TotalLoopAllocations::Array{Array{Int,1},1}
    LHS_LastLoopAllocations::Array{Array{Int,1},2}
    RHS_LastLoopAllocations::Array{Array{Int,1},1}
    user_params::Union{Dict{Symbol,Any},Nothing} # dictionary with user parameters
end

#
# Default context information with help info.
#
default_solver_kwargs()=Dict{Symbol,Tuple{Any,String,Bool,Bool}}(
    :subiterations => ("auto", "an array of equation subsets (each an array) that should be solved together in each fixpoint iteration",true,true),
    :timedependent_equations => ([], "array of the equations that should get a time derivative (only for TimeControlSolver)",false,true),
    :target_residual => (1e-12, "stop fixpoint iterations if the absolute (nonlinear) residual is smaller than this number",true,true),
    :abstol => (1e-12, "abstol for linear solver (if iterative)",true,true),
    :reltol => (1e-12, "reltol for linear solver (if iterative)",true,true),
    :maxiterations => ("auto", "maximal number of nonlinear iterations (TimeControlSolver runs that many in each time step)",true,true),
    :check_nonlinear_residual => ("auto", "check the nonlinear residual in last nonlinear iteration (causes one more reassembly of nonlinear terms)",true,true),
    :time => (0, "time at which time-dependent data functions are evaluated or initial time for TimeControlSolver",true,true),
    :skip_update => ([1], "matrix update (for the j-th sub-iteration) will be performed each skip_update[j] iteration; -1 means only in the first iteration",true,true),
    :damping => (0, "damp the new iteration with this part of the old iteration (0 = undamped), also a function is allowed with the interface (old_iterate, new_iterate, fixed_dofs) that returns the new damping value",true,false),
    :anderson_iterations => (0, "use Anderson acceleration with this many previous iterates (to hopefully speed up/enable convergence of fixpoint iterations)",true,false),
    :anderson_metric => ("l2", "String that encodes the desired convergence metric for the Anderson acceleration (possible values: ''l2'' or ''L2'' or ''H1'')",true,false),
    :anderson_damping => (1, "Damping factor in Anderson acceleration (1 = undamped)",true,false),
    :anderson_unknowns => ([1], "an array of unknown numbers that should be included in the Anderson acceleration",true,false),
    :fixed_penalty => (1e60, "penalty that is used for the values of fixed degrees of freedom (e.g. by Dirichlet boundary data or global constraints)",true,true),
    :parallel_storage => (false, "assemble storaged operators in parallel for loop", true, true),
    :linsolver => (UMFPACKFactorization, "any AbstractFactorization from LinearSolve.jl (default = UMFPACKFactorization)",true,true),
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
            if typeof(v[1]) <: String
                println(lines_out,"  - $(k): $(v[2]). Default: ''$(v[1])''\n")
            else
                println(lines_out,"  - $(k): $(v[2]). Default: $(v[1])\n")
            end
        end
    end
    
    String(take!(lines_out))
end

#
# Update solver params from dict
#
function _update_solver_params!(user_params,kwargs)
    for (k,v) in kwargs
        user_params[Symbol(k)] = v
    end
    return nothing
end

function _LinearProblem(A,b,SC::SolverConfig)
    @logmsg MoreInfo "Initializing linear solver (linsolver = $(SC.user_params[:linsolver]))..."
    LP = LinearProblem(A, b)

    linear_cache = init(
        LP,
        SC.user_params[:linsolver];
        abstol = SC.user_params[:abstol],
        reltol = SC.user_params[:reltol]
    )

    return linear_cache
end


function set_nonzero_pattern!(A::FEMatrix, AT::Type{<:AssemblyType} = ON_CELLS)
    @debug "Setting nonzero pattern for FEMatrix..."
    for block = 1 : length(A)
        apply_nonzero_pattern!(A[block],AT)
    end
end


"""
````
SolverConfig{T,TvG,TiG}(PDE::PDEDescription; kwargs...) where {T,TvG,TiG}
````

Generates a solver configuration (that can used for triggering assembly manually).

Keyword arguments:
$(_myprint(default_solver_kwargs(),false))

"""
function SolverConfig{T,TvG,TiG}(PDE::PDEDescription; kwargs...) where {T,TvG,TiG}
    user_params=Dict{Symbol,Any}( k => v[1] for (k,v) in default_solver_kwargs())
    _update_solver_params!(user_params,kwargs)
    return SolverConfig{T,TvG,TiG}(PDE, user_params)
end


function SolverConfig(PDE::PDEDescription, ::FEVector{T,TvG,TiG}; kwargs...) where {T,TvG,TiG}
    user_params=Dict{Symbol,Any}( k => v[1] for (k,v) in default_solver_kwargs())
    _update_solver_params!(user_params,kwargs)
    return SolverConfig{T,TvG,TiG}(PDE, user_params)
end

# check if PDE is nonlinear or time-dependent and which blocks require recalculation
# and devise some initial solver strategy
function SolverConfig{T,TvG,TiG}(PDE::PDEDescription, user_params) where {T,TvG,TiG}

    ## declare subiterations
    if user_params[:subiterations] == "auto"
        user_params[:subiterations] = [1:size(PDE.LHSOperators,1)]
    end
    ## declare linear solver
#    @assert user_params[:linsolver] <: AbstractFactorization
    while length(user_params[:skip_update]) < length(user_params[:subiterations])
        push!(user_params[:skip_update], 1)
    end

    ## check if subiterations are nonlinear or time-dependent
    nonlinear::Bool = false
    timedependent::Bool = false
    block_nonlinear::Bool = false
    block_timedependent::Bool = false
    block_trigger::Type{<:AbstractAssemblyTrigger} = AssemblyNever
    op_nonlinear::Bool = false
    op_timedependent::Bool = false
    LHS_ATs = Array{DataType,2}(undef,size(PDE.LHSOperators,1),size(PDE.LHSOperators,2))
    LHS_dep = Array{Array{Int,1},2}(undef,size(PDE.LHSOperators,1),size(PDE.LHSOperators,2))
    current_dep::Array{Int,1} = []
    all_equations::Array{Int,1} = []
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
        block_trigger = AssemblyNever
        current_dep = [j,k]
        for o = 1 : length(PDE.LHSOperators[j,k])
            # check nonlinearity and time-dependency of operator
            op_nonlinear, op_timedependent, op_trigger = check_PDEoperator(PDE.LHSOperators[j,k][o],all_equations)
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
            if block_trigger <: op_trigger
                block_trigger = op_trigger
            end
            @debug "PDEoperator $o at LHS[$j,$k] is $(op_nonlinear ? "nonlinear" : "linear")$(op_timedependent ? " and timedependent" : "") and has the assembly trigger $op_trigger"
        end
        # check nonlinearity and time-dependency of whole block
        # and assign appropriate AssemblyTrigger
        if length(PDE.LHSOperators[j,k]) == 0
            LHS_ATs[j,k] = AssemblyNever
        else
            LHS_ATs[j,k] = block_trigger
            if block_timedependent== true
                timedependent = true
            end
            if block_nonlinear == true
                nonlinear = true
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
        block_trigger = AssemblyNever
        current_dep = []
        for o = 1 : length(PDE.RHSOperators[j])
            op_nonlinear, op_timedependent, op_trigger = check_PDEoperator(PDE.RHSOperators[j][o],all_equations)
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
            if block_trigger <: op_trigger
                block_trigger = op_trigger
            end

            @debug "PDEoperator $o at RHS[$j] is $(op_nonlinear ? "nonlinear" : "linear")$(op_timedependent ? " and timedependent" : "") and has the assembly trigger $op_trigger"
        end
        if length(PDE.RHSOperators[j]) == 0
            RHS_ATs[j] = AssemblyNever
        else
            RHS_ATs[j] = block_trigger
            if block_timedependent== true
                timedependent = true
            end
            if block_nonlinear == true
                nonlinear = true
            end
        end
        # assign dependencies
        RHS_dep[j] = deepcopy(Base.unique(current_dep))
    end
    LHS_APs = Array{Array{AssemblyPattern,1},2}(undef,size(PDE.LHSOperators,1),size(PDE.LHSOperators,1))
    RHS_APs = Array{Array{AssemblyPattern,1},1}(undef,size(PDE.RHSOperators,1))
    LHS_AssemblyTimes = Array{Array{Float64,1},2}(undef,size(PDE.LHSOperators,1),size(PDE.LHSOperators,2))
    RHS_AssemblyTimes = Array{Array{Float64,1},1}(undef,size(PDE.RHSOperators,1))
    LHS_TotalLoopAllocations = Array{Array{Int,1},2}(undef,size(PDE.LHSOperators,1),size(PDE.LHSOperators,2))
    RHS_TotalLoopAllocations = Array{Array{Int,1},1}(undef,size(PDE.RHSOperators,1))
    LHS_LastLoopAllocations = Array{Array{Int,1},2}(undef,size(PDE.LHSOperators,1),size(PDE.LHSOperators,2))
    RHS_LastLoopAllocations = Array{Array{Int,1},1}(undef,size(PDE.RHSOperators,1))
    for j = 1 : size(PDE.LHSOperators,1), k = 1 : size(PDE.LHSOperators,2)
        LHS_APs[j,k] = Array{AssemblyPattern,1}(undef, length(PDE.LHSOperators[j,k]))
        LHS_AssemblyTimes[j,k] = zeros(Float64, length(PDE.LHSOperators[j,k]))
        LHS_LastLoopAllocations[j,k] = zeros(Int, length(PDE.LHSOperators[j,k]))
        LHS_TotalLoopAllocations[j,k] = zeros(Int, length(PDE.LHSOperators[j,k]))
        for o = 1 : length(PDE.LHSOperators[j,k])
            LHS_APs[j,k][o] = AssemblyPattern()
        end
    end
    for j = 1 : size(PDE.RHSOperators,1)
        RHS_APs[j] = Array{AssemblyPattern,1}(undef, length(PDE.RHSOperators[j]))
        RHS_AssemblyTimes[j] = zeros(Float64, length(PDE.RHSOperators[j]))
        RHS_LastLoopAllocations[j] = zeros(Int, length(PDE.RHSOperators[j]))
        RHS_TotalLoopAllocations[j] = zeros(Int, length(PDE.RHSOperators[j]))
        for o = 1 : length(PDE.RHSOperators[j])
            RHS_APs[j][o] = AssemblyPattern()
        end
    end
    ## correct ATs for blocks not actively solved in the subiterations
    for s = 1 : length(subiterations), k = 1 : size(PDE.LHSOperators,1)
        if (k in subiterations[s]) == false
            for j in subiterations[s]
                if LHS_ATs[j,k] == AssemblyInitial
                    @debug "AssemblyTrigger of block [$j,$k] upgraded to AssemblyEachTimeStep due to subiteration configuration"
                    LHS_ATs[j,k] = AssemblyEachTimeStep
                    if RHS_ATs[j] == AssemblyInitial
                        @debug "AssemblyTrigger of rhs block [$j] upgraded to AssemblyEachTimeStep due to subiteration configuration"
                        RHS_ATs[j] = AssemblyEachTimeStep
                    end
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

    return SolverConfig{T,TvG,TiG}(nonlinear, timedependent, LHS_ATs, LHS_dep, RHS_ATs, RHS_dep, LHS_APs, RHS_APs, LHS_AssemblyTimes, RHS_AssemblyTimes, LHS_TotalLoopAllocations, RHS_TotalLoopAllocations, LHS_LastLoopAllocations, RHS_LastLoopAllocations, user_params)
end



function Base.show(io::IO, SC::SolverConfig)

    println(io, "\nSOLVER-CONFIGURATION")
    println(io, "======================")
    println(io, "  overall nonlinear = $(SC.is_nonlinear)")
    println(io, "  overall timedependent = $(SC.is_timedependent)")

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
    println(io, "                     (I = Once, T = EachTimeStep/SubIteration, A = Always, N = Never)")
    println(io, "\n  LHS_dependencies = $(SC.LHS_dependencies)\n")
end

function show_statistics(PDE::PDEDescription, SC::SolverConfig)

    subiterations = SC.user_params[:subiterations]

    info_msg = "=================================== STATISTICS ==================================="
    info_msg *= "\n\top position \t| runtime (s) \t| last alloc \t| total alloc \t| op name"
    info_msg *= "\n\t----------------------------------------------------------------------------------"

    for s = 1 : length(subiterations)
        for j = 1 : length(subiterations[s])
            eq = subiterations[s][j]
            for k = 1 : size(SC.LHS_AssemblyTimes,2)
                for o = 1 : size(SC.LHS_AssemblyTimes[eq,k],1)
                    info_msg *= "\n\tLHS[$eq,$k][$o] \t| $(@sprintf("%.4e", SC.LHS_AssemblyTimes[eq,k][o])) \t| $(@sprintf("%.4e", SC.LHS_LastLoopAllocations[eq,k][o])) \t| $(@sprintf("%.4e", SC.LHS_TotalLoopAllocations[eq,k][o])) \t| $(parse_unknowns(PDE.LHSOperators[eq,k][o].name, PDE, (eq,k), PDE.LHSOperators[eq,k][o].fixed_arguments_ids))"
                end
            end
        end
    end

    for s = 1 : length(subiterations)
        for j = 1 : length(subiterations[s])
            eq = subiterations[s][j]
            for o = 1 : size(SC.RHS_AssemblyTimes[eq],1)
                info_msg *= "\n\tRHS[$eq,][$o] \t| $(@sprintf("%.4e", SC.RHS_AssemblyTimes[eq][o])) \t| $(@sprintf("%.4e", SC.RHS_LastLoopAllocations[eq][o])) \t| $(@sprintf("%.4e", SC.RHS_TotalLoopAllocations[eq][o])) \t| $(parse_unknowns(PDE.RHSOperators[eq][o].name, PDE, (eq), PDE.RHSOperators[eq][o].fixed_arguments_ids))"
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
    min_trigger::Type{<:AbstractAssemblyTrigger} = AssemblyInitial,
    storage_trigger = "same as min_trigger",
    only_lhs::Bool = false,
    only_rhs::Bool = false) where {T}
    
    @assert only_lhs * only_rhs == 0 "Cannot assemble with only_lhs and only_rhs both true"

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
    LHSOperators::Array{Array{AbstractPDEOperator,1},2} = PDE.LHSOperators
    RHSOperators::Array{Array{AbstractPDEOperator,1},1} = PDE.RHSOperators

    stored_operators = []

    nequations::Int = length(equations)
    # force (re)assembly of stored bilinearforms and LinearForms
    if !only_rhs || only_lhs
        op_nonlinear::Bool = false
        op_timedependent::Bool = false
        op_trigger::Type{<:AbstractAssemblyTrigger} = AssemblyNever
        @logmsg DeepInfo "Locking for triggered storage updates..."
        for j = 1:nequations
            for k = 1 : size(LHSOperators,2)
                if (storage_trigger <: SC.LHS_AssemblyTriggers[equations[j],k]) == true
                    for o = 1 : length(LHSOperators[equations[j],k])
                        O = LHSOperators[equations[j],k][o]
                        if has_storage(O)
                            op_nonlinear, op_timedependent, op_trigger = check_PDEoperator(LHSOperators[equations[j],k][o],equations)
                            if (min_trigger <: op_trigger) 
                                push!(stored_operators, [O,j,k,o])
                            end
                        end
                    end
                end
            end
        end
        for j = 1:nequations
            if (storage_trigger <: SC.RHS_AssemblyTriggers[equations[j]]) == true
                for o = 1 : length(RHSOperators[equations[j]])
                    O = RHSOperators[equations[j]][o]
                    if has_storage(O)
                        op_nonlinear, op_timedependent, op_trigger = check_PDEoperator(RHSOperators[equations[j]][o],equations)
                        if (min_trigger <: op_trigger) 
                            push!(stored_operators, [O,j,-1,o])
                        end
                    end
                end
            end
        end
    end

    ## run storage updates in parallel threads
    if SC.user_params[:parallel_storage]
        Threads.@threads for s = 1 : length(stored_operators)
            O = stored_operators[s][1]
            j = stored_operators[s][2]::Int
            k = stored_operators[s][3]::Int
            o = stored_operators[s][4]::Int
            if k > 0
                elapsedtime = @elapsed SC.LHS_LastLoopAllocations[equations[j],k][o] = update_storage!(O, SC, CurrentSolution, equations[j], k, o; time = time)
                SC.LHS_TotalLoopAllocations[equations[j],k][o] += SC.LHS_LastLoopAllocations[equations[j],k][o]
                SC.LHS_AssemblyTimes[equations[j],k][o] += elapsedtime
            else
                elapsedtime = @elapsed SC.RHS_LastLoopAllocations[equations[j]][o] = update_storage!(O, SC, CurrentSolution, equations[j], o ; time = time)
                SC.RHS_TotalLoopAllocations[equations[j]][o] += SC.RHS_LastLoopAllocations[equations[j]][o]
                SC.RHS_AssemblyTimes[equations[j]][o] += elapsedtime
            end
        end
    else
        for s = 1 : length(stored_operators)
            O = stored_operators[s][1]
            j = stored_operators[s][2]::Int
            k = stored_operators[s][3]::Int
            o = stored_operators[s][4]::Int
            if k > 0
                elapsedtime = @elapsed SC.LHS_LastLoopAllocations[equations[j],k][o] = update_storage!(O, SC, CurrentSolution, equations[j], k, o; time = time)
                SC.LHS_TotalLoopAllocations[equations[j],k][o] += SC.LHS_LastLoopAllocations[equations[j],k][o]
                SC.LHS_AssemblyTimes[equations[j],k][o] += elapsedtime
            else
                elapsedtime = @elapsed SC.RHS_LastLoopAllocations[equations[j]][o] = update_storage!(O, SC, CurrentSolution, equations[j], o ; time = time)
                SC.RHS_TotalLoopAllocations[equations[j]][o] += SC.RHS_LastLoopAllocations[equations[j]][o]
                SC.RHS_AssemblyTimes[equations[j]][o] += elapsedtime
            end
        end
    end

    # (re)assembly right-hand side
    # if NonlinearOperators are in the LHS we need to trigger reassembly of the right-hand side !!!!
    elapsedtime::Float64 = 0
    rhs_block_has_been_erased = zeros(Bool,nequations)
    nonlinear_operator_present::Bool = false
    if !only_lhs || only_rhs
        for j = 1 : nequations
            # check for nonlinear operators that require reassembly
            nonlinear_operator_present = false
            for k = 1 : size(LHSOperators,2)
                if length(intersect(SC.LHS_dependencies[equations[j],k], if_depends_on)) > 0
                    if (k in equations) && !only_rhs # LHS operator is assembled into A if variable is not solved in equations
                        for o = 1 : length(LHSOperators[equations[j],k])
                            O = LHSOperators[equations[j],k][o]
                            if get_pattern(O) <: APT_NonlinearForm
                                nonlinear_operator_present = true
                            end
                        end
                    end
                end
            end
            if (min_trigger <: SC.RHS_AssemblyTriggers[equations[j]]) && (length(RHSOperators[equations[j]]) > 0) || nonlinear_operator_present
                @debug "Erasing rhs block [$j]"
                fill!(b[j],0.0)
                rhs_block_has_been_erased[j] = true
                @logmsg DeepInfo "Locking what to assemble in rhs block [$(equations[j])]..."
                for o = 1 : length(RHSOperators[equations[j]])
                    O = RHSOperators[equations[j]][o]
                    elapsedtime = @elapsed assemble!(b[j], SC, equations[j], o, O, CurrentSolution; time = time)
                    SC.RHS_AssemblyTimes[equations[j]][o] += elapsedtime
                    if !has_storage(O)
                        SC.RHS_LastLoopAllocations[equations[j]][o] = SC.RHS_AssemblyPatterns[equations[j]][o].last_allocations
                        SC.RHS_TotalLoopAllocations[equations[j]][o] += SC.RHS_AssemblyPatterns[equations[j]][o].last_allocations
                    end
                end
            end
        end
    end

    # (re)assembly left-hand side

    lhs_block_has_been_erased = zeros(Bool,nequations,nequations)
    subblock::Int = 0
    for j = 1:nequations
        for k = 1 : size(LHSOperators,2)
            if length(intersect(SC.LHS_dependencies[equations[j],k], if_depends_on)) > 0
                if (k in equations) && !only_rhs # LHS operator is assembled into A if variable is not solved in equations
                    subblock += 1
                    #println("\n  Equation $j, subblock $subblock")
                    if (min_trigger <: SC.LHS_AssemblyTriggers[equations[j],k]) && (length(LHSOperators[equations[j],k]) > 0)
                        @debug "Erasing lhs block [$j,$subblock]"
                        fill!(A[j,subblock],0)
                        lhs_block_has_been_erased[j, subblock] = true
                        @logmsg DeepInfo "Locking what to assemble in lhs block [$(equations[j]),$k]..."
                        for o = 1 : length(LHSOperators[equations[j],k])
                            O = LHSOperators[equations[j],k][o]
                            if get_pattern(O) <: APT_NonlinearForm
                                if rhs_block_has_been_erased[j] == false
                                    blah[:] = 0
                                    @debug "Erasing rhs block [$j]"
                                    fill!(b[j],0)
                                    rhs_block_has_been_erased[j] = true
                                end
                                elapsedtime = @elapsed full_assemble!(A[j,subblock], b[j], SC, equations[j],k,o, O, CurrentSolution; time = time)
                            else
                                if has_copy_block(O)
                                    elapsedtime = @elapsed assemble!(A[j,subblock], SC, equations[j],k,o, O, CurrentSolution; time = time, At = A[subblock,j])
                                else
                                    elapsedtime = @elapsed assemble!(A[j,subblock], SC, equations[j],k,o, O, CurrentSolution; time = time)
                                end  
                            end
                            SC.LHS_AssemblyTimes[equations[j],k][o] += elapsedtime
                            if !has_storage(O)
                                SC.LHS_LastLoopAllocations[equations[j],k][o] = SC.LHS_AssemblyPatterns[equations[j],k][o].last_allocations
                                SC.LHS_TotalLoopAllocations[equations[j],k][o] += SC.LHS_AssemblyPatterns[equations[j],k][o].last_allocations
                            end
                        end  
                    end
                elseif !(k in equations) && !only_lhs # LHS operator is assembled into b if variable is not solved in equations
                    if (min_trigger <: SC.LHS_AssemblyTriggers[equations[j],k]) == true
                        if (length(LHSOperators[equations[j],k]) > 0) && (!(min_trigger <: SC.RHS_AssemblyTriggers[equations[j]]))
                            if rhs_block_has_been_erased[j] == false
                                @debug "Erasing rhs block [$j]"
                                fill!(b[j],0)
                                rhs_block_has_been_erased[j] = true
                            end
                        end
                        for o = 1 : length(LHSOperators[equations[j],k])
                            O = LHSOperators[equations[j],k][o]
                            @debug "Assembling lhs block[$j,$k] into rhs block[$j] ($k not in equations): $(O.name)"
                            elapsedtime = @elapsed assemble!(b[j], SC, equations[j],k, o, O, CurrentSolution; factor = -1.0, time = time, fixed_component = k)
                            SC.LHS_AssemblyTimes[equations[j],k][o] += elapsedtime
                            if !has_storage(O)
                                SC.LHS_LastLoopAllocations[equations[j],k][o] = SC.LHS_AssemblyPatterns[equations[j],k][o].last_allocations
                                SC.LHS_TotalLoopAllocations[equations[j],k][o] += SC.LHS_AssemblyPatterns[equations[j],k][o].last_allocations
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

function apply_boundarydata!(A,b,Solution,k,Problem,SC; time = 0)
    penalty = SC.user_params[:fixed_penalty]
    fixed_dofs = boundarydata!(Solution[k], Problem.BoundaryOperators[k], Solution; time = time)
    fixed_dofs .+= Solution[k].offset
    @views b.entries[fixed_dofs] .= penalty * Solution.entries[fixed_dofs]
    apply_penalties!(A.entries, fixed_dofs, penalty)
    return fixed_dofs
end

# for linear, stationary PDEs that can be solved in one step
function solve_direct!(Target::FEVector{T,Tv,Ti}, PDE::PDEDescription, SC::SolverConfig{T,Tv,Ti}; time::Real = 0, show_details::Bool = false) where {T, Tv, Ti}

    assembly_time = @elapsed begin

    # ASSEMBLE SYSTEM
    FEs = FESpaces(Target)
    A = FEMatrix{T}(FEs; name = "system matrix")
    b = FEVector{T}(FEs; name = "system rhs")
    assemble!(A,b,PDE,SC,Target; equations = Array{Int,1}(1:length(FEs)), min_trigger = AssemblyInitial, time = time)

    # ASSEMBLE BOUNDARY DATA
    fixed_dofs = []
    for j = 1 : length(Target.FEVectorBlocks)
        new_fixed_dofs = boundarydata!(Target[j], PDE.BoundaryOperators[j], Target; time = time)
        new_fixed_dofs .+= Target[j].offset
        append!(fixed_dofs, new_fixed_dofs)
    end    

    # PREPARE GLOBALCONSTRAINTS
    flush!(A.entries)
    for j = 1 : length(PDE.GlobalConstraints)
        additional_fixed_dofs = apply_constraint!(A, b, PDE.GlobalConstraints[j], Target)
        append!(fixed_dofs,additional_fixed_dofs)
    end

    # PENALIZE FIXED DOFS
    # (from boundary conditions and constraints)
    fixed_dofs::Array{Ti,1} = Base.unique(fixed_dofs)
    fixed_penalty = SC.user_params[:fixed_penalty]
    @views b.entries[fixed_dofs] .= fixed_penalty * Target.entries[fixed_dofs]
    apply_penalties!(A.entries, fixed_dofs, fixed_penalty)
    end # @elapsed

    # SOLVE
    solver_time = @elapsed begin
        ## solve via LinearSolve.jl
        linear_cache = _LinearProblem(A.entries.cscmatrix, b.entries, SC)
        Target.entries .= LinearSolve.solve(linear_cache)

        # CHECK RESIDUAL
        residual = deepcopy(b)
        fill!(residual.entries, 0)
        mul!(residual.entries, A.entries, Target.entries)
        residual.entries .-= b.entries
        residual.entries[fixed_dofs] .= 0
        residuals = norms(residual)

        # REALIZE GLOBAL GLOBALCONSTRAINTS 
        # (possibly changes some entries of Target)
        for j = 1 : length(PDE.GlobalConstraints)
            realize_constraint!(Target, PDE.GlobalConstraints[j])
        end
    end

    if SC.user_params[:show_statistics]
        @info "assembly/solver time = $(assembly_time)/$(solver_time) (s)"
        show_statistics(PDE,SC)
    end

    if SC.user_params[:show_iteration_details]
        @info "overall residual = $(norm(residuals))"
    end
    return norm(residuals)
end

mutable struct AndersonAccelerationManager{T,Tv,Ti}
    LastIterates::Array{FEVector{T,Tv,Ti},1}
    LastIteratesTilde::Array{FEVector{T,Tv,Ti},1}
    NormOperator::AbstractMatrix{T}
    Matrix::AbstractMatrix{T}
    Rhs::AbstractVector{T}
    alpha::AbstractVector{T}
    anderson_iterations::Int
    cdepth::Int
    Target::FEVector{T,Tv,Ti}
    anderson_unknowns::Array{Int,1}
    damping::T
end

function AndersonAccelerationManager(anderson_metric::String, Target::FEVector{T,Tv,Ti}; anderson_iterations::Int = 1, anderson_unknowns = "all", anderson_damping = 1) where {T, Tv, Ti}

    @assert anderson_iterations > 0 "Anderson accelerations needs anderson_iterations > 0!"
    @assert anderson_damping > 0 && anderson_damping <= 1 "Anderson damping factor must be >0 and <=1"
    # we need to save the last iterates
    LastIterates = Array{FEVector,1}(undef, anderson_iterations+1) # actual iterates u_k
    LastIteratesTilde = Array{FEVector,1}(undef, anderson_iterations+1) # auxiliary iterates \tilde u_k

    ## assemble convergence metric
    FEs = Array{FESpace{Tv,Ti},1}([])
    for j = 1 : length(Target.FEVectorBlocks)
        push!(FEs,Target.FEVectorBlocks[j].FES)
    end    

    if anderson_unknowns == "all"
        anderson_unknowns = 1 : length(Target.FEVectorBlocks)
    end
    NormOperator = FEMatrix{T}(FEs)
    for u in anderson_unknowns
        if anderson_metric == "L2"
            AA_METRIC = DiscreteSymmetricBilinearForm([Identity, Identity], [FEs[u], FEs[u]], NoAction())
            assemble!(NormOperator[u,u], AA_METRIC)
        elseif anderson_metric == "H1"
                AA_METRIC = DiscreteSymmetricBilinearForm([Gradient,Gradient], [FEs[u], FEs[u]], NoAction())
                assemble!(NormOperator[u,u], AA_METRIC)
        elseif anderson_metric == "l2"
            for j = 1 : FEs[u].ndofs
                NormOperator[u,u][j,j] = 1
            end
        end
    end
    NormOperator = NormOperator.entries
    flush!(NormOperator)

    Matrix = zeros(T, anderson_iterations+2, anderson_iterations+2)
    Rhs = zeros(T, anderson_iterations+2)
    Rhs[anderson_iterations+2] = 1 # sum of all coefficients should equal 1
    alpha = zeros(T, anderson_iterations+2)
    for j = 1 : anderson_iterations+1
        LastIterates[j] = deepcopy(Target)
        LastIteratesTilde[j] = deepcopy(Target)
        Matrix[j,j] = 1
    end
    return AndersonAccelerationManager{T,Tv,Ti}(LastIterates,LastIteratesTilde,NormOperator,Matrix,Rhs,alpha,anderson_iterations,-1,Target,anderson_unknowns, anderson_damping)
end

# call this every time Target has got its new values
function update_anderson!(AAM::AndersonAccelerationManager{T,Tv,Ti}) where {T,Tv,Ti}
    anderson_iterations::Int = AAM.anderson_iterations
    cdepth::Int = min(anderson_iterations,AAM.cdepth)
    alpha::Array{T,1} = AAM.alpha
    damping::T = AAM.damping

    # move last tilde iterates to the front in memory
    for j = 1 : anderson_iterations
        AAM.LastIteratesTilde[j].entries .= AAM.LastIteratesTilde[j+1].entries
    end

    # save fixpoint iterate as new tilde iterate
    AAM.LastIteratesTilde[anderson_iterations+1].entries .= AAM.Target.entries

    if cdepth > 0
        # fill matrix until depth (rest of matrix is still identity matrix from init)
        for j = 1 : cdepth+1, k = j : cdepth+1
            AAM.Matrix[j,k] = ldrdmatmul(AAM.LastIteratesTilde[anderson_iterations+2-j].entries, AAM.LastIterates[anderson_iterations+2-j].entries,
                                    AAM.NormOperator,
                                    AAM.LastIteratesTilde[anderson_iterations+2-k].entries, AAM.LastIterates[anderson_iterations+2-k].entries)
            AAM.Matrix[k,j] = AAM.Matrix[j,k]
            AAM.Matrix[j,anderson_iterations+2] = 1
            AAM.Matrix[anderson_iterations+2,j] = 1
        end

        # solve for alpha coefficients
        alpha .= AAM.Matrix\AAM.Rhs

        # check residual
        res = sqrt(sum((AAM.Matrix * AAM.alpha - AAM.Rhs).^2))
        if res > 1e-4
            @warn "Anderson acceleration local system residual unsatisfactory ($res > 1e-4)"
        end

        # compute next iterate
        offset::Int = 0
        for u in AAM.anderson_unknowns
            fill!(AAM.Target[u],0)
            offset = AAM.Target[u].offset
            for a = 1 : cdepth+1
                for j = 1 : AAM.Target[u].FES.ndofs
                    AAM.Target.entries[offset+j] += damping * alpha[a] * AAM.LastIteratesTilde[anderson_iterations+2-a].entries[offset+j]
                    AAM.Target.entries[offset+j] += (1-damping) * alpha[a] * AAM.LastIterates[anderson_iterations+2-a].entries[offset+j]
                end
            end
        end

        # move last iterates to the front in memory
        for j = 1 : anderson_iterations
            AAM.LastIterates[j].entries .= AAM.LastIterates[j+1].entries
        end
    end

    # save new iterate
    AAM.LastIterates[anderson_iterations+1].entries .= AAM.Target.entries

    ## increase depth for next call
    AAM.cdepth += 1
    return AAM.Target
end


# solve full system iteratively until fixpoint is reached
function solve_fixpoint_full!(Target::FEVector{T,Tv,Ti}, PDE::PDEDescription, SC::SolverConfig{T,Tv,Ti}; time::Real = 0) where {T,Tv,Ti}

    time_total = @elapsed begin

    ## get relevant solver parameters
    fixed_penalty = SC.user_params[:fixed_penalty]
    damping = SC.user_params[:damping]
    damping_val::Float64 = 0
    if typeof(damping) <: Function
        damping_val = 1
    elseif typeof(damping) <: Real
        damping_val = damping
    end
    maxiterations = SC.user_params[:maxiterations]
    target_residual = SC.user_params[:target_residual]
    skip_update = SC.user_params[:skip_update]

    # ASSEMBLE SYSTEM INIT
    FEs = Array{FESpace{Tv,Ti},1}([])
    for j = 1 : length(Target.FEVectorBlocks)
        push!(FEs,Target.FEVectorBlocks[j].FES)
    end    
    assembly_time = @elapsed begin
        A = FEMatrix{T}(FEs; name = "system matrix")
        b = FEVector{T}(FEs; name = "system rhs")
       # set_nonzero_pattern!(A)
        assemble!(A,b,PDE,SC,Target; time = time, equations = Array{Int,1}(1:length(FEs)), min_trigger = AssemblyInitial)
    end

    # ASSEMBLE BOUNDARY DATA
    fixed_dofs = []
    assembly_time += @elapsed for j= 1 : length(Target.FEVectorBlocks)
        new_fixed_dofs = boundarydata!(Target[j],PDE.BoundaryOperators[j], Target; time = time) .+ Target[j].offset
        append!(fixed_dofs, new_fixed_dofs)
    end    

    # ANDERSON ITERATIONS
    anderson_iterations = SC.user_params[:anderson_iterations]
    anderson_metric = SC.user_params[:anderson_metric]
    anderson_unknowns = SC.user_params[:anderson_unknowns]
    anderson_damping = SC.user_params[:anderson_damping]
    anderson_time = 0
    if anderson_iterations > 0
        anderson_time += @elapsed begin    
            @logmsg MoreInfo "Preparing Anderson acceleration for unknown(s) = $anderson_unknowns with convergence metric $anderson_metric and $anderson_iterations iterations"
            AAM = AndersonAccelerationManager(anderson_metric, Target; anderson_iterations = anderson_iterations, anderson_unknowns = anderson_unknowns, anderson_damping = anderson_damping)
        end
    end
    if damping_val > 0
        LastIterate = deepcopy(Target)
    end

    residual = deepcopy(b)
    linresnorm::T = 0.0
    resnorm::T = 0.0

    ## INIT SOLVER
    time_solver = @elapsed linear_cache = _LinearProblem(A.entries.cscmatrix, b.entries, SC)
    end

    if SC.user_params[:show_iteration_details]
        if SC.user_params[:show_statistics]
            @printf("\n\tITERATION |  LSRESIDUAL  |  NLRESIDUAL  | TIME ASSEMBLY/SOLVE/TOTAL (s)")
            @printf("\n\t-----------------------------------------------------------------------")
            @printf("\n\t    init  |                             | %.2e/%.2e/%.2e\n",assembly_time,time_solver,time_total)
        else
            @printf("\n\tITERATION |  LSRESIDUAL  |  NLRESIDUAL")
            @printf("\n\t--------------------------------------\n")
        end
    end

    overall_time = time_total
    overall_solver_time = time_solver
    overall_assembly_time = assembly_time

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
        @views b.entries[fixed_dofs] .= fixed_penalty * Target.entries[fixed_dofs]
        apply_penalties!(A.entries, fixed_dofs, fixed_penalty)

        # SOLVE
        time_solver = @elapsed begin
            flush!(A.entries)
            if j == 1 || (j % skip_update[1] == 0 && skip_update[1] != -1)
                linear_cache = LinearSolve.set_A(linear_cache, A.entries.cscmatrix)
            end
            linear_cache = LinearSolve.set_b(linear_cache, b.entries)
            sol = LinearSolve.solve(linear_cache)
            linear_cache = sol.cache
            copyto!(Target.entries, sol.u)
        end

        # CHECK LINEAR RESIDUAL
        if SC.user_params[:show_iteration_details]
            fill!(residual.entries,0)
            mul!(residual.entries, A.entries, Target.entries)
            residual.entries .-= b.entries
            residual.entries[fixed_dofs] .= 0
            linresnorm = norm(residual)
        end

        # POSTPROCESS : ANDERSON ITERATE
        if anderson_iterations > 0
            anderson_time += @elapsed begin
                update_anderson!(AAM)
            end
        end

        # POSTPROCESS : DAMPING
        if typeof(damping) <: Function
            damping_val = damping(LastIterate, Target, fixed_dofs)
        end    
        if damping_val > 0
            @logmsg MoreInfo "Damping with value $damping_val"
            Target.entries .= damping_val * LastIterate.entries + (1-damping_val) * Target.entries
        end
        if damping_val > 0 || typeof(damping) <: Function
            LastIterate.entries .= Target.entries
        end

        # REALIZE GLOBAL GLOBALCONSTRAINTS 
        # (possibly changes some entries of Target)
        overall_time += @elapsed for j = 1 : length(PDE.GlobalConstraints)
            #if AssemblyFinal <: PDE.GlobalConstraints[j].when_assemble 
                realize_constraint!(Target,PDE.GlobalConstraints[j])
            #end
        end

        # REASSEMBLE NONLINEAR PARTS
        time_reassembly = @elapsed lhs_erased, rhs_erased = assemble!(A, b, PDE, SC, Target; time = time, equations = 1:size(PDE.RHSOperators,1), min_trigger = AssemblyAlways)

        # PREPARE GLOBALCONSTRAINTS
        flush!(A.entries)
        for j = 1 : length(PDE.GlobalConstraints)
            additional_fixed_dofs = apply_constraint!(A,b,PDE.GlobalConstraints[j],Target; lhs_mask = lhs_erased, rhs_mask = rhs_erased)
            append!(fixed_dofs,additional_fixed_dofs)
        end

        # CHECK NONLINEAR RESIDUAL
        fill!(residual.entries,0)
        mul!(residual.entries, A.entries, Target.entries)
        residual.entries .-= b.entries
        residual.entries[fixed_dofs] .= 0
        resnorm = norm(residual)

        end #elapsed
        
        overall_time += time_total
        overall_solver_time += time_solver
        overall_assembly_time += time_reassembly

        if SC.user_params[:show_iteration_details]
            @printf("\t   %4d  ", j)
            @printf(" | %e", linresnorm)
            @printf(" | %e", resnorm)
            if SC.user_params[:show_statistics]
                @printf(" | %.2e/%.2e/%.2e", time_reassembly, time_solver, time_total)
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

    # REALIZE GLOBAL GLOBALCONSTRAINTS 
    # (possibly changes some entries of Target)
    overall_time += @elapsed for j = 1 : length(PDE.GlobalConstraints)
        if AssemblyFinal <: PDE.GlobalConstraints[j].when_assemble 
            realize_constraint!(Target,PDE.GlobalConstraints[j])
        end
    end

    if SC.user_params[:show_statistics] 
        @printf("\t    total |                             | %.2e/%.2e/%.2e\n\n",overall_assembly_time,overall_solver_time,overall_time)
        if anderson_iterations > 0
            @info "anderson acceleration took $(anderson_time)s"
        end
        show_statistics(PDE,SC)
    elseif SC.user_params[:show_iteration_details]
        @printf("\n")
    end

    return resnorm
end


# solve system iteratively until fixpoint is reached
# by consecutively solving subiterations
function solve_fixpoint_subiterations!(Target::FEVector{T,Tv,Ti}, PDE::PDEDescription, SC::SolverConfig{T,Tv,Ti}; time = 0) where {T,Tv,Ti}

    time_total = @elapsed begin
    ## get relevant solver parameters
    subiterations = SC.user_params[:subiterations]
    fixed_penalty = SC.user_params[:fixed_penalty]
    damping = SC.user_params[:damping]
    damping_val::Float64 = 0
    if typeof(damping) <: Function
        damping_val = 1
    elseif typeof(damping) <: Real
        damping_val = damping
    end
    maxiterations = SC.user_params[:maxiterations]
    target_residual = SC.user_params[:target_residual]
    skip_update = SC.user_params[:skip_update]

    # ASSEMBLE SYSTEM INIT
    assembly_time = @elapsed begin
        FEs = Array{FESpace{Tv,Ti},1}([])
        for j=1 : length(Target.FEVectorBlocks)
            push!(FEs,Target.FEVectorBlocks[j].FES)
        end    
        nsubiterations = length(subiterations)
        eqoffsets::Array{Array{Int,1},1} = Array{Array{Int,1},1}(undef,nsubiterations)
        A = Array{FEMatrix{T},1}(undef,nsubiterations)
        b = Array{FEVector{T},1}(undef,nsubiterations)
        x = Array{FEVector{T},1}(undef,nsubiterations)
        for i = 1 : nsubiterations
            A[i] = FEMatrix{T}(FEs[subiterations[i]]; name = "system matrix $i")
            b[i] = FEVector{T}(FEs[subiterations[i]]; name = "system rhs $i")
            x[i] = FEVector{T}(FEs[subiterations[i]]; name = "system sol $i")
            #set_nonzero_pattern!(A[i])
            assemble!(A[i],b[i],PDE,SC,Target; time = time, equations = subiterations[i], min_trigger = AssemblyInitial)
            eqoffsets[i] = zeros(Int,length(subiterations[i]))
            for j= 1 : length(Target.FEVectorBlocks), eq = 1 : length(subiterations[i])
                if j < subiterations[i][eq]
                    eqoffsets[i][eq] += FEs[j].ndofs
                end
            end

            # FILL x[s] WITH TARGET ENTRIES
            for k = 1 : length(subiterations[i])
                d = subiterations[i][k]
                for dof = 1 : Target[d].FES.ndofs
                    x[i][k][dof] = Target[d][dof]
                end
            end
        end

        # ASSEMBLE BOUNDARY DATA
        fixed_dofs::Array{Int,1} = []
        eqdof::Int = 0
        for j = 1 : length(Target.FEVectorBlocks)
            new_fixed_dofs = boundarydata!(Target[j],PDE.BoundaryOperators[j], Target; time = time) .+ Target[j].offset
            append!(fixed_dofs, new_fixed_dofs)
        end    
    end

    # ANDERSON ITERATIONS
    anderson_iterations = SC.user_params[:anderson_iterations]
    anderson_metric = SC.user_params[:anderson_metric]
    anderson_unknowns = SC.user_params[:anderson_unknowns]
    anderson_damping = SC.user_params[:anderson_damping]
    anderson_time = 0
    if anderson_iterations > 0
        anderson_time += @elapsed begin    
            @logmsg MoreInfo "Preparing Anderson acceleration for unknown(s) = $anderson_unknowns with convergence metric $anderson_metric and $anderson_iterations iterations"
            AAM = AndersonAccelerationManager(anderson_metric, Target; anderson_iterations = anderson_iterations, anderson_unknowns = anderson_unknowns, anderson_damping = anderson_damping)
        end
    end
    if damping_val > 0
        LastIterate = deepcopy(Target)
    end
    
    residual = Array{FEVector{T},1}(undef,nsubiterations)
    for s = 1 : nsubiterations
        residual[s] = FEVector{T}(FEs[subiterations[s]])
    end
    linresnorm = zeros(T,nsubiterations)
    resnorm = zeros(T,nsubiterations)


    ## INIT SOLVERS
    time_solver = @elapsed begin
        linear_cache = Array{LinearSolve.LinearCache,1}(undef,nsubiterations)
        for s = 1 : nsubiterations
            linear_cache[s] = _LinearProblem(A[s].entries.cscmatrix, b[s].entries, SC)
        end
    end

    end

    if SC.user_params[:show_iteration_details]
        if SC.user_params[:show_statistics]
            @printf("\n\tITERATION |  LSRESIDUAL  |  NLRESIDUAL  | TIME ASSEMBLY/SOLVE/TOTAL (s)")
            @printf("\n\t-----------------------------------------------------------------------")
            @printf("\n\t    init  |                             | %.2e/%.2e/%.2e\n",assembly_time,time_solver,time_total)
        else
            @printf("\n\tITERATION |  LSRESIDUAL  |  NLRESIDUAL")
            @printf("\n\t--------------------------------------\n")
        end
    end

    overall_time = time_total
    overall_solver_time = time_solver
    overall_assembly_time = assembly_time
    lhs_erased, rhs_erased = nothing, nothing 

    for iteration = 1 : maxiterations

        time_reassembly = 0
        time_solver = 0
        time_total = @elapsed for s = 1 : nsubiterations

            # PREPARE GLOBALCONSTRAINTS
            # known bug: this will only work if no components in front of the constrained component(s)
            # are missing in the subiteration
            for j = 1 : length(PDE.GlobalConstraints)
                if PDE.GlobalConstraints[j].component in subiterations[s]
                   additional_fixed_dofs = apply_constraint!(A[s],b[s],PDE.GlobalConstraints[j],Target; lhs_mask = lhs_erased, rhs_mask = rhs_erased, current_equations = subiterations[s])
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
                    linear_cache[s] = LinearSolve.set_A(linear_cache[s], A[s].entries.cscmatrix)
                end
                linear_cache[s] = LinearSolve.set_b(linear_cache[s], b[s].entries)
                sol = LinearSolve.solve(linear_cache[s])
                linear_cache[s] = sol.cache
                copyto!(x[s].entries, sol.u)
            end

            # CHECK LINEAR RESIDUAL
            if SC.user_params[:show_iteration_details]
                fill!(residual[s].entries,0)
                mul!(residual[s].entries, A[s].entries, x[s].entries)
                residual[s].entries .-= b[s].entries

                for j = 1 : length(fixed_dofs)
                    for eq = 1 : length(subiterations[s])
                        # check if fixed_dof is necessary for subiteration
                        if fixed_dofs[j] > eqoffsets[s][eq] && fixed_dofs[j] <= eqoffsets[s][eq]+FEs[subiterations[s][eq]].ndofs
                            eqdof = fixed_dofs[j] - eqoffsets[s][eq]::Int
                            residual[s][eq][eqdof] = 0
                        end
                    end
                end
                linresnorm[s] = norm(residual[s])
            end

            # WRITE INTO Target
            for j = 1 : length(subiterations[s])
                si = subiterations[s][j]
                fill!(Target[si],0)
                addblock!(Target[si], x[s][j])
            end

            if s == nsubiterations
                # POSTPROCESS : ANDERSON ITERATE
                if anderson_iterations > 0
                    anderson_time += @elapsed begin
                        update_anderson!(AAM)
                    end
                end

                # POSTPROCESS : DAMPING
                if typeof(damping) <: Function
                    damping_val = damping(LastIterate, Target, fixed_dofs)
                end    
                if damping_val > 0
                    @logmsg MoreInfo "Damping with value $damping_val"
                    for j = 1 : length(Target.entries)
                        Target.entries[j] = damping_val * LastIterate.entries[j] + (1-damping_val) * Target.entries[j]
                    end
                    for j = 1 : length(subiterations[s])
                        for k = 1 : length(Target[subiterations[s][j]])
                            x[s][j][k] = Target[subiterations[s][j]][k]
                        end
                    end
                end
                if damping_val > 0 || typeof(damping) <: Function
                    LastIterate.entries .= Target.entries
                end
            end

            # REASSEMBLE PARTS FOR NEXT SUBITERATION
            time_reassembly += @elapsed begin
                next_eq = (s == nsubiterations) ? 1 : s+1
                lhs_erased, rhs_erased = assemble!(A[next_eq],b[next_eq],PDE,SC,Target; time = time, equations = subiterations[next_eq], min_trigger = AssemblyEachTimeStep)
            end
        end

        # PREPARE GLOBALCONSTRAINTS
        # known bug: this will only work if no components in front of the constrained component(s)
        # are missing in the subiteration
        for s = 1 : nsubiterations
            for j = 1 : length(PDE.GlobalConstraints)
                if PDE.GlobalConstraints[j].component in subiterations[s]
                additional_fixed_dofs = apply_constraint!(A[s],b[s],PDE.GlobalConstraints[j],Target; lhs_mask = lhs_erased, rhs_mask = rhs_erased, current_equations = subiterations[s])
                append!(fixed_dofs, additional_fixed_dofs)
                end
            end
        end

        # CHECK NONLINEAR RESIDUAL
        for s = 1 : nsubiterations
            fill!(residual[s].entries,0)
            mul!(residual[s].entries, A[s].entries, x[s].entries)
            residual[s].entries .-= b[s].entries
            
            for j = 1 : length(fixed_dofs)
                for eq = 1 : length(subiterations[s])
                    if fixed_dofs[j] > eqoffsets[s][eq] && fixed_dofs[j] <= eqoffsets[s][eq]+FEs[subiterations[s][eq]].ndofs
                        eqdof = fixed_dofs[j] - eqoffsets[s][eq]
                        residual[s][eq][eqdof] = 0
                    end
                end
            end
            resnorm[s] = norm(residual[s])
        end

        overall_time += time_total
        overall_solver_time += time_solver
        overall_assembly_time += time_reassembly

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

    # REALIZE GLOBALCONSTRAINTS 
    # (possibly changes some entries of Target)
    overall_time += @elapsed for j = 1 : length(PDE.GlobalConstraints)
        realize_constraint!(Target,PDE.GlobalConstraints[j])
    end

    if SC.user_params[:show_statistics] 
        @printf("\t    total |                             | %.2e/%.2e/%.2e\n\n",overall_assembly_time,overall_solver_time,overall_time)
        if anderson_iterations > 0
            @info "anderson acceleration took $(anderson_time)s"
        end
        show_statistics(PDE,SC)
    elseif SC.user_params[:show_iteration_details]
        @printf("\n")
    end

    return sqrt(sum(resnorm.^2))
end


function FEVector(FES::FESpace, PDE::PDEDescription) where {Tv, Ti}
    return FEVector([FES], PDE)
end

function FEVector(FES::Array{<:FESpace{Tv,Ti}}, PDE::PDEDescription) where {Tv, Ti}
    @assert length(PDE.unknown_names) == length(FES)
    return FEVector(FES; name = [PDE.variables[j][1] for j in 1 : length(FES)])
end

"""
````
function solve(
    PDE::PDEDescription,
    FES;
    kwargs...)
````

Returns a solution of the PDE as an FEVector for the provided FESpace(s) FES (to be used to discretised the unknowns of the PDEs).
To provide nonzero initial values (for nonlinear problems) the solve! function must be used.

This function extends the CommonSolve.solve interface and the PDEDEscription takes the role of
the ProblemType and FES takes the role of the SolverType.

Keyword arguments:
$(_myprint(default_solver_kwargs(),false))

Depending on the subiterations and detected/configured nonlinearities the whole system is
either solved directly in one step or via a fixed-point iteration.

"""
function CommonSolve.solve(PDE::PDEDescription, FES; kwargs...)
    Solution = FEVector(FES, PDE)
    CommonSolve.solve!(Solution, PDE; kwargs...)
    return Solution
end

"""
````
function solve!(
    Target::FEVector,       # contains initial guess and final solution after solve
    PDE::PDEDescription;
    kwargs)
````

Solves a given PDE (provided as a PDEDescription) and writes the solution into the FEVector Target (which knows the discrete ansatz spaces).

Keyword arguments:
$(_myprint(default_solver_kwargs(),false))

Depending on the subiterations and detected/configured nonlinearities the whole system is either solved directly in one step or via a fixed-point iteration.

"""
function CommonSolve.solve!(
    Target::FEVector{T,Tv,Ti},    # contains initial guess and final solution after solve
    PDE::PDEDescription;
    kwargs...) where {T, Tv, Ti}

    ## generate solver configurations
    SC = SolverConfig{T,Tv,Ti}(PDE; kwargs...)
    user_params = SC.user_params

    ## logging stuff
    if SC.is_timedependent
        moreinfo_string = "========== Solving $(PDE.name) (at fixed time $(user_params[:time])) =========="
    else
        moreinfo_string = "========== Solving $(PDE.name) =========="
    end
    nsolvedofs = 0
    subiterations = user_params[:subiterations]
    for s = 1 : length(subiterations), o = 1 : length(subiterations[s])
        d = subiterations[s][o]
        moreinfo_string *= "\n\tEquation ($s.$d) $(PDE.equation_names[d]) for $(PDE.unknown_names[d]) (discretised by $(Target[d].FES.name), ndofs = $(Target[d].FES.ndofs))"
        nsolvedofs += Target[d].FES.ndofs
    end
    @info moreinfo_string
    if user_params[:show_solver_config]
        @show SC
    else
        @debug SC
    end

    ## choose solver strategy
    if user_params[:subiterations] == [1:size(PDE.LHSOperators,1)] # full problem is solved
        if SC.is_nonlinear == false
            residual = solve_direct!(Target, PDE, SC; time = user_params[:time])
        else
            residual = solve_fixpoint_full!(Target, PDE, SC; time = user_params[:time])
        end
    else # only part(s) of the problem is solved
        residual = solve_fixpoint_subiterations!(Target, PDE, SC; time = user_params[:time])
    end

    ## report and check final residual
    if !user_params[:show_iteration_details]
        @info "overall residual = $residual"
    end
    if residual > user_params[:target_residual]
        @warn "residual was larger than desired target_residual = $(user_params[:target_residual])!"
    end
    return residual
end
