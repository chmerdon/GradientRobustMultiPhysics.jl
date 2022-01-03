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

abstract type AbstractLinearSystem{Tv,Ti} end

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
    LHS_LoopAllocations::Array{Array{Int,1},2}
    RHS_LoopAllocations::Array{Array{Int,1},1}
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
    :damping => (0, "damp the new iteration with this part of the old iteration (0 = undamped), also a function is allowed with the interface (old_iterate, new_iterate, fixed_dofs) that returns the new damping value",true,false),
    :anderson_iterations => (0, "use Anderson acceleration with this many previous iterates (to hopefully speed up/enable convergence of fixpoint iterations)",true,false),
    :anderson_metric => ("l2", "String that encodes the desired convergence metric for the Anderson acceleration (possible values: ''l2'' or ''L2'' or ''H1'')",true,false),
    :anderson_damping => (1, "Damping factor in Anderson acceleration (1 = undamped)",true,false),
    :anderson_unknowns => ([1], "an array of unknown numbers that should be included in the Anderson acceleration",true,false),
    :fixed_penalty => (1e60, "penalty that is used for the values of fixed degrees of freedom (e.g. by Dirichlet boundary data or global constraints)",true,true),
    :linsolver => ("UMFPACK", "String that encodes the linear solver, or type name of self-defined solver (see corressponding example), or type name of ExtendableSparse.AbstractFactorization",true,true),
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
        user_params[Symbol(k)]=v
    end
    return nothing
end


struct LinearSystem{Tv,Ti,FT <: ExtendableSparse.AbstractFactorization{Tv,Ti}} <: AbstractLinearSystem{Tv,Ti} 
    x::AbstractVector{Tv}
    A::ExtendableSparseMatrix{Tv,Ti}
    b::AbstractVector{Tv}
    factorization::FT
    LinearSystem{Tv,Ti,FT}(x,A,b) where {Tv,Ti,FT} = new{Tv,Ti,FT}(x,A,b,FT(A,nothing,0))
end

function createlinsolver(ST::Type{<:AbstractLinearSystem},x::AbstractVector,A::ExtendableSparseMatrix,b::AbstractVector)
    return ST(x,A,b)
end

function update_factorization!(LS::AbstractLinearSystem)
    # do nothing for an abstract solver
end

function update_factorization!(LS::LinearSystem)
    @logmsg MoreInfo "Updating factorization = $(typeof(LS).parameters[3])..."
    ExtendableSparse.update!(LS.factorization)
end

function solve!(LS::LinearSystem) 
    @logmsg MoreInfo "Solving with factorization = $(typeof(LS).parameters[3])..."
    ldiv!(LS.x,LS.factorization,LS.b)
end

function set_nonzero_pattern!(A::FEMatrix, AT::Type{<:AssemblyType} = ON_CELLS)
    @debug "Setting nonzero pattern for FEMatrix..."
    for block = 1 : length(A)
        apply_nonzero_pattern!(A[block],AT)
    end
end

# check if PDE is nonlinear or time-dependent and which blocks require recalculation
# and devise some initial solver strategy
function SolverConfig{T}(PDE::PDEDescription, ::ExtendableGrid{TvG,TiG}, user_params) where {T,TvG,TiG}

    ## declare subiterations
    if user_params[:subiterations] == "auto"
        user_params[:subiterations] = [1:size(PDE.LHSOperators,1)]
    end
    ## declare linear solver
    if user_params[:linsolver] == "UMFPACK"
        user_params[:linsolver] = LinearSystem{T, Int64, ExtendableSparse.LUFactorization{T,Int64}}
    elseif user_params[:linsolver] == "MKLPARDISO"
        user_params[:linsolver] = LinearSystem{T, Int64, ExtendableSparse.MKLPardisoLU{T,Int64}}
    elseif user_params[:linsolver] <: ExtendableSparse.AbstractFactorization{T,Int64}
        user_params[:linsolver] = LinearSystem{T, Int64, user_params[:linsolver]}
    elseif user_params[:linsolver] <: ExtendableSparse.AbstractFactorization
        user_params[:linsolver] = LinearSystem{T, Int64, user_params[:linsolver]{T,Int64}}
    end
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
    LHS_LoopAllocations = Array{Array{Int,1},2}(undef,size(PDE.LHSOperators,1),size(PDE.LHSOperators,2))
    RHS_LoopAllocations = Array{Array{Int,1},1}(undef,size(PDE.RHSOperators,1))
    for j = 1 : size(PDE.LHSOperators,1), k = 1 : size(PDE.LHSOperators,2)
        LHS_APs[j,k] = Array{AssemblyPattern,1}(undef, length(PDE.LHSOperators[j,k]))
        LHS_AssemblyTimes[j,k] = zeros(Float64, length(PDE.LHSOperators[j,k]))
        LHS_LoopAllocations[j,k] = zeros(Int, length(PDE.LHSOperators[j,k]))
        for o = 1 : length(PDE.LHSOperators[j,k])
            LHS_APs[j,k][o] = AssemblyPattern()
        end
    end
    for j = 1 : size(PDE.RHSOperators,1)
        RHS_APs[j] = Array{AssemblyPattern,1}(undef, length(PDE.RHSOperators[j]))
        RHS_AssemblyTimes[j] = zeros(Float64, length(PDE.RHSOperators[j]))
        RHS_LoopAllocations[j] = zeros(Int, length(PDE.RHSOperators[j]))
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

    return SolverConfig{T,TvG,TiG}(nonlinear, timedependent, LHS_ATs, LHS_dep, RHS_ATs, RHS_dep, LHS_APs, RHS_APs, LHS_AssemblyTimes, RHS_AssemblyTimes, LHS_LoopAllocations, RHS_LoopAllocations, user_params)
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

    info_msg = "ACCUMULATED ASSEMBLY TIMES AND ALLOCATIONS"
    info_msg *= "\n\t=========================="

    for s = 1 : length(subiterations)
        for j = 1 : length(subiterations[s])
            eq = subiterations[s][j]
            for k = 1 : size(SC.LHS_AssemblyTimes,2)
                for o = 1 : size(SC.LHS_AssemblyTimes[eq,k],1)
                    info_msg *= "\n\tLHS[$eq,$k][$o] ($(PDE.LHSOperators[eq,k][o].name)) = $(SC.LHS_AssemblyTimes[eq,k][o])s"
                    info_msg *= "\t ( $(SC.LHS_LoopAllocations[eq,k][o]) allocations)"
                end
            end
        end
    end

    for s = 1 : length(subiterations)
        for j = 1 : length(subiterations[s])
            eq = subiterations[s][j]
            for o = 1 : size(SC.RHS_AssemblyTimes[eq],1)
                info_msg *= "\n\tRHS[$eq][$o] ($(PDE.RHSOperators[eq][o].name)) = $(SC.RHS_AssemblyTimes[eq][o])s"
                info_msg *= "\t ( $(SC.RHS_LoopAllocations[eq][o]) allocations)"
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

    elapsedtime::Float64 = 0
    nequations::Int = length(equations)
    # force (re)assembly of stored bilinearforms and RhsOperators
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
                                elapsedtime = @elapsed SC.LHS_LoopAllocations[equations[j],k][o] += update_storage!(O, CurrentSolution, equations[j], k; time = time)
                                SC.LHS_AssemblyTimes[equations[j],k][o] += elapsedtime
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
                            elapsedtime = @elapsed SC.RHS_LoopAllocations[equations[j]][o] += update_storage!(O, CurrentSolution, equations[j] ; time = time)
                            SC.RHS_AssemblyTimes[equations[j]][o] += elapsedtime
                        end
                    end
                end
            end
        end
    end

    # (re)assembly right-hand side
    rhs_block_has_been_erased = zeros(Bool,nequations)
    if !only_lhs || only_rhs
        for j = 1 : nequations
            if (min_trigger <: SC.RHS_AssemblyTriggers[equations[j]]) && (length(RHSOperators[equations[j]]) > 0)
                @debug "Erasing rhs block [$j]"
                fill!(b[j],0.0)
                rhs_block_has_been_erased[j] = true
                @logmsg DeepInfo "Locking what to assemble in rhs block [$(equations[j])]..."
                for o = 1 : length(RHSOperators[equations[j]])
                    O = RHSOperators[equations[j]][o]
                    elapsedtime = @elapsed assemble!(b[j], SC, equations[j], o, O, CurrentSolution; time = time)
                    SC.RHS_AssemblyTimes[equations[j]][o] += elapsedtime
                    if !has_storage(O)
                        SC.RHS_LoopAllocations[equations[j]][o] += SC.RHS_AssemblyPatterns[equations[j]][o].last_allocations
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
                            if has_copy_block(O)
                                elapsedtime = @elapsed assemble!(A[j,subblock], SC, equations[j],k,o, O, CurrentSolution; time = time, At = A[subblock,j])
                            else
                                elapsedtime = @elapsed assemble!(A[j,subblock], SC, equations[j],k,o, O, CurrentSolution; time = time)
                            end  
                            SC.LHS_AssemblyTimes[equations[j],k][o] += elapsedtime
                            if !has_storage(O)
                                SC.LHS_LoopAllocations[equations[j],k][o] += SC.LHS_AssemblyPatterns[equations[j],k][o].last_allocations
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
                            elapsedtime = @elapsed assemble!(b[j], SC, equations[j],k,o, O, CurrentSolution; factor = -1.0, time = time, fixed_component = k)
                            SC.LHS_AssemblyTimes[equations[j],k][o] += elapsedtime
                            if !has_storage(O)
                                SC.LHS_LoopAllocations[equations[j],k][o] += SC.LHS_AssemblyPatterns[equations[j],k][o].last_allocations
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
function solve_direct!(Target::FEVector{T,Tv,Ti}, PDE::PDEDescription, SC::SolverConfig{T,Tv,Ti}; time::Real = 0, show_details::Bool = false) where {T, Tv, Ti}

    FEs = Array{FESpace{Tv,Ti},1}([])
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
    fixed_dofs::Array{Ti,1} = Base.unique(fixed_dofs)
    fixed_penalty::T = SC.user_params[:fixed_penalty]
    for j = 1 : length(fixed_dofs)
        b.entries[fixed_dofs[j]] = fixed_penalty * Target.entries[fixed_dofs[j]]
        A[1][fixed_dofs[j],fixed_dofs[j]] = fixed_penalty
    end

    # SOLVE
    LS = createlinsolver(SC.user_params[:linsolver], Target.entries, A.entries, b.entries)
    flush!(A.entries)
    update_factorization!(LS)
    solve!(LS)

   # @time Target.entries .= A.entries\b.entries

    residuals::Array{T,1} = zeros(T, length(Target.FEVectorBlocks))
    # CHECK RESIDUAL
    residual::Array{T,1} = A.entries*Target.entries - b.entries
    residual[fixed_dofs] .= 0
    for j = 1 : length(Target.FEVectorBlocks)
        for k = 1 : Target.FEVectorBlocks[j].FES.ndofs
            residuals[j] += residual[k + Target.FEVectorBlocks[j].offset].^2
        end
    end
    resnorm::T = sum(residuals)
    residuals = sqrt.(residuals)

    # REALIZE GLOBAL GLOBALCONSTRAINTS 
    # (possibly changes some entries of Target)
    for j = 1 : length(PDE.GlobalConstraints)
        realize_constraint!(Target,PDE.GlobalConstraints[j])
    end

    if SC.user_params[:show_statistics]
        show_statistics(PDE,SC)
    end

    if SC.user_params[:show_iteration_details]
        @info "overall residual = $(sqrt(resnorm))"
    end
    return sqrt(resnorm)
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
    NormOperator = FEMatrix{T}("AA-Metric",FEs)
    for u in anderson_unknowns
        if anderson_metric == "L2"
            AA_METRIC = SymmetricBilinearForm(Float64, ON_CELLS, [FEs[u], FEs[u]], [Identity, Identity], NoAction())
            assemble!(NormOperator[u,u], AA_METRIC)
        elseif anderson_metric == "H1"
                AA_METRIC = SymmetricBilinearForm(Float64, ON_CELLS, [FEs[u], FEs[u]], [Gradient,Gradient], NoAction())
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

    residual = zeros(T,length(b.entries))
    linresnorm::T = 0.0
    resnorm::T = 0.0

    ## INIT SOLVER
    LS = createlinsolver(SC.user_params[:linsolver],Target.entries,A.entries,b.entries)

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
                update_factorization!(LS)
            end
            solve!(LS)
        end

        # CHECK LINEAR RESIDUAL
        if SC.user_params[:show_iteration_details]
            residual = A.entries*Target.entries - b.entries
            residual[fixed_dofs] .= 0
            linresnorm = (sqrt(sum(residual.^2, dims = 1)[1]))
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
            for j = 1 : length(Target.entries)
                Target.entries[j] = damping_val * LastIterate.entries[j] + (1-damping_val) * Target.entries[j]
            end
        end
        if damping_val > 0 || typeof(damping) <: Function
            LastIterate.entries .= Target.entries
        end


        # REASSEMBLE NONLINEAR PARTS
        time_reassembly = @elapsed assemble!(A,b,PDE,SC,Target; time = time, equations = 1:size(PDE.RHSOperators,1), min_trigger = AssemblyAlways)

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
function solve_fixpoint_subiterations!(Target::FEVector{T,Tv,Ti}, PDE::PDEDescription, SC::SolverConfig{T,Tv,Ti}; time = 0) where {T,Tv,Ti}

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

            # FILL x[s] WITH TARGET ENTRIES
            for k = 1 : length(subiterations[i])
                d = subiterations[i][k]
                for dof = 1 : Target[d].FES.ndofs
                    x[i][k][dof] = Target[d][dof]
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
        residual[s] = FEVector{T}("residual subiteration $s", FEs[subiterations[s]])
    end
    linresnorm = zeros(T,nsubiterations)
    resnorm = zeros(T,nsubiterations)


    ## INIT SOLVERS
    LS = Array{AbstractLinearSystem{T,Int64},1}(undef,nsubiterations)
    for s = 1 : nsubiterations
        LS[s] = createlinsolver(SC.user_params[:linsolver],x[s].entries,A[s].entries,b[s].entries)
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
                    update_factorization!(LS[s])
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
            for j = 1 : length(subiterations[s])
                for k = 1 : length(Target[subiterations[s][j]])
                    Target[subiterations[s][j]][k] = x[s][j][k]
                end
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
            if SC.user_params[:show_iteration_details]
                @info "target residual reached after $iteration iterations"
            end
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
    Target::FEVector,       # contains initial guess and final solution after solve
    PDE::PDEDescription;
    kwargs)
````

Solves a given PDE (provided as a PDEDescription) and writes the solution into the FEVector Target (which knows the discrete ansatz spaces).

Keyword arguments:
$(_myprint(default_solver_kwargs(),false))

Depending on the subiterations and detected/configured nonlinearities the whole system is either solved directly in one step or via a fixed-point iteration.

"""
function solve!(
    Target::FEVector{T,Tv,Ti},    # contains initial guess and final solution after solve
    PDE::PDEDescription;
    kwargs...) where {T, Tv, Ti}

    ## generate solver configurations
    xgrid = Target[1].FES.xgrid
    user_params=Dict{Symbol,Any}( k => v[1] for (k,v) in default_solver_kwargs())
    _update_solver_params!(user_params,kwargs)
    SC = SolverConfig{T}(PDE, xgrid, user_params)

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
        @info "overall residual = $residual"
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
abstract type CrankNicolson <: AbstractTimeIntegrationRule end

mutable struct TimeControlSolver{T,Tt,TiM,Tv,Ti,TIR<:AbstractTimeIntegrationRule}
    PDE::PDEDescription                     # PDE description (operators, data etc.)
    SC::SolverConfig{T,Tv,Ti}               # solver configurations (subiterations, penalties etc.)
    LS::Array{AbstractLinearSystem{T,TiM},1}  # array for linear solvers of all subiterations
    ctime::Tt                               # current time
    cstep::Int                              # current timestep count
    last_timestep::Tt                       # last timestep
    AM::Array{FEMatrix{T,TiM,Tv,Ti},1}          # heap for mass matrices for each equation package
    which::Array{Int,1}                     # which equations shall have a time derivative?
    dt_is_nonlinear::Array{Bool,1}             # if true mass matrix is recomputed in each iteration
    dt_operator::Array{DataType,1}         # operators associated with the time derivative
    dt_actions::Array{<:AbstractAction,1}   # actions associated with the time derivative
    S::Array{FEMatrix{T,TiM,Tv,Ti},1}           # heap for system matrix for each equation package
    rhs::Array{FEVector{T,Tv,Ti},1}         # heap for system rhs for each equation package
    A::Array{FEMatrix{T,TiM,Tv,Ti},1}           # heap for spacial discretisation matrix for each equation package
    b::Array{FEVector{T,Tv,Ti},1}           # heap for spacial rhs for each equation package
    x::Array{FEVector{T,Tv,Ti},1}           # heap for current solution for each equation package
    res::Array{FEVector{T,Tv,Ti},1}         # residual vector
    X::FEVector{T,Tv,Ti}                    # full solution vector
    LastIterate::FEVector{T,Tv,Ti}          # helper variable if maxiterations > 1
    fixed_dofs::Array{Ti,1}                 # fixed dof numbes (values are stored in X)
    eqoffsets::Array{Array{Int,1},1}        # offsets for subblocks of each equation package
    ALU::Array{Any,1}                       # LU decompositions of matrices A
    statistics::Array{T,2}                  # statistics of last timestep
end



function assemble_massmatrix4subiteration!(TCS::TimeControlSolver{T,Tt,Tv,Ti}, i::Int; force::Bool = false) where {T,Tt,Tv,Ti}
    subiterations = TCS.SC.user_params[:subiterations]
    for j = 1 : length(subiterations[i])
        if (subiterations[i][j] in TCS.which) == true 

            # assembly mass matrix into AM block
            pos = findall(x->x==subiterations[i][j], TCS.which)[1]

            if TCS.dt_is_nonlinear[pos] == true || force == true
                @logmsg DeepInfo "Assembling mass matrix for block [$j,$j] of subiteration $i with action of type $(typeof(TCS.dt_actions[pos]))"
                A = TCS.AM[i][j,j]
                FE1 = A.FESX
                FE2 = A.FESY
                operator1 = TCS.dt_operator[pos]
                operator2 = TCS.dt_operator[pos]
                BLF = SymmetricBilinearForm(T, ON_CELLS, [FE1, FE2], [operator1, operator2], TCS.dt_actions[pos])    
                time = @elapsed assemble!(A, BLF, skip_preps = false)
            end
        end
    end
    flush!(TCS.AM[i].entries)
end


"""
````
function TimeControlSolver(
    PDE::PDEDescription,                                        # PDE system description
    InitialValues::FEVector{T,Tv,Ti},                           # contains initial values and stores solution of advance! methods
    TIR::Type{<:AbstractTimeIntegrationRule} = BackwardEuler;   # Time integration rule
    dt_operator = [],                                           # Operator in time derivative (default: Identity) for each subiteration (applied to test and ansatz function)
    dt_action = [],                                             # Action in time derivative (deulta: NoAction) for each subiteration
    dt_is_nonlinear = [],                                          # is the time derivative nonlinear?
    T_time = Float64,                                           # Type for timestep and total time
    kwargs...) where {T,Tv,Ti}                                  # additional solver arguments
````

Creates a time-dependent solver that can be advanced in time with advance!.
The FEVector Solution stores the initial state but also the solution at the current time.
The argument TIR carries the time integration rule to be use (e.g. BackwardEuler or CrankNicolson).
T_time determines the NumberType of the timesteps and total time.

Keyword arguments:
$(_myprint(default_solver_kwargs(),true))

Further (very experimental) optional arguments for TimeControlSolver are:
- dt_operator : (array of) operators applied to testfunctions in time derivative (default: Identity)
- dt_action : (array of) actions that are applied to the ansatz function in the time derivative (to include parameters etc.)
- dt_is_nonlinear : (array of) booleans to decide which time derivatives should be recomputed in each timestep

"""
function TimeControlSolver(
    PDE::PDEDescription,
    InitialValues::FEVector{T,Tv,Ti},    # contains initial values and stores solution of advance! methods
    TIR::Type{<:AbstractTimeIntegrationRule} = BackwardEuler;
    dt_operator = [],
    dt_action = [],
    dt_is_nonlinear = [],
    T_time = Float64,
    kwargs...) where {T,Tv,Ti}

    ## generate solver configurations
    user_params=Dict{Symbol,Any}( k => v[1] for (k,v) in default_solver_kwargs())
    _update_solver_params!(user_params,kwargs)
    xgrid = InitialValues[1].FES.xgrid
    SC = SolverConfig{T}(PDE, xgrid, user_params)
    start_time = user_params[:time]

    # generate solver for time-independent problem
    for j = 1 : length(InitialValues.FEVectorBlocks)
        if (SC.RHS_AssemblyTriggers[j] <: AssemblyEachTimeStep)
            SC.RHS_AssemblyTriggers[j] = AssemblyEachTimeStep
        end
    end

    ## logging stuff
    moreinfo_string = "----- Preparing time control solver for $(PDE.name) using $(TIR) -----"
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
    FEs = Array{FESpace{Tv,Ti},1}(undef,length(InitialValues.FEVectorBlocks))
    for j = 1 : length(InitialValues.FEVectorBlocks)
        FEs[j] = InitialValues.FEVectorBlocks[j].FES
    end    
    eqoffsets = Array{Array{Int,1},1}(undef,nsubiterations)
    AM = Array{FEMatrix{T},1}(undef,nsubiterations)
    A = Array{FEMatrix{T},1}(undef,nsubiterations)
    b = Array{FEVector{T},1}(undef,nsubiterations)
    x = Array{FEVector{T},1}(undef,nsubiterations)
    res = Array{FEVector{T},1}(undef,nsubiterations)
    for i = 1 : nsubiterations
        A[i] = FEMatrix{T}("SystemMatrix subiteration $i", FEs[subiterations[i]])
        AM[i] = FEMatrix{T}("MassMatrix subiteration $i", FEs[subiterations[i]])
        b[i] = FEVector{T}("SystemRhs subiteration $i", FEs[subiterations[i]])
        x[i] = FEVector{T}("Solution subiteration $i", FEs[subiterations[i]])
        res[i] = FEVector{T}("Residual subiteration $i", FEs[subiterations[i]])
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
    fixed_dofs::Array{Ti,1} = zeros(Ti,0)
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
                if length(dt_is_nonlinear) < pos
                    push!(dt_is_nonlinear, false)
                end
                if length(dt_action) < pos
                    push!(dt_action, NoAction())
                end
                if length(dt_operator) < pos
                    push!(dt_operator, Identity)
                end
                # set diagonal equations block and rhs block to nonlinear
                if dt_is_nonlinear[pos] == true
                    SC.LHS_AssemblyTriggers[d,d] = AssemblyEachTimeStep
                    SC.RHS_AssemblyTriggers[d] = AssemblyEachTimeStep
                end
            end
        end
    end

    dt_action = Array{AbstractAction,1}(dt_action)
    dt_operator = Array{DataType,1}(dt_operator)

    if TIR == BackwardEuler
        S = A
        rhs = b
    elseif TIR == CrankNicolson
        S = deepcopy(A)
        rhs = deepcopy(b)
    end

    # INIT LINEAR SOLVERS
    LS = Array{AbstractLinearSystem{T,Int64},1}(undef,nsubiterations)
    for s = 1 : nsubiterations
        LS[s] = createlinsolver(SC.user_params[:linsolver],x[s].entries,S[s].entries,rhs[s].entries)
    end

    # storage for last iterate (to compute change properly)
    LastIterate = deepcopy(InitialValues)

    # generate TimeControlSolver
    statistics = zeros(Float64,length(InitialValues),4)
    TCS = TimeControlSolver{T,T_time,Int64,Tv,Ti,TIR}(PDE,SC,LS,start_time,0,0,AM,timedependent_equations,dt_is_nonlinear,dt_operator,dt_action,S,rhs,A,b,x,res,InitialValues, LastIterate, fixed_dofs, eqoffsets, Array{SuiteSparse.UMFPACK.UmfpackLU{T,Int64},1}(undef,length(subiterations)), statistics)

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
function advance!(TCS::TimeControlSolver{T,Tt,TiM,Tv,Ti,TIR}, timestep::Real = 1e-1) where {T,Tt,TiM,Tv,Ti,TIR}
    # update timestep counter
    TCS.cstep += 1
    TCS.ctime += timestep

    # unpack
    SC = TCS.SC
    PDE = TCS.PDE
    S = TCS.S
    rhs = TCS.rhs
    A = TCS.A 
    AM = TCS.AM
    b = TCS.b
    x = TCS.x
    res = TCS.res
    X = TCS.X
    LS = TCS.LS
    fixed_dofs::Array{Ti,1} = TCS.fixed_dofs
    eqoffsets::Array{Array{Int,1},1} = TCS.eqoffsets
    statistics::Array{Float64,2} = TCS.statistics
    fill!(statistics,0)

    ## get relevant solver parameters
    subiterations::Array{Array{Int,1},1} = SC.user_params[:subiterations]
    #anderson_iterations = SC.user_params[:anderson_iterations]
    fixed_penalty::T = SC.user_params[:fixed_penalty]
    #damping = SC.user_params[:damping]
    maxiterations::Int = SC.user_params[:maxiterations]
    check_nonlinear_residual::Bool = SC.user_params[:check_nonlinear_residual]
    #timedependent_equations = SC.user_params[:timedependent_equations]
    target_residual::T = SC.user_params[:target_residual]
    skip_update::Array{Int,1} = SC.user_params[:skip_update]

    # save current solution to LastIterate
    LastIterate = TCS.LastIterate
    LastIterate.entries .= X.entries

    ## LOOP OVER ALL SUBITERATIONS
    resnorm::T = 1e30
    eqdof::Int = 0
    d::Int = 0
    nsubitblocks::Int = 0
    update_matrix::Bool = true
    factors::Array{Int,1} = ones(Int,length(X))

    for s = 1 : length(subiterations)
        nsubitblocks = length(subiterations[s])

        # decide if matrix needs to be updated
        update_matrix = skip_update[s] != -1 || TCS.cstep == 1 || TCS.last_timestep != timestep

        # UPDATE SYSTEM (BACKWARD EULER)
        if TIR == BackwardEuler # here: S = A, rhs = b
            if update_matrix # update matrix
                fill!(A[s].entries.cscmatrix.nzval,0)
                fill!(b[s].entries,0)
                assemble!(A[s],b[s],PDE,SC,X; equations = subiterations[s], time = TCS.ctime, min_trigger = AssemblyInitial, storage_trigger = AssemblyEachTimeStep)

                ## update mass matrix and add time derivative
                assemble_massmatrix4subiteration!(TCS, s; force = false)
                for k = 1 : nsubitblocks
                    d = subiterations[s][k]
                    addblock!(A[s][k,k],AM[s][k,k]; factor = 1.0/timestep)
                    addblock_matmul!(b[s][k],AM[s][k,k],LastIterate[d]; factor = 1.0/timestep)
                end
                flush!(A[s].entries)
            else # only update rhs
                fill!(b[s].entries,0)
                assemble!(A[s],b[s],PDE,SC,X; equations = subiterations[s], time = TCS.ctime, min_trigger = AssemblyInitial, storage_trigger = AssemblyEachTimeStep, only_rhs = true)

                ## add time derivative
                for k = 1 : nsubitblocks
                    d = subiterations[s][k]
                    addblock_matmul!(b[s][k],AM[s][k,k],LastIterate[d]; factor = 1.0/timestep)
                end
            end
        elseif TIR == CrankNicolson # S and A are separate, A[s] contains spacial discretisation of last timestep

            ## reset rhs and add terms for old solution F^n - A^n u^n
            ## last iterates of algebraic constraints are ignored
            fill!(rhs[s].entries,0)
            rhs[s].entries .+= b[s].entries
            for j = 1 : nsubitblocks
                d = subiterations[s][j]
                if PDE.algebraic_constraint[d] == false
                    for k = 1 : nsubitblocks
                        if PDE.algebraic_constraint[subiterations[s][k]] == false
                            addblock_matmul!(rhs[s][k],A[s][k,j],LastIterate[d]; factor = -1)
                        end
                    end
                else
                    # old iterates of algebraic constraints are not used, instead the weight of the coressponding equation
                    # is adjusted (e.g. the pressure in Navier-Stokes equations)
                    factors[d] = 2
                end
            end

            if update_matrix # update matrix S[s] and right-hand side rhs[s]

                ## reset system matrix
                fill!(S[s].entries.cscmatrix.nzval,0)

                ## assembly new right-hand side and matrix and add to system
                fill!(b[s].entries,0)
                fill!(A[s].entries.cscmatrix.nzval,0)
                assemble!(A[s],b[s],PDE,SC,X; equations = subiterations[s], time = TCS.ctime, min_trigger = AssemblyInitial, storage_trigger = AssemblyEachTimeStep)
                rhs[s].entries .+= b[s].entries
                add!(S[s],A[s])
                flush!(S[s].entries)
                
                ## update and add time derivative to system matrix and right-hand side
                assemble_massmatrix4subiteration!(TCS, s; force = false)
                for k = 1 : nsubitblocks
                    d = subiterations[s][k]
                    addblock!(S[s][k,k],AM[s][k,k]; factor = 2.0/timestep)
                    addblock_matmul!(rhs[s][k],AM[s][k,k],LastIterate[d]; factor = 2.0/timestep)
                end
            else # S[s] stays the same, only update rhs[s]

                ## assembly of new right-hand side 
                fill!(b[s].entries,0)
                assemble!(A[s],b[s],PDE,SC,X; equations = subiterations[s], time = TCS.ctime, min_trigger = AssemblyInitial, storage_trigger = AssemblyEachTimeStep, only_rhs = true)
                rhs[s].entries .+= b[s].entries
                
                ## update and add time derivative to system matrix and right-hand side
                assemble_massmatrix4subiteration!(TCS, s; force = false)
                for k = 1 : nsubitblocks
                    d = subiterations[s][k]
                    addblock_matmul!(rhs[s][k],AM[s][k,k],LastIterate[d]; factor = 2.0/timestep)
                end
            end

        end

        # ASSEMBLE (TIME-DEPENDENT) BOUNDARY DATA
        for k = 1 : nsubitblocks
            d = subiterations[s][k]
            if any(PDE.BoundaryOperators[d].timedependent) == true
                boundarydata!(x[s][k],PDE.BoundaryOperators[d]; time = TCS.ctime, skip_enumerations = true)
            end
        end    

        ## START (NONLINEAR) ITERATION(S)
        for iteration = 1 : maxiterations
            statistics[s,4] = iteration.^2 # will be square-rooted later

            # PENALIZE FIXED DOFS (IN CASE THE MATRIX CHANGED)
            # (from boundary conditions and global constraints)
            for dof in fixed_dofs
                for eq = 1 : nsubitblocks
                    if dof > eqoffsets[s][eq] && dof <= eqoffsets[s][eq]+X[subiterations[s][eq]].FES.ndofs
                        eqdof = dof - eqoffsets[s][eq]
                        rhs[s][eq][eqdof] = fixed_penalty * x[s][eq][eqdof]
                        S[s][eq,eq][eqdof,eqdof] = fixed_penalty
                    end
                end
            end

            ## SOLVE for x[s]
            time_solver = @elapsed begin
                flush!(S[s].entries)
                if update_matrix || (TCS.cstep % skip_update[s] == 0 && skip_update[s] != -1)
                    update_factorization!(LS[s])
                end
                solve!(LS[s])
            end

            ## CHECK LINEAR RESIDUAL
            mul!(res[s].entries,S[s].entries,x[s].entries)
            res[s].entries .-= rhs[s].entries
            for dof in fixed_dofs
                for eq = 1 : nsubitblocks
                    if dof > eqoffsets[s][eq] && dof <= eqoffsets[s][eq]+X[subiterations[s][eq]].FES.ndofs
                        eqdof = dof - eqoffsets[s][eq]
                        res[s][eq][eqdof] = 0
                    end
                end
            end
            for j = 1 : nsubitblocks
                statistics[subiterations[s][j],1] = 0
                for k = res[s][j].offset+1:res[s][j].last_index
                    statistics[subiterations[s][j],1] += res[s][j].entries[k].^2
                end
            end

            # WRITE x[s] INTO X
            for j = 1 : nsubitblocks
                d = subiterations[s][j]
                for k = 1 : length(LastIterate[subiterations[s][j]])
                    X[d][k] = x[s][j][k] / factors[d]
                end
            end

            ## REASSEMBLE NONLINEAR PARTS
            if maxiterations > 1 || check_nonlinear_residual

                if TIR == BackwardEuler
                    # update matrix A
                    lhs_erased, rhs_erased = assemble!(A[s],b[s],PDE,SC,X; equations = subiterations[s], min_trigger = AssemblyAlways, time = TCS.ctime)

                    ## S = A, so we need to readd the time-derivative for reassembled blocks
                    for k = 1 : nsubitblocks
                        d = subiterations[s][k]
                        if lhs_erased[k,k]
                            addblock!(A[s][k,k],AM[s][k,k]; factor = 1.0/timestep)
                        end
                        if rhs_erased[k]
                            addblock_matmul!(b[s][k],AM[s][k,k],LastIterate[d]; factor = 1.0/timestep)
                        end
                    end
                elseif TIR == CrankNicolson
                    # subtract current matrix A
                    add!(S[s],A[s]; factor = -1)
                    rhs[s].entries .-= b[s].entries

                    # update matrix A
                    lhs_erased, rhs_erased = assemble!(A[s],b[s],PDE,SC,X; equations = subiterations[s], min_trigger = AssemblyAlways, time = TCS.ctime)

                    # add new matrix A
                    add!(S[s],A[s])
                    rhs[s].entries .+= b[s].entries
                end

                # CHECK NONLINEAR RESIDUAL
                if sum(lhs_erased[:]) + sum(rhs_erased) > 0
                    mul!(res[s].entries,S[s].entries,x[s].entries)
                    res[s].entries .-= rhs[s].entries
                    for dof in fixed_dofs
                        for eq = 1 : nsubitblocks
                            if dof > eqoffsets[s][eq] && dof <= eqoffsets[s][eq]+X[subiterations[s][eq]].FES.ndofs
                                eqdof = dof - eqoffsets[s][eq]
                                res[s][eq][eqdof] = 0
                            end
                        end
                    end
                    for j = 1 : nsubitblocks
                        statistics[subiterations[s][j],3] = 0
                        for k = res[s][j].offset+1:res[s][j].last_index
                            statistics[subiterations[s][j],3] += res[s][j].entries[k].^2
                        end
                    end
                    resnorm = norm(res[s].entries)
                else
                    for j = 1 : nsubitblocks
                        statistics[subiterations[s][j],3] = statistics[subiterations[s][j],1]
                    end
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
    for j = 1 : length(PDE.GlobalConstraints)
        realize_constraint!(X,PDE.GlobalConstraints[j])
    end

    # COMPUTE CHANGE
    for j = 1 : length(X)
        for k = 1 : length(X[j])
            statistics[j,2] += (LastIterate[j][k] - X[j][k])^2
        end
    end

    # remember last timestep
    TCS.last_timestep = timestep

    statistics .= sqrt.(statistics)
    return nothing
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
function advance_until_stationarity!(TCS::TimeControlSolver{T,Tt}, timestep::Tt; stationarity_threshold = 1e-11, maxTimeSteps = 100, do_after_each_timestep = nothing) where {T,Tt}
    statistics = TCS.statistics
    maxiterations = TCS.SC.user_params[:maxiterations]
    show_details = TCS.SC.user_params[:show_iteration_details]
    check_nonlinear_residual = TCS.SC.user_params[:check_nonlinear_residual]
    @info "Advancing in time until stationarity..."
    steptime::Float64 = 0
    if show_details
        if maxiterations > 1 || check_nonlinear_residual
            @printf("\n\t  STEP  |    TIME    | LSRESIDUAL |   NLRESIDUAL   |  RUNTIME  |   CHANGE ")
            for j = 1 : size(statistics,1)
                @printf("       ")
            end
            if do_after_each_timestep !== nothing
                do_after_each_timestep(0, statistics)
            end
            @printf("\n\t        |            |  (total)   |  (total,nits)  |    (s)    |")
        else
            @printf("\n\t  STEP  |    TIME    | LSRESIDUAL |  RUNTIME  |   CHANGE ")
            for j = 1 : size(statistics,1)
                @printf("       ")
            end
            if do_after_each_timestep !== nothing
                do_after_each_timestep(0, statistics)
            end
            @printf("\n\t        |            |  (total)   |    (s)    |")
        end
        for j = 1 : size(statistics,1)
            @printf(" %s ",center_string(TCS.PDE.unknown_names[j],10))
        end
    end
    totaltime = @elapsed for iteration = 1 : maxTimeSteps
        steptime = @elapsed advance!(TCS, timestep)
        if show_details
            @printf("\n")
            @printf("\t  %4d  ",iteration)
            @printf("| %.4e ",TCS.ctime)
            @printf("| %.4e |",sqrt(sum(statistics[:,1].^2)))
            if maxiterations > 1 || check_nonlinear_residual
                @printf(" %.4e (%d) |",sqrt(sum(view(statistics,:,3).^2)), statistics[1,4])
            end
            @printf(" %.3e |",steptime)
            for j = 1 : size(statistics,1)
                @printf(" %.4e ",statistics[j,2])
            end
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
        @printf("\n")
    end

    if TCS.SC.user_params[:show_statistics]
        show_statistics(TCS.PDE,TCS.SC)
        @info "totaltime = $(totaltime)s"
    end

end


"""
````
advance_until_time!(TCS::TimeControlSolver, timestep, finaltime; finaltime_tolerance = 1e-15, do_after_each_timestep = nothing)
````

Advances a TimeControlSolver in time with the given (initial) timestep until the specified finaltime is reached (up to the specified tolerance).
The function do_after_timestep is called after each timestep and can be used to print/save data (and maybe timestep control in future).
"""
function advance_until_time!(TCS::TimeControlSolver{T,Tt}, timestep::Tt, finaltime; finaltime_tolerance = 1e-15, do_after_each_timestep = nothing) where {T,Tt}
    statistics = TCS.statistics
    maxiterations = TCS.SC.user_params[:maxiterations]
    show_details = TCS.SC.user_params[:show_iteration_details]
    show_statistics = TCS.SC.user_params[:show_statistics]
    check_nonlinear_residual = TCS.SC.user_params[:check_nonlinear_residual]
    @info "Advancing in time from $(TCS.ctime) until $finaltime"
    steptime::Float64 = 0
    if show_details
        if maxiterations > 1 || check_nonlinear_residual
            @printf("\n\t  STEP  |    TIME    | LSRESIDUAL |   NLRESIDUAL   |  RUNTIME  |   CHANGE ")
            for j = 1 : size(statistics,1)
                @printf("       ")
            end
            if do_after_each_timestep !== nothing
                do_after_each_timestep(0, statistics)
            end
            @printf("\n\t        |            |  (total)   |    (total)     |    (s)    |")
        else
            @printf("\n\t  STEP  |    TIME    | LSRESIDUAL |  RUNTIME  |   CHANGE ")
            for j = 1 : size(statistics,1)
                @printf("       ")
            end
            if do_after_each_timestep !== nothing
                do_after_each_timestep(0, statistics)
            end
            @printf("\n\t        |            |  (total)   |    (s)    ")
        end
        for j = 1 : size(statistics,1)
            @printf(" %s ",center_string(TCS.PDE.unknown_names[j],10))
        end
    end
    totaltime = @elapsed while TCS.ctime < finaltime - finaltime_tolerance
        steptime = @elapsed advance!(TCS, timestep)
        if show_details
            @printf("\n")
            @printf("\t  %4d  ",TCS.cstep)
            @printf("| %.4e ",TCS.ctime)
            @printf("| %.4e |",sqrt(sum(statistics[:,1].^2)))
            if maxiterations > 1 || check_nonlinear_residual
                @printf(" %.4e (%d) |",sqrt(sum(view(statistics,:,3).^2)), statistics[1,4])
            end
            @printf(" %.3e |",steptime)
            for j = 1 : size(statistics,1)
                @printf(" %.4e ",statistics[j,2])
            end
        end
        if do_after_each_timestep !== nothing
            do_after_each_timestep(TCS.cstep, statistics)
        end
    end

    if show_details
        @printf("\n")
    end

    if TCS.SC.user_params[:show_statistics]
        show_statistics(TCS.PDE,TCS.SC)
        @info "totaltime = $(totaltime)s"
    end

    if  abs(TCS.ctime - finaltime) > finaltime_tolerance
        @warn "final time not reached within tolerance! (consider another timestep)"
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
function advance_until_time!(DiffEQ::Module, sys::TimeControlSolver{T,Tt}, timestep::Tt, finaltime; solver = nothing, abstol = 1e-1, reltol = 1e-1, dtmin = 0, adaptive::Bool = true) where {T,Tt}
    if solver === nothing 
        solver = DiffEQ.Rosenbrock23(autodiff = false)
    end

    @info "Advancing in time from $(sys.ctime) until $finaltime using $DiffEQ with solver = $solver"

    ## generate ODE problem
    f = DiffEQ.ODEFunction(eval_rhs!, jac=eval_jacobian!, jac_prototype=jac_prototype(sys), mass_matrix=mass_matrix(sys))
    prob = DiffEQ.ODEProblem(f,sys.X.entries, (sys.ctime,finaltime),sys)

    ## solve ODE problem
    sol = DiffEQ.solve(prob,solver, abstol=abstol, reltol=reltol, dt = timestep, dtmin = dtmin, initializealg=DiffEQ.NoInit(), adaptive = adaptive)

    ## pickup solution at final time
    sys.X.entries .= view(sol,:,size(sol,2))
end