
#########################
# TIME-DEPENDENT SOLVER #
#########################


abstract type AbstractTimeIntegrationRule end
abstract type BackwardEuler <: AbstractTimeIntegrationRule end
abstract type CrankNicolson <: AbstractTimeIntegrationRule end

mutable struct TimeControlSolver{T,Tt,TiM,Tv,Ti,TIR<:AbstractTimeIntegrationRule,MassMatrixFType}
    PDE::PDEDescription                     # PDE description (operators, data etc.)
    SC::SolverConfig{T,Tv,Ti}               # solver configurations (subiterations, penalties etc.)
    LS::Array{AbstractLinearSystem{T,TiM},1}  # array for linear solvers of all subiterations
    ctime::Tt                               # current time
    cstep::Int                              # current timestep count
    last_timestep::Tt                       # last timestep
    AM::Array{FEMatrix{T,TiM,Tv,Ti},1}          # heap for mass matrices for each equation package
    which::Array{Int,1}                     # which equations shall have a time derivative?
    massmatrix_assembler::MassMatrixFType # function that assembles the massmatrix for each subiteration
    dt_is_nonlinear::Array{Bool,1}             # if true mass matrix is recomputed in each iteration
    dt_operator::Array{DataType,1}         # operators associated with the time derivative
    dt_actions::Array{<:AbstractAction,1}   # actions associated with the time derivative
    dt_lump::Array{T,1}                  # if > 0, mass matrix of the time derivative is diagona-lumped with this factor
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
                if TCS.dt_lump[pos] > 0
                    BLF = DiscreteLumpedBilinearForm([operator1, operator2], [FE1, FE2], TCS.dt_actions[pos]; T = T)
                    time = @elapsed begin
                        assemble!(A, BLF, skip_preps = false)
                        A.entries .*= TCS.dt_lump[pos]  
                    end
                else
                    BLF = DiscreteBilinearForm([operator1, operator2], [FE1, FE2], TCS.dt_actions[pos]; T = T)   
                    time = @elapsed assemble!(A, BLF, skip_preps = false) 
                end
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
    dt_lump = [],
    massmatrix_assembler = assemble_massmatrix4subiteration!,
    T_time = Float64,
    kwargs...) where {T,Tv,Ti}

    ## generate solver configurations
    SC = SolverConfig{T,Tv,Ti}(PDE; kwargs...)
    user_params = SC.user_params
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
        moreinfo_string *= "\n\tEquation ($s.$d) $(PDE.equation_names[d]) for $(PDE.unknown_names[d]) (discretised by ($(InitialValues[d].FES.name), ndofs = $(InitialValues[d].FES.ndofs)), timedependent = $(d in timedependent_equations ? "yes" : "no")"
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
        A[i] = FEMatrix{T}(FEs[subiterations[i]]; name = "system matrix $i")
        AM[i] = FEMatrix{T}(FEs[subiterations[i]]; name = "mass matrix $i")
        b[i] = FEVector{T}(FEs[subiterations[i]]; name = "system rhs $i")
        x[i] = FEVector{T}(FEs[subiterations[i]]; name = "system sol $i")
        res[i] = FEVector{T}(FEs[subiterations[i]]; name = "system residual $i")
        #set_nonzero_pattern!(A[i])
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
        new_fixed_dofs = boundarydata!(InitialValues[j],PDE.BoundaryOperators[j], InitialValues) .+ InitialValues[j].offset
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
            fill!(x[s][k],0)
            addblock!(x[s][k], InitialValues[subiterations[s][k]])
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
                if length(dt_lump) < pos
                    push!(dt_lump, 0)
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
        if TIR == CrankNicolson
            update_factorization!(LS[s]) # otherwise Crank-Nicolson test case fails (why???)
        end
    end

    # storage for last iterate (to compute change properly)
    LastIterate = deepcopy(InitialValues)

    # generate TimeControlSolver
    statistics = zeros(Float64,length(InitialValues),4)
    TCS = TimeControlSolver{T,T_time,Int64,Tv,Ti,TIR,typeof(massmatrix_assembler)}(PDE,SC,LS,start_time,0,0,AM,timedependent_equations,massmatrix_assembler,dt_is_nonlinear,dt_operator,dt_action,dt_lump,S,rhs,A,b,x,res,InitialValues, LastIterate, fixed_dofs, eqoffsets, Array{SuiteSparse.UMFPACK.UmfpackLU{T,Int64},1}(undef,length(subiterations)), statistics)

    # trigger initial assembly of all time derivative mass matrices
    for i = 1 : nsubiterations
        massmatrix_assembler(TCS, i; force = true)
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
                TCS.massmatrix_assembler(TCS, s; force = false)
                add!(A[s], AM[s]; factor = 1.0/timestep)

                for k = 1 : nsubitblocks
                    for j = 1 : nsubitblocks
                        d = subiterations[s][k]
                        addblock_matmul!(b[s][j], AM[s][j,k],LastIterate[d]; factor = 1.0/timestep)
                    end
                end
                #b[s].entries .+= (1.0/timestep) * AM[s].entries * LastIterate.entries
                flush!(A[s].entries)
            else # only update rhs
                fill!(b[s].entries,0)
                assemble!(A[s],b[s],PDE,SC,X; equations = subiterations[s], time = TCS.ctime, min_trigger = AssemblyInitial, storage_trigger = AssemblyEachTimeStep, only_rhs = true)

                ## add time derivative
                for k = 1 : nsubitblocks
                    for j = 1 : nsubitblocks
                        d = subiterations[s][k]
                        addblock_matmul!(b[s][j], AM[s][j,k],LastIterate[d]; factor = 1.0/timestep)
                    end
                end
                #b[s].entries .+= (1.0/timestep) * AM[s].entries * LastIterate.entries
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
                
                ## update and add time derivative to system matrix and right-hand side
                TCS.massmatrix_assembler(TCS, s; force = false)
                add!(S[s], AM[s]; factor = 2.0/timestep)
                rhs[s].entries .+= (2.0/timestep) * AM[s].entries * LastIterate.entries

                flush!(S[s].entries)
            else # S[s] stays the same, only update rhs[s]

                ## assembly of new right-hand side 
                fill!(b[s].entries,0)
                assemble!(A[s],b[s],PDE,SC,X; equations = subiterations[s], time = TCS.ctime, min_trigger = AssemblyInitial, storage_trigger = AssemblyEachTimeStep, only_rhs = true)
                rhs[s].entries .+= b[s].entries
                
                ## update and add time derivative to system matrix and right-hand side
                TCS.massmatrix_assembler(TCS, s; force = false)
                rhs[s].entries .+= (2.0/timestep) * AM[s].entries * LastIterate.entries
            end

        end

        # ASSEMBLE (TIME-DEPENDENT) BOUNDARY DATA
        for k = 1 : nsubitblocks
            d = subiterations[s][k]
            if any(PDE.BoundaryOperators[d].timedependent) == true
                boundarydata!(x[s][k],PDE.BoundaryOperators[d],x[s]; time = TCS.ctime, skip_enumerations = true)
            end
        end    

        # PREPARE GLOBALCONSTRAINTS
        for j = 1 : length(PDE.GlobalConstraints)
            if PDE.GlobalConstraints[j].component in subiterations[s]
                additional_fixed_dofs = apply_constraint!(A[s],b[s],PDE.GlobalConstraints[j],x[s]; current_equations = subiterations[s])
                append!(fixed_dofs, additional_fixed_dofs)
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
            mul!(res[s].entries, S[s].entries, x[s].entries)
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
                        for j = 1 : nsubitblocks
                            if lhs_erased[j,k]
                                addblock!(A[s][j,k], AM[s][j,k]; factor = 1.0/timestep)
                            end
                            if rhs_erased[j]
                                d = subiterations[s][k]
                                addblock_matmul!(b[s][j], AM[s][j,k],LastIterate[d]; factor = 1.0/timestep)
                            end
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
                
                # PREPARE GLOBALCONSTRAINTS
                for j = 1 : length(PDE.GlobalConstraints)
                    if PDE.GlobalConstraints[j].component in subiterations[s]
                        additional_fixed_dofs = apply_constraint!(A[s],b[s],PDE.GlobalConstraints[j],x[s]; current_equations = subiterations[s])
                        append!(fixed_dofs, additional_fixed_dofs)
                    end
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

"""
````
advance_until_stationarity!(TCS::TimeControlSolver, timestep; stationarity_threshold = 1e-11, maxtimesteps = 100, do_after_each_timestep = nothing)
````

Advances a TimeControlSolver in time with the given (initial) timestep until stationarity is detected (change of variables below threshold) or a maximal number of time steps is exceeded.
The function do_after_timestep is called after each timestep and can be used to print/save data (and maybe timestep control in future).
"""
function advance_until_stationarity!(TCS::TimeControlSolver{T,Tt}, timestep::Tt; stationarity_threshold = 1e-11, maxtimesteps = 100, do_after_each_timestep = nothing) where {T,Tt}
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
    totaltime = @elapsed for iteration = 1 : maxtimesteps
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
        if iteration == maxtimesteps 
            @warn "maxtimesteps = $maxtimesteps reached"
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
    @info "Advancing in time from $(Float64(TCS.ctime)) until $(Float64(finaltime))"
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
function advance_until_time!(DiffEQ::Module, sys::TimeControlSolver{T,Tt}, timestep, finaltime; solver = nothing, abstol = 1e-1, reltol = 1e-1, dtmin = 0, adaptive::Bool = true) where {T,Tt}
    if solver === nothing 
        solver = DiffEQ.Rosenbrock23(autodiff = false)
    end

    @info "Advancing in time from $(Float64(sys.ctime)) until $(Float64(finaltime)) using $DiffEQ with solver = $solver"

    ## generate ODE problem
    f = DiffEQ.ODEFunction(eval_rhs!, jac=eval_jacobian!, jac_prototype=jac_prototype(sys), mass_matrix=mass_matrix(sys))
    prob = DiffEQ.ODEProblem(f,sys.X.entries, (Float64(sys.ctime),Float64(finaltime)),sys)

    ## solve ODE problem
    sol = DiffEQ.solve(prob,solver, abstol=abstol, reltol=reltol, dt = Float64(timestep), dtmin = Float64(dtmin), initializealg=DiffEQ.NoInit(), adaptive = adaptive)

    ## pickup solution at final time
    sys.X.entries .= view(sol,:,size(sol,2))
end