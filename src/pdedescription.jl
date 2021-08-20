

"""
````
mutable struct PDEDescription
    name::String
    equation_names::Array{String,1}
    unknown_names::Array{String,1}
    algebraic_constraint::Array{Bool,1}
    LHS::Array{Array{AbstractPDEOperator,1},2}
    RHS::Array{Array{AbstractPDEOperator,1},1}
    BoundaryOperators::Array{BoundaryOperator,1}
    GlobalConstraints::Array{AbstractGlobalConstraint,1}
end
````

struct that describes a PDE system with n equations and n unknowns

A PDE system is described by
- its name
- the names of its equations
- the names of its unknowns
- is the variable related to an algebraic constraint? (e.g. pressure in incompressible CFD, this has implications e.g. for the time discretisation)
- a size n x n array of Array{AbstractPDEOperator,1} LHS that describes the left-hand sides
- a length n array of Array{AbstractPDEOperator,1} RHS that describes the right-hand sides
- a length n array of BoundaryOperators that describes the boundary conditions for each unknown
- an array of GlobalConstraints that describes additional global constraints

A PDEDescription mainly is a set of PDEOperators arranged in a quadratic n by n matrix.
Every matrix row refers to one equation and the positioning of the PDEOperators (e.g. a bilinearform)
immediately sets the information which unknowns have to be used to evaluate the operator. Also 
nonlinear PDEOperators are possible where extra information on the further involved uknowns have to be specified.
UserData is also assigned to the PDEDescription depending on their type. Operator coefficients are
assigned directly to the PDEOperators (in form of AbstractActions), right-hand side data is assigned
to the right-hand side array of PDEOperators and boundary data is assigned to the BoundaryOperators
of the PDEDescription. Additionaly global constraints (like a global zero integral mean) can be assigned as a GlobalConstraint.

"""

mutable struct PDEDescription
    name::String
    equation_names::Array{String,1}
    unknown_names::Array{String,1}
    algebraic_constraint::Array{Bool,1}
    LHSOperators::Array{Array{AbstractPDEOperator,1},2}
    RHSOperators::Array{Array{AbstractPDEOperator,1},1}
    BoundaryOperators::Array{BoundaryOperator,1}
    GlobalConstraints::Array{AbstractGlobalConstraint,1}
end


"""
$(TYPEDSIGNATURES)

Create empty PDEDescription with no unknowns.
"""
function PDEDescription(name::String)
    return PDEDescription(name, 0)
end

"""
$(TYPEDSIGNATURES)

Create empty PDEDescription for a specified number of unknowns.
"""
function PDEDescription(name::String, nunknowns::Int; algebraic::Array{Bool,1} = Array{Bool,1}(undef,0), unknown_names::Array{String,1} = Array{String,1}(undef,0), equation_names::Array{String,1} = Array{String,1}(undef,0))

    # LEFT-HAND-SIDE
    MyLHS = Array{Array{AbstractPDEOperator,1},2}(undef,nunknowns,nunknowns)
    for j=1:nunknowns, k = 1:nunknowns
        MyLHS[j,k] = []
    end

    # RIGHT-HAND SIDE
    MyRHS = Array{Array{AbstractPDEOperator,1},1}(undef,nunknowns)
    for j=1:nunknowns
        MyRHS[j] = []
        if length(unknown_names) < j
            push!(unknown_names,"unknown $j")
        end
        if length(equation_names) < j
            push!(equation_names,"equation $j")
        end
        if length(algebraic) < j
            push!(algebraic,false)
        end
    end

    # BOUNDARY OPERATOR
    MyBoundary = Array{BoundaryOperator,1}(undef,nunknowns)
    for j=1:nunknowns
        MyBoundary[j] = BoundaryOperator()
    end

    # GLOBAL CONSTRAINTS
    MyGlobalConstraints = Array{AbstractGlobalConstraint,1}(undef,0)

    if nunknowns == 0
        @logmsg DeepInfo "Created empty PDEDescription $name"
    else
        @logmsg DeepInfo "Created PDEDescription $name with $nunknowns unknowns $unknown_names"
    end

    return PDEDescription(name, equation_names, unknown_names, algebraic, MyLHS, MyRHS, MyBoundary, MyGlobalConstraints)
end



"""
$(TYPEDSIGNATURES)

Adds another unknown to the PDEDescription. With the optional argument algebraic_constraint = true the unknown and the related equation
can be mask as an algebraic constraint. (Currently this only has a consequence if the system is integrated in time with the Crank-Nicolson rule.)
"""
function add_unknown!(PDE::PDEDescription; equation_name::String = "", unknown_name::String = "", algebraic_constraint::Bool = false)
    nunknowns = length(PDE.RHSOperators)+1
    if equation_name == ""
        equation_name = "equation $nunknowns"
    end
    if unknown_name == ""
        unknown_name = "unknown $nunknowns"
    end
    push!(PDE.equation_names,equation_name)
    push!(PDE.unknown_names,unknown_name)
    push!(PDE.algebraic_constraint,algebraic_constraint)
    push!(PDE.RHSOperators,[])
    push!(PDE.BoundaryOperators,BoundaryOperator())
    NewLHS = Array{Array{AbstractPDEOperator,1},2}(undef,nunknowns,nunknowns)
    for j=1:nunknowns, k = 1:nunknowns
        if j < nunknowns && k < nunknowns
            NewLHS[j,k] = deepcopy(PDE.LHSOperators[j,k])
        else
            NewLHS[j,k] = []
        end
    end
    PDE.LHSOperators = NewLHS
    @logmsg DeepInfo "Added unknown $unknown_name to PDEDescription $(PDE.name)"
end

"""
$(TYPEDSIGNATURES)

Adds the given abstract PDEOperator to the left-hand side of the PDEDescription at the specified position.
The id of the operator in the coressponding LHS block of PDEDescription is returned.
"""
function add_operator!(PDE::PDEDescription,position::Array{Int,1},O::AbstractPDEOperator; equation_name::String = "")
    push!(PDE.LHSOperators[position[1],position[2]],O)
    if equation_name != ""
        PDE.equation_names[position[1]] = equation_name
    end
    @logmsg DeepInfo "Added operator $(O.name) to LHS block $position of PDEDescription $(PDE.name)"
    return length(PDE.LHSOperators[position[1],position[2]])
end

"""
$(TYPEDSIGNATURES)

Adds the given PDEOperator to the left-hand side of the PDEDescription at the specified position. Optionally, the name of the equation can be changed.
The id of the operator in the coressponding LHS block of PDEDescription is returned (in case of a nonlinear operator with a RHS contribution also the id
in the coressponding RHS block is returned).
"""
function add_operator!(PDE::PDEDescription,position::Array{Int,1},O::PDEOperator; equation_name::String = "")
    ## nonlinear forms may push additional left-hand side and right-hand side operators to the description
    if typeof(O).parameters[2] <: APT_NonlinearForm
        add_operator!(PDE, position[1],O; equation_name = equation_name)
    else
        push!(PDE.LHSOperators[position[1],position[2]],O)
        if equation_name != ""
            PDE.equation_names[position[1]] = equation_name
        end
        @logmsg DeepInfo "Added operator $(O.name) to LHS block $position of PDEDescription $(PDE.name)"
        return length(PDE.LHSOperators[position[1],position[2]])
    end
end

function add_operator!(PDE::PDEDescription,position::Array{Int,1},O::SchurComplement; equation_name::String = "")
    push!(PDE.LHSOperators[position[1],position[2]],O)
    if equation_name != ""
        PDE.equation_names[position[1]] = equation_name
    end
    push!(PDE.RHSOperators[position[1]],O)
    @logmsg DeepInfo "Added Schur complement operator $(O.name) to LHS block $position of PDEDescription $(PDE.name)"
    @logmsg DeepInfo "Added Schur complement operator $(O.name) to RHS block $(position[1]) of PDEDescription $(PDE.name)"
    return length(PDE.LHSOperators[position[1],position[2]]), length(PDE.RHSOperators[position[1]])
end


"""
$(TYPEDSIGNATURES)

Adds the given nonlinear PDEOperator to the specified equation of the PDEDescription. Optionally, the name of the equation can be changed.
"""
function add_operator!(PDE::PDEDescription,equation::Int,O::PDEOperator{T,APT}; equation_name::String = "") where{T, APT <: APT_NonlinearForm}
    ## nonlinear forms may push additional left-hand side and right-hand side operators to the description
    dependencies = unique(O.fixed_arguments_ids)
    if length(dependencies) > 0 
        for j = 1 : length(dependencies)
            Oc = deepcopy(O)
            Oc.newton_arguments = findall(x -> x == dependencies[j], O.fixed_arguments_ids)
            Oc.name = O.name * " [∂$(PDE.unknown_names[dependencies[j]])]"
            push!(PDE.LHSOperators[equation,dependencies[j]],Oc)
            @logmsg DeepInfo "Added operator $(O.name) to LHS block [$equation,$(dependencies[j])] of PDEDescription $(PDE.name) with newton_arguments = O.newton_arguments"
            dependencies[j] = 0
        end
    end
    if typeof(O.action_rhs) != NoAction
        push!(PDE.RHSOperators[equation],O)
        @logmsg DeepInfo "Added operator $(O.name) also to RHS block $equation of PDEDescription $(PDE.name)"
    end
end

function add_operator!(PDE::PDEDescription,position::Array{Int,1},O::SchurComplement; equation_name::String = "")
    push!(PDE.LHSOperators[position[1],position[2]],O)
    if equation_name != ""
        PDE.equation_names[position[1]] = equation_name
    end
    push!(PDE.RHSOperators[position[1]],O)
    @logmsg DeepInfo "Added Schur complement operator $(O.name) to LHS block $position of PDEDescription $(PDE.name)"
    @logmsg DeepInfo "Added Schur complement operator $(O.name) to RHS block $(position[1]) of PDEDescription $(PDE.name)"
    return length(PDE.LHSOperators[position[1],position[2]]), length(PDE.RHSOperators[position[1]])
end


"""
$(TYPEDSIGNATURES)

Adds the given PDEOperator to the right-hand side of the PDEDescription at the specified position. The id of the operator in the coressponding RHS block of PDEDescription is returned.
"""
function add_rhsdata!(PDE::PDEDescription,position::Int,O::AbstractPDEOperator)
    push!(PDE.RHSOperators[position],O)
    @logmsg DeepInfo "Added operator $(O.name) to RHS block $position of PDEDescription $(PDE.name)"
    return length(PDE.RHSOperators[position])
end

"""
$(TYPEDSIGNATURES)

Adds the given boundary data with the specified AbstractBoundaryType at the specified position in the BoundaryOperator of the PDEDescription.

If timedependent == true, that data function depends also on time t and is reassembled in any advance! step of a TimeControlSolver.
"""
function add_boundarydata!(PDE::PDEDescription,position::Int,regions, btype::Type{<:AbstractBoundaryType}; data = Nothing)
    Base.append!(PDE.BoundaryOperators[position],regions, btype; data = data)
    @logmsg DeepInfo "Added boundary_data for unknown $(PDE.unknown_names[position]) in region(s) $regions to PDEDescription $(PDE.name)"
end


"""
$(TYPEDSIGNATURES)

Adds the given global constraint to the PDEDescription.
"""
function add_constraint!(PDE::PDEDescription,GC::AbstractGlobalConstraint)
    Base.push!(PDE.GlobalConstraints,GC)
    @logmsg DeepInfo "Added global constraint to PDEDescription $(PDE.name)"
end


"""
$(TYPEDSIGNATURES)

Custom `show` function for `PDEDescription` that prints the PDE systems and all assigned operators
"""
function Base.show(io::IO, PDE::PDEDescription)
    println("\nPDE-DESCRIPTION")
    println("===============")
    println("  system name = $(PDE.name)\n")

    println("     id   | unknown name / equation name")
    for j=1:length(PDE.unknown_names)
        print("    [$j]   | $(PDE.unknown_names[j]) / $(PDE.equation_names[j]) \n")
    end


    println("\n  LHS block | PDEOperator(s)")
    for j=1:size(PDE.LHSOperators,1), k=1:size(PDE.LHSOperators,2)
        if length(PDE.LHSOperators[j,k]) > 0
            print("    [$j,$k]   | ")
            for o = 1 : length(PDE.LHSOperators[j,k])
                if typeof(PDE.LHSOperators[j,k][o]) <: PDEOperator
                    print("$(PDE.LHSOperators[j,k][o].name) (APT = $(typeof(PDE.LHSOperators[j,k][o]).parameters[2]), AT = $(typeof(PDE.LHSOperators[j,k][o]).parameters[3]), regions = $(PDE.LHSOperators[j,k][o].regions))")
                else
                    print("$(typeof(PDE.LHSOperators[j,k][o]))")
                end
                if o == length(PDE.LHSOperators[j,k])
                    println("")
                else
                    print("\n            | ")
                end
            end
        else    
            println("    [$j,$k]   | none")
        end
    end

    println("\n  RHS block | PDEOperator(s)")
    for j=1:size(PDE.RHSOperators,1)
        if length(PDE.RHSOperators[j]) > 0
            print("     [$j]    | ")
            for o = 1 : length(PDE.RHSOperators[j])

                if typeof(PDE.RHSOperators[j][o]) <: PDEOperator
                    print("$(PDE.RHSOperators[j][o].name) (APT = $(typeof(PDE.RHSOperators[j][o]).parameters[2]), AT = $(typeof(PDE.RHSOperators[j][o]).parameters[3]), regions = $(PDE.RHSOperators[j][o].regions))")
                else
                    print("$(typeof(PDE.RHSOperators[j][o]))")
                end
                if o == length(PDE.RHSOperators[j])
                    println("")
                else
                    print("\n            | ")
                end
            end
        else    
            println("     [$j]    | none")
        end
    end

    println("")
    for j=1:length(PDE.BoundaryOperators)
        print("   BoundaryOperator[$j] : ")
        try
            if length(PDE.BoundaryOperators[j].regions4boundarytype[BestapproxDirichletBoundary]) > 0
                print("BestapproxDirichletBoundary -> $(PDE.BoundaryOperators[j].regions4boundarytype[BestapproxDirichletBoundary])\n                         ")
            end
        catch
        end
        try
            if length(PDE.BoundaryOperators[j].regions4boundarytype[InterpolateDirichletBoundary]) > 0
                print("InterpolateDirichletBoundary -> $(PDE.BoundaryOperators[j].regions4boundarytype[InterpolateDirichletBoundary])\n                         ")
            end
        catch
        end
        try
            if length(PDE.BoundaryOperators[j].regions4boundarytype[HomogeneousDirichletBoundary]) > 0
                print("HomogeneousDirichletBoundary -> $(PDE.BoundaryOperators[j].regions4boundarytype[HomogeneousDirichletBoundary])\n                          ")
            end
        catch
        end
        println("")
    end

    println("")
    for j=1:length(PDE.GlobalConstraints)
        println("  GlobalConstraints[$j] : $(PDE.GlobalConstraints[j].name) ")
    end
end
