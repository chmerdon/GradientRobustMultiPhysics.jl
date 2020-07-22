
"""
````
mutable struct PDEDescription
    name::String
    LHS::Array{Array{AbstractPDEOperator,1},2}
    RHS::Array{Array{AbstractPDEOperator,1},1}
    BoundaryOperators::Array{BoundaryOperator,1}
    GlobalConstraints::Array{AbstractGlobalConstraint,1}
end
````

struct that describes a PDE system with n equations and n unknowns

A PDE system is described by
- its name::String
- an size n x n array of Array{AbstractPDEOperator,1} LHS that describes the left-hand sides
- an length n array of Array{AbstractPDEOperator,1} RHS that describes the right-hand sides
- an length n array of BoundaryOperators that describes the boundary conditions for each unknown
- an array of GlobalConstraints that describes additional global constraints
"""
mutable struct PDEDescription
    name::String
    LHSOperators::Array{Array{AbstractPDEOperator,1},2}
    RHSOperators::Array{Array{AbstractPDEOperator,1},1}
    BoundaryOperators::Array{BoundaryOperator,1}
    GlobalConstraints::Array{AbstractGlobalConstraint,1}
end


"""
$(TYPEDSIGNATURES)

Create empty PDEDEscription with no unknowns.
"""
function PDEDescription(name)
    return PDEDescription(name, 0, 0, 0)
end

"""
$(TYPEDSIGNATURES)

Create empty PDEDEscription for a specified number of unknowns.
"""
function PDEDescription(name, nunknowns::Int, ncomponents::Array{Int,1}, dim::Int = 2)
    # LEFT-HAND-SIDE
    MyLHS = Array{Array{AbstractPDEOperator,1},2}(undef,nunknowns,nunknowns)
    for j=1:nunknowns, k = 1:nunknowns
        MyLHS[j,k] = []
    end

    # RIGHT-HAND SIDE
    MyRHS = Array{Array{AbstractPDEOperator,1},1}(undef,nunknowns)
    for j=1:nunknowns
        MyRHS[j] = []
    end

    # BOUNDARY OPERATOR
    MyBoundary = Array{BoundaryOperator,1}(undef,nunknowns)
    for j=1:nunknowns
        MyBoundary[j] = BoundaryOperator(dim,ncomponents[j])
    end

    # GLOBAL CONSTRAINTS
    MyGlobalConstraints = Array{AbstractGlobalConstraint,1}(undef,0)

    return PDEDescription(name, MyLHS, MyRHS, MyBoundary, MyGlobalConstraints)
end



"""
$(TYPEDSIGNATURES)

Adds another unknown of specified dimensions to the PDEDescription.
"""
function add_unknown!(PDE::PDEDescription, ncomponents::Int, dim::Int = 2)
    nunknowns = length(PDE.RHSOperators)+1
    push!(PDE.RHSOperators,[])
    push!(PDE.BoundaryOperators,BoundaryOperator(dim,ncomponents))
    NewLHS = Array{Array{AbstractPDEOperator,1},2}(undef,nunknowns,nunknowns)
    for j=1:nunknowns, k = 1:nunknowns
        if j < nunknowns && k < nunknowns
            NewLHS[j,k] = deepcopy(PDE.LHSOperators[j,k])
        else
            NewLHS[j,k] = []
        end
    end
    PDE.LHSOperators = NewLHS
end

"""
$(TYPEDSIGNATURES)

Adds the given PDEOperator to the left-hand side of the PDEDescription at the specified position.
"""
function add_operator!(PDE::PDEDescription,position::Array{Int,1},O::AbstractPDEOperatorLHS)
    push!(PDE.LHSOperators[position[1],position[2]],O)
end

"""
$(TYPEDSIGNATURES)

Adds the given PDEOperator to the right-hand side of the PDEDEscription at the specified position.
"""
function add_rhsdata!(PDE::PDEDescription,position::Int,O::AbstractPDEOperatorRHS)
    push!(PDE.RHSOperators[position],O)
end

"""
$(TYPEDSIGNATURES)

Adds the given boundary data to the at specified position in the BoundaryOperator of the PDEDEscription.

If timedependent == true, that data function depends also on time t and is reassembled in any advance! step of a TimeControlSolver.
"""
function add_boundarydata!(PDE::PDEDescription,position::Int,regions, btype::Type{<:AbstractBoundaryType}; timedependent::Bool = false, data = Nothing, bonus_quadorder::Int = 0)
    Base.append!(PDE.BoundaryOperators[position],regions, btype; data = data, timedependent = timedependent, bonus_quadorder = bonus_quadorder)
end


"""
$(TYPEDSIGNATURES)

Adds the given global constraint to the PDEDEscription.
"""
function add_constraint!(PDE::PDEDescription,GC::AbstractGlobalConstraint)
    Base.push!(PDE.GlobalConstraints,GC)
end


"""
$(TYPEDSIGNATURES)

Custom `show` function for `PDEDescription` that prints the PDE systems and all assigned operators
"""
function Base.show(io::IO, PDE::PDEDescription)
    println("\nPDE-DESCRIPTION")
    println("===============")
    println("  name = $(PDE.name)\n")

    println("  LHS block | PDEOperator(s)")
    for j=1:size(PDE.LHSOperators,1), k=1:size(PDE.LHSOperators,2)
        if length(PDE.LHSOperators[j,k]) > 0
            print("    [$j,$k]   | ")
            for o = 1 : length(PDE.LHSOperators[j,k])
                try
                    print("$(PDE.LHSOperators[j,k][o].name) (regions = $(PDE.LHSOperators[j,k][o].regions))")
                catch
                    print("$(PDE.LHSOperators[j,k][o].name) (regions = [0])")
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
                try
                    print("$(typeof(PDE.RHSOperators[j][o])) (regions = $(PDE.RHSOperators[j][o].regions))")
                catch
                    print("$(typeof(PDE.RHSOperators[j][o])) (regions = [0])")
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
        println("  GlobalConstraints[$j] : $(PDE.GlobalConstraints[j].name)")
    end
end
