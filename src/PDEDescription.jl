
"""
$(TYPEDEF)

struct that describes a PDE system

A PDE system is described by
- its name::String
- an Array{Array{AbstractPDEOperator,1},2} that describes the left-hand sides
- an Array{Array{AbstractPDEOperator,1},1} that describes the right-hand sides
- an Array{BoundaryOperator,1} that describes the boundary conditions
- an Array{AbstractGlobalConstraint,1} that describes additional global constraints
"""
mutable struct PDEDescription
    name::String
    LHSOperators::Array{Array{AbstractPDEOperator,1},2}
    RHSOperators::Array{Array{AbstractPDEOperator,1},1}
    BoundaryOperators::Array{BoundaryOperator,1}
    GlobalConstraints::Array{AbstractGlobalConstraint,1}
end

function PDEDescription(name, LHS, RHS, BoundaryOperators)
    nFEs = length(RHS)
    NoConstraints = Array{AbstractGlobalConstraint,1}(undef,0)
    return PDEDescription(name, LHS, RHS, BoundaryOperators, NoConstraints)
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
