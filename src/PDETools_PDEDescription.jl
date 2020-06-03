
##################
# PDEDescription #
##################
#
# A PDE is described by an nxn matrix and vector of PDEOperator
# the indices of matrix relate to FEBlocks given to solver
# all Operators of [i,k] - matrixblock are assembled into system matrix block [i*n+k]
# all Operators of [i] - rhs-block are assembled into rhs block [i]
#
# additionally BoundayOperators and GlobalConstraints are assigned to handle
# boundary data and global side constraints (like a fixed global integral mean)

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
                print("$(typeof(PDE.RHSOperators[j][o])) (regions = $(PDE.RHSOperators[j][o].regions))")
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
