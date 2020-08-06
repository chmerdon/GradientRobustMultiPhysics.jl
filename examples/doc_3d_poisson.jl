#= 

# 3D Poisson-Problem
([source code](SOURCE_URL))

This example computes the solution ``u`` of the three dimensional Poisson problem
```math
\begin{aligned}
-\Delta u & = f \quad \text{in } \Omega
\end{aligned}
```
with some right-hand side ``f`` on the unit cube domain ``\Omega`` on a series of uniform refined meshes (tetrahedra or parallelepipeds).

=#

push!(LOAD_PATH, "../src")
using Printf
using GradientRobustMultiPhysics

## include file where mesh is defined
include("../src/testgrids.jl")

## problem data
function exact_function!(result,x)
    result[1] = x[1]*x[2]*x[3]^2 - x[1]*x[2]
end
function exact_gradient!(result,x)
    result[1] = x[2]*x[3]^2 - x[2]
    result[2] = x[1]*x[3]^2 - x[1]
    result[3] = 2*x[1]*x[2]*x[3]
end
function rhs!(result,x)
    result[1] = - 2*x[1]*x[2]
end

function main()

    ## choose initial mesh
    ## (replace Parallelepiped3D by Tetrahedron3D to change the cell geometries)
    xgrid = grid_unitcube(Parallelepiped3D)
    nlevels = 5 # maximal number of refinement levels

    ## set finite element type used for discretisation
    FEType = H1P1{1}

    ## create Poisson problem via prototype and add data
    Problem = PoissonProblem(3; diffusion = 1.0)
    add_boundarydata!(Problem, 1, [1,2,3,4,5,6], BestapproxDirichletBoundary; data = exact_function!, bonus_quadorder = 4)
    add_rhsdata!(Problem, 1,  RhsOperator(Identity, [0], rhs!, 3, 1; bonus_quadorder = 1))

    ## prepare error calculation
    L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 3, 1; bonus_quadorder = 4)
    H1ErrorEvaluator = L2ErrorIntegrator(exact_gradient!, Gradient, 3, 3; bonus_quadorder = 4)
    L2error = []; H1error = []; NDofs = []

    ## loop over levels
    for level = 1 : nlevels

        ## uniform mesh refinement
        xgrid = uniform_refine(xgrid)
        
        ## create finite element space
        FES = FESpace{FEType}(xgrid; dofmaps_needed = [AssemblyTypeCELL, AssemblyTypeBFACE])

        ## solve the problem
        Solution = FEVector{Float64}("Solution",FES)
        push!(NDofs,length(Solution.entries))
        solve!(Solution, Problem; verbosity = 1)

        ## calculate L2 and H1 error
        append!(L2error,sqrt(evaluate(L2ErrorEvaluator,Solution[1])))
        append!(H1error,sqrt(evaluate(H1ErrorEvaluator,Solution[1])))

    end

    ## output errors in a nice table
    println("\n   NDOF  |   L2ERROR   |   H1ERROR")
    for j=1:nlevels
        @printf("  %6d |",NDofs[j]);
        @printf(" %.5e |",L2error[j])
        @printf(" %.5e\n",H1error[j])
    end
end

main()