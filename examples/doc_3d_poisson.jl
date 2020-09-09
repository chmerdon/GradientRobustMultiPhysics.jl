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

module Example_3DPoisson

using GradientRobustMultiPhysics
using Printf

## problem data
function exact_function!(result,x)
    result[1] = x[1]*(x[3] - x[2]) + x[2]*x[2]
    return nothing
end
function exact_gradient!(result,x)
    result[1] = x[3] - x[2]
    result[2] = - x[1] + 2*x[2]
    result[3] = x[1]
    return nothing
end
function rhs!(result,x)
    result[1] = - 2
    return nothing
end

## everything is wrapped in a main function
function main(; verbosity = 1)

    ## choose initial mesh
    ## (replace Parallelepiped3D by Tetrahedron3D to change the cell geometries)
    xgrid_b = grid_unitcube(Parallelepiped3D)
    nlevels = 5 # maximal number of refinement levels
    split_into_tets = true # grid is split into Tetrahedron3D after each refinement of the basis grid
    write_vtk = false

    ## set finite element type used for discretisation
    FEType = H1P1{1}

    ## create Poisson problem via prototype and add data
    Problem = PoissonProblem(3; diffusion = 1.0)
    add_boundarydata!(Problem, 1, [1,2,3,4,5,6], BestapproxDirichletBoundary; data = exact_function!, bonus_quadorder = 2)
    add_rhsdata!(Problem, 1,  RhsOperator(Identity, [0], rhs!, 3, 1; bonus_quadorder = 0))

    ## prepare error calculation
    L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 3, 1; bonus_quadorder = 2)
    H1ErrorEvaluator = L2ErrorIntegrator(exact_gradient!, Gradient, 3, 3; bonus_quadorder = 1)
    L2error = []; H1error = []; NDofs = []

    ## loop over levels
    for level = 1 : nlevels

        ## uniform mesh refinement
        if level > 1 
            xgrid_b = uniform_refine(xgrid_b)
        end
        if split_into_tets
            xgrid = split_grid_into(xgrid_b, Tetrahedron3D)
        else
            xgrid = xgrid_b
        end
        
        ## create finite element space
        FES = FESpace{FEType}(xgrid; dofmaps_needed = [CellDofs, BFaceDofs])

        ## solve the problem
        Solution = FEVector{Float64}("Solution",FES)
        push!(NDofs,length(Solution.entries))
        solve!(Solution, Problem; verbosity = verbosity)

        ## calculate L2 and H1 error
        append!(L2error,sqrt(evaluate(L2ErrorEvaluator,Solution[1])))
        append!(H1error,sqrt(evaluate(H1ErrorEvaluator,Solution[1])))

        if write_vtk
            mkpath("data/example_poisson3d/")
            writeVTK!("data/example_poisson3d/results_level$(level).vtk", Solution)
        end

    end

    ## output errors in a nice table
    println("\n   NDOF  |   L2ERROR   |   H1ERROR")
    for j=1:nlevels
        @printf("  %6d |",NDofs[j]);
        @printf(" %.5e |",L2error[j])
        @printf(" %.5e\n",H1error[j])
    end
end

end