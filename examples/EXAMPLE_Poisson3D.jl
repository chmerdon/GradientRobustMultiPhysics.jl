
using ExtendableGrids
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
using Printf

# load finite element module
push!(LOAD_PATH, "../src")
using GradientRobustMultiPhysics


include("../src/testgrids.jl")

# problem data
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

    verbosity = 1 # <-- increase/decrease this number to get more/less printouts on what is happening

    # initial mesh
    xgrid = testgrid_cube_uniform(Hexahedron3D)
    #xgrid = testgrid_cube_uniform(Tetrahedron3D)
    nlevels = 4 # maximal number of refinement levels

    # Define Poisson problem via PDETooles_PDEProtoTypes
    Problem = PoissonProblem(3; diffusion = 1.0)
    add_boundarydata!(Problem, 1, [2,5], HomogeneousDirichletBoundary)
    add_boundarydata!(Problem, 1, [1,3,4,6], BestapproxDirichletBoundary; data = exact_function!, bonus_quadorder = 4)
    add_rhsdata!(Problem, 1,  RhsOperator(Identity, [rhs!], 3, 1; bonus_quadorder = 1))
    show(Problem)

    # prepare error calculation
    L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 3, 1; bonus_quadorder = 4)
    H1ErrorEvaluator = L2ErrorIntegrator(exact_gradient!, Gradient, 3, 3; bonus_quadorder = 4)
    L2error = []
    H1error = []
    NDofs = []

    # loop over levels
    for level = 1 : nlevels

        # uniform mesh refinement
        xgrid = uniform_refine(xgrid)
        # choose some finite element space
        FEType = H1P1{1}
        FES = FESpace{FEType}(xgrid; dofmaps_needed = [AssemblyTypeCELL, AssemblyTypeBFACE], verbosity = verbosity - 1)

        # solve the problem
        Solution = FEVector{Float64}("Solution",FES)
        push!(NDofs,length(Solution.entries))
        solve!(Solution, Problem; verbosity = verbosity)

        # calculate L2 error and L2 divergence error
        append!(L2error,sqrt(evaluate(L2ErrorEvaluator,Solution[1])))
        append!(H1error,sqrt(evaluate(H1ErrorEvaluator,Solution[1])))

        # output
        if (level == nlevels)
            println("\n   NDOF  |   L2ERROR   |   H1ERROR")
            for j=1:nlevels
                @printf("  %6d |",NDofs[j]);
                @printf(" %.5e |",L2error[j])
                @printf(" %.5e\n",H1error[j])
            end
        end    
    end

end

main()