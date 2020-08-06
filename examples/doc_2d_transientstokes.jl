#= 

# 2D Transient Stokes-Problem
([source code](SOURCE_URL))

This example computes a velocity ``\mathbf{u}`` and pressure ``\mathbf{o}`` of the incompressible Navier--Stokes problem
```math
\begin{aligned}
\mathbf{u}_t - \mu \Delta \mathbf{u} + \nabla p & = \mathbf{f}\\
\mathrm{div}(u) & = 0
\end{aligned}
```
with (possibly time-dependent) exterior force ``\mathbf{f}`` and some viscosity parameter ``\mu``.

In this example we solve a analytical toy problem with prescribed solution
```math
\begin{aligned}
\mathbf{u}(\mathbf{x},t) & = (1+t)(\cos(x_2), \sin(x_1))^T\\
p(\mathbf{x}) &= \sin(x_1+x_2) - 2\sin(1) + \sin(2)
\end{aligned}
```
with time-dependent right-hand side and inhomogeneous Dirichlet boundary data. The example showcases the
benefits of pressure-robustness in time-dependent linear Stokes problem in presence of complicated pressures and small viscosities.
The problem is solved on series of finer and finer unstructured simplex meshes and compares the error of the discrete Stokes solution,
an interpolation into the same space and the best-approximations into the same space. While a pressure-robust variant shows optimally
converging errors close to the best-approximations, a non pressure-robust discretisations show suboptimal (or no) convergence!
Compare e.g. Bernardi--Raugel and bernardi--Raugel pressure-robust by (un)commenting the responsible lines in this example.
=#

push!(LOAD_PATH, "../src")
using ExtendableGrids
#using VTKView
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
using Printf
using Triangulate
using GradientRobustMultiPhysics

## file that includes the mesh definition
include("../src/testgrids.jl")

## problem data
function exact_pressure!(result,x)
    result[1] = sin(x[1]+x[2]) - 2*sin(1)+sin(2)
end
function exact_velocity!(result,x,t)
    result[1] = (1+t)*cos(x[2]);
    result[2] = (1+t)*sin(x[1]);
end
function exact_velocity_gradient!(result,x,t)
    result[1] = 0.0
    result[2] = -(1+t)*sin(x[2]);
    result[3] = (1+t)*cos(x[1]);
    result[4] = 0.0;
end
function exact_rhs!(viscosity)
    function closure(result,x,t)
        result[1] = viscosity*(1+t)*cos(x[2]) + cos(x[1]+x[2]) + cos(x[2])
        result[2] = viscosity*(1+t)*sin(x[1]) + cos(x[1]+x[2]) + sin(x[1])
    end
end


## grid generator that generates  unstructured simplex mesh
function grid_unitsquare_unstructured(maxarea::Float64)
    triin=Triangulate.TriangulateIO()
    triin.pointlist=Matrix{Cdouble}([0 0; 1 0; 1 1; 0 1]');
    triin.segmentlist=Matrix{Cint}([1 2 ; 2 3 ; 3 4 ; 4 1 ]')
    triin.segmentmarkerlist=Vector{Int32}([4, 1, 2, 3])
    xgrid = simplexgrid("pALVa$(@sprintf("%.15f",maxarea))", triin)
    xgrid[CellRegions] = VectorOfConstants(Int32(1),num_sources(xgrid[CellNodes]))
    xgrid[CellGeometries] = VectorOfConstants(Triangle2D,num_sources(xgrid[CellNodes]))
    return xgrid
end


function main()
    #####################################################################################
    #####################################################################################

    ## problem parameters
    viscosity = 1e-6
    timestep = 1e-3
    T = 1e-2 # final time
    initial_h = 1/4 # initial mesh width of subtriangles
    nlevels = 4 # maximal number of refinement levels
    barycentric_refinement = false; reconstruct = false # do not change

    ## choose one of these (inf-sup stable) finite element type pairs
    #FETypes = [H1P2{2,2}, H1P1{1}] # Taylor--Hood
    #FETypes = [H1CR{2}, L2P0{1}] # Crouzeix--Raviart
    #FETypes = [H1CR{2}, L2P0{1}]; reconstruct = true # Crouzeix-Raviart gradient-robust
    #FETypes = [H1MINI{2,2}, H1P1{1}] # MINI element on triangles only
    #FETypes = [H1MINI{2,2}, H1CR{1}] # MINI element on triangles/quads
    #FETypes = [H1BR{2}, L2P0{1}] # Bernardi--Raugel
    FETypes = [H1BR{2}, L2P0{1}]; reconstruct = true # Bernardi--Raugel gradient-robust
    #FETypes = [H1P2{2,2}, L2P1{1}]; initial_h *= 2; barycentric_refinement = true # Scott-Vogelius 

    #####################################################################################    
    #####################################################################################

    ## set testfunction operator for certain testfunctions
    ## (pressure-robustness chooses a reconstruction that can exploit the L2-orthogonality onto gradients)
    ## (Scott-Vogelius is divergence-free and is pressure-robust without modifications)
    if reconstruct
        testfunction_operator = ReconstructionIdentity{HDIVRT0{2}}
    else
        testfunction_operator = Identity
    end

    ## load Stokes problem prototype and assign data
    StokesProblem = IncompressibleNavierStokesProblem(2; viscosity = viscosity, nonlinear = false)
    add_boundarydata!(StokesProblem, 1, [1,2,3,4], BestapproxDirichletBoundary; data = exact_velocity!, bonus_quadorder = 5, timedependent = true)
    add_rhsdata!(StokesProblem, 1, RhsOperator(testfunction_operator, [1], exact_rhs!(viscosity), 2, 2; bonus_quadorder = 5, timedependent = true))

    ## define bestapproximation problems
    L2PressureBestapproximationProblem = L2BestapproximationProblem(exact_pressure!, 2, 1; bestapprox_boundary_regions = [], bonus_quadorder = 5)
    L2VelocityBestapproximationProblemInitial = L2BestapproximationProblem(exact_velocity!, 2, 2; bestapprox_boundary_regions = [1,2,3,4], bonus_quadorder = 5, time_dependent_data = true, time = 0)
    L2VelocityBestapproximationProblemFinal = L2BestapproximationProblem(exact_velocity!, 2, 2; bestapprox_boundary_regions = [1,2,3,4], bonus_quadorder = 5, time_dependent_data = true, time = T)
    H1VelocityBestapproximationProblemFinal = H1BestapproximationProblem(exact_velocity_gradient!, exact_velocity!, 2, 2; bestapprox_boundary_regions = [1,2,3,4], bonus_quadorder = 5, bonus_quadorder_boundary = 5, time_dependent_data = true, time = T)

    ## define ItemIntegrators for L2/H1 error computation and arrays to store them
    L2VelocityErrorEvaluator = L2ErrorIntegrator(exact_velocity!, Identity, 2, 2; bonus_quadorder = 5, time_dependent_data = true, time = T)
    L2PressureErrorEvaluator = L2ErrorIntegrator(exact_pressure!, Identity, 2, 1; bonus_quadorder = 5)
    H1VelocityErrorEvaluator = L2ErrorIntegrator(exact_velocity_gradient!, Gradient, 2, 4; bonus_quadorder = 5, time_dependent_data = true, time = T)
    L2error_velocity = []; L2error_pressure = []; L2errorInterpolation_velocity = []; NDofs = []
    L2errorInterpolation_pressure = []; L2errorBestApproximation_velocity = []; L2errorBestApproximation_pressure = []
    H1error_velocity = []; H1errorInterpolation_velocity = []; H1errorBestApproximation_velocity = []
    
    ## loop over levels
    for level = 1 : nlevels

        xgrid = grid_unitsquare_unstructured((initial_h/2^(level-1))^2)
        ## in case of Scott-Vogelius we use barycentric refinement
        if barycentric_refinement == true
            xgrid = barycentric_refine(xgrid)
        end

        ## generate FESpaces
        FESpaceVelocity = FESpace{FETypes[1]}(xgrid)
        FESpacePressure = FESpace{FETypes[2]}(xgrid)

        ## generate solution fector
        Solution = FEVector{Float64}("Stokes velocity",FESpaceVelocity)
        append!(Solution,"Stokes pressure",FESpacePressure)
        push!(NDofs,length(Solution.entries))
        Base.show(Solution)

        ## set initial solution ( = bestapproximation at time 0)
        L2VelocityBestapproximation = FEVector{Float64}("L2-Bestapproximation velocity",FESpaceVelocity)
        solve!(L2VelocityBestapproximation, L2VelocityBestapproximationProblemInitial)
        Solution[1][:] = L2VelocityBestapproximation[1][:]

        ## generate time-dependent solver and chance rhs data
        TCS = TimeControlSolver(StokesProblem, Solution, BackwardEuler; timedependent_equations = [1], dt_testfunction_operator = [testfunction_operator], verbosity = 1)

        ## time loop
        change = 0.0
        maxIterations = ceil(T / timestep)
        for iteration = 1 : maxIterations
            if iteration > 1
                change = advance!(TCS, timestep; reuse_matrix = [true])
            else
                change = advance!(TCS, timestep)
            end
            @printf("  iteration %4d",iteration)
            @printf("  time = %.4e",TCS.ctime)
            @printf("  change = %.4e \n",change)
        end

        ## interpolate exact solution at final time for comparison
        Interpolation = FEVector{Float64}("Interpolation velocity",FESpaceVelocity)
        append!(Interpolation,"Interpolation pressure",FESpacePressure)
        interpolate!(Interpolation[1], exact_velocity!; bonus_quadorder = 2, time_dependent = true, time = T)
        interpolate!(Interpolation[2], exact_pressure!; bonus_quadorder = 1)

        ## solve bestapproximation problems at final time for comparison
        L2PressureBestapproximation = FEVector{Float64}("L2-Bestapproximation pressure",FESpacePressure)
        H1VelocityBestapproximation = FEVector{Float64}("H1-Bestapproximation velocity",FESpaceVelocity)
        solve!(L2VelocityBestapproximation, L2VelocityBestapproximationProblemFinal;)
        solve!(L2PressureBestapproximation, L2PressureBestapproximationProblem;)
        solve!(H1VelocityBestapproximation, H1VelocityBestapproximationProblemFinal;)

        ## compute L2 and H1 error of all solutions
        append!(L2error_velocity,sqrt(evaluate(L2VelocityErrorEvaluator,Solution[1])))
        append!(L2errorInterpolation_velocity,sqrt(evaluate(L2VelocityErrorEvaluator,Interpolation[1])))
        append!(L2errorBestApproximation_velocity,sqrt(evaluate(L2VelocityErrorEvaluator,L2VelocityBestapproximation[1])))
        append!(L2error_pressure,sqrt(evaluate(L2PressureErrorEvaluator,Solution[2])))
        append!(L2errorInterpolation_pressure,sqrt(evaluate(L2PressureErrorEvaluator,Interpolation[2])))
        append!(L2errorBestApproximation_pressure,sqrt(evaluate(L2PressureErrorEvaluator,L2PressureBestapproximation[1])))
        append!(H1error_velocity,sqrt(evaluate(H1VelocityErrorEvaluator,Solution[1])))
        append!(H1errorInterpolation_velocity,sqrt(evaluate(H1VelocityErrorEvaluator,Interpolation[1])))
        append!(H1errorBestApproximation_velocity,sqrt(evaluate(H1VelocityErrorEvaluator,H1VelocityBestapproximation[1])))
        
        ## ouput errors
        if (level == nlevels)
            println("\n         |   L2ERROR   |   L2ERROR   |   L2ERROR")
            println("   NDOF  | VELO-STOKES | VELO-INTERP | VELO-L2BEST");
            for j=1:nlevels
                @printf("  %6d |",NDofs[j]);
                @printf(" %.5e |",L2error_velocity[j])
                @printf(" %.5e |",L2errorInterpolation_velocity[j])
                @printf(" %.5e\n",L2errorBestApproximation_velocity[j])
            end
            println("\n         |   H1ERROR   |   H1ERROR   |   H1ERROR")
            println("   NDOF  | VELO-STOKES | VELO-INTERP | VELO-H1BEST");
            for j=1:nlevels
                @printf("  %6d |",NDofs[j]);
                @printf(" %.5e |",H1error_velocity[j])
                @printf(" %.5e |",H1errorInterpolation_velocity[j])
                @printf(" %.5e\n",H1errorBestApproximation_velocity[j])
            end
            println("\n         |   L2ERROR   |   L2ERROR   |   L2ERROR")
            println("   NDOF  | PRES-STOKES | PRES-INTERP | PRES-L2BEST");
            for j=1:nlevels
                @printf("  %6d |",NDofs[j]);
                @printf(" %.5e |",L2error_pressure[j])
                @printf(" %.5e |",L2errorInterpolation_pressure[j])
                @printf(" %.5e\n",L2errorBestApproximation_pressure[j])
            end
            println("\nLEGEND\n======")
            println("VELO-STOKES : discrete Stokes velocity solution ($(FESpaceVelocity.name))")
            println("VELO-INTERP : interpolation of exact velocity")
            println("VELO-L2BEST : L2-Bestapproximation of exact velocity (with boundary data)")
            println("VELO-H1BEST : H1-Bestapproximation of exact velocity (with boudnary data)")
            println("PRES-STOKES : discrete Stokes pressure solution ($(FESpacePressure.name))")
            println("PRES-INTERP : interpolation of exact pressure")
            println("PRES-L2BEST : L2-Bestapproximation of exact pressure (without boundary data)")
        end    
    end    
end


main()
