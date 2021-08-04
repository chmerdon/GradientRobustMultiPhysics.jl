#= 

# 231 : Stokes ``(P1 \oplus RT0) \times P0``
([source code](SOURCE_URL))

This example computes the velocity ``\mathbf{u}`` and pressure ``\mathbf{p}`` of the incompressible Navier--Stokes problem
```math
\begin{aligned}
- \mu \Delta \mathbf{u} + \nabla p & = \mathbf{f}\\
\mathrm{div}(u) & = 0
\end{aligned}
```
with exterior force ``\mathbf{f}`` and some parameter ``\mu`` and inhomogeneous Dirichlet boundary data.

The problem will be solved by a ``(P1 \oplus RT0) \times P0`` scheme suggested by [Li/Rui,arXiv:2012.01689 [math.NA]](https://arxiv.org/abs/2012.01689).
The velocity space employs continuous P1 functions plus additional (only H(div)-conforming) RT0 functions and a P0 pressure space
leading to an exactly divergence-free discrete velocity.
=#

module Example231_StokesHdivP1RT

using GradientRobustMultiPhysics

## functions that define the exact solution and the data 
function exact_pressure!(result,x,t)
    result[1] = cos(t)*(sin(x[1])*cos(x[2]) + (cos(1) -1)*sin(1))
end
function u!(result,x,t)
    result[1] = cos(t)*(sin(π*x[1]-0.7)*sin(π*x[2]+0.2))
    result[2] = cos(t)*(cos(π*x[1]-0.7)*cos(π*x[2]+0.2))
end
function exact_velogradient!(result,x,t)
    result[1] = π*cos(t)*(cos(π*x[1]-0.7)*sin(π*x[2]+0.2))
    result[2] = π*cos(t)*(sin(π*x[1]-0.7)*cos(π*x[2]+0.2))
    result[3] = -π*cos(t)*(sin(π*x[1]-0.7)*cos(π*x[2]+0.2))
    result[4] = -π*cos(t)*(cos(π*x[1]-0.7)*sin(π*x[2]+0.2))
end
function rhs(μ)
    function closure!(result,x,t)
        ## exact Laplacian
        result[1] = 2*π*π*μ*cos(t)*(sin(π*x[1]-0.7)*sin(π*x[2]+0.2))
        result[2] = 2*π*π*μ*cos(t)*(cos(π*x[1]-0.7)*cos(π*x[2]+0.2))
        ## exact pressure gradient
        result[1] += cos(t)*cos(x[1])*cos(x[2])
        result[2] -= cos(t)*sin(x[1])*sin(x[2])
    end
end

## everything is wrapped in a main function
function main(; μ = 1e-3, nlevels = 5, Plotter = nothing, verbosity = 0, T = 1, α = 20)

    ## set log level
    set_verbosity(verbosity)

    ## FEType
    FETypes = [H1P1{2}, HDIVRT0{2}, H1P0{1}]
    
    ## initial grid
    xgrid = grid_unitsquare(Triangle2D)
    xCellVolumes = xgrid[CellVolumes] # will be overwritten if mesh is refined
    
    ## load exact flow data
    u = DataFunction(u!, [2,2]; dependencies = "XT", name = "u", quadorder = 5)
    p = DataFunction(exact_pressure!, [1,2]; dependencies = "XT", name = "p", quadorder = 4)
    ∇u = DataFunction(exact_velogradient!, [4,2]; dependencies = "XT", name = "∇u", quadorder = 4)
    f = DataFunction(rhs(μ), [2,2]; dependencies = "XT", name = "f", quadorder = 5)

    ## load Stokes problem prototype and assign data
    Problem = PDEDescription("Stokes problem")
    add_unknown!(Problem; equation_name = "momentum equation (P1 part)", unknown_name = "u_P1")
    add_unknown!(Problem; unknown_name = "u_RT", equation_name = "momentum equation (RT0 part)")
    add_unknown!(Problem; equation_name = "incompressibility constraint", unknown_name = "pressure", algebraic_constraint = true)

    ## add Laplacian for both velocity blocks
    add_operator!(Problem, [1,1], LaplaceOperator(μ))
    add_operator!(Problem, [2,2], LaplaceOperator(μ))

    ## add stabilising term for RT0 block
    stabaction = Action(Float64, (result, input, item) -> (result .= input / xCellVolumes[item]), [2,2]; dependencies = "I", quadorder = 0)
    add_operator!(Problem, [2,2], AbstractBilinearForm([Identity, Identity], stabaction; name = "α h^-2 (u_RT,v_RT)", factor = α, AT = ON_CELLS))

    ## add Lagrange multiplier for divergence of velocity
    add_operator!(Problem, [1,3], LagrangeMultiplier(Divergence))
    add_operator!(Problem, [2,3], LagrangeMultiplier(Divergence))

    ## add right-hand side
    add_rhsdata!(Problem, 1, RhsOperator(Identity, [1], f))
    add_rhsdata!(Problem, 2, RhsOperator(Identity, [1], f))

    ## add boundary data and global constraints
    add_boundarydata!(Problem, 1, [1,2,3,4], BestapproxDirichletBoundary; data = u)
    add_boundarydata!(Problem, 2, [1,2,3,4], HomogeneousDirichletBoundary)
    add_constraint!(Problem, FixedIntegralMean(3,0))

    ## show final problem description
    @show Problem

    ## prepare error calculation
    L2VelocityErrorEvaluator = L2ErrorIntegrator(Float64, u, [Identity, Identity]; time = T)
    L2PressureErrorEvaluator = L2ErrorIntegrator(Float64, p, Identity; time = T)
    H1VelocityErrorEvaluator = L2ErrorIntegrator(Float64, ∇u, Gradient; time = T)
    L2VeloDivEvaluator = L2NormIntegrator(Float64,1 , [Divergence, Divergence])
    Results = zeros(Float64,nlevels,4); NDofs = zeros(Int,nlevels)

    ## loop over levels
    for level = 1 : nlevels

        ## refine grid and update grid component references
        xgrid = uniform_refine(xgrid)
        xCellVolumes = xgrid[CellVolumes]

        ## generate FES spaces and solution vector
        FES = [FESpace{FETypes[1]}(xgrid), FESpace{FETypes[2]}(xgrid), FESpace{FETypes[3]}(xgrid)]
        Solution = FEVector{Float64}(["u_P1", "u_RT", "p_h"],FES)

        ## solve
        solve!(Solution, Problem; time = T)

        ## plot
        GradientRobustMultiPhysics.plot(xgrid, [Solution[1], Solution[2], Solution[3]], [Identity, Identity, Identity]; Plotter = Plotter)

        ## compute L2 and H1 errors and save data
        NDofs[level] = length(Solution.entries)
        Results[level,1] = sqrt(evaluate(L2VelocityErrorEvaluator,[Solution[1],Solution[2]]))
        Results[level,2] = sqrt(evaluate(L2PressureErrorEvaluator,Solution[3]))
        Results[level,3] = sqrt(evaluate(H1VelocityErrorEvaluator,Solution[1]))
        Results[level,4] = sqrt(evaluate(L2VeloDivEvaluator,[Solution[1], Solution[2]]))
    end    

    ## print/show convergence history
    print_convergencehistory(NDofs, Results; X_to_h = X -> X.^(-1/2), ylabels = ["|| u - u_h ||", "|| p - p_h ||", "|| ∇(u - u_P1) ||", "|| div(u_h) ||"])
    plot_convergencehistory(NDofs, Results[:,1:3]; add_h_powers = [1,2], X_to_h = X -> X.^(-1/2), Plotter = Plotter, ylabels = ["|| u - u_h ||", "|| p - p_h ||", "|| ∇(u - u_h) ||"])
end
end