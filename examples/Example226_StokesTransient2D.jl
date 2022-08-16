#= 

# 226 : Stokes Transient 2D
([source code](SOURCE_URL))

This example computes a velocity ``\mathbf{u}`` and pressure ``\mathbf{p}`` of the incompressible Navier--Stokes problem
```math
\begin{aligned}
\mathbf{u}_t - \mu \Delta \mathbf{u} + \nabla p & = \mathbf{f}\\
\mathrm{div}(u) & = 0
\end{aligned}
```
with (possibly time-dependent) exterior force ``\mathbf{f}`` and some ν parameter ``\mu``.

In this example we solve an analytical toy problem with prescribed solution
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
Compare e.g. Bernardi--Raugel and Bernardi--Raugel pressure-robust by switching 'reconstruct'.
=#

module Example226_StokesTransient2D

using GradientRobustMultiPhysics
using ExtendableGrids

## everything is wrapped in a main function
function main(; 
    nlevels = 4,            # number of refinement levels
    dt = 1e-3,              # time step
    T = 1e-2,               # final time
    ν = 1e-6,               # viscosity
    reconstruct = true,     # use gradient-robust modification ?
    graddiv = 0,            # factor for grad-div stabilisation
    verbosity = 0           # higher number increases printed (debug) messages
    )
    
    ## set log level 
    set_verbosity(verbosity)

    ## initial grid
    xgrid = grid_unitsquare(Triangle2D);

    ## choose one of these (inf-sup stable) finite element type pairs
    FETypes = [H1BR{2}, L2P0{1}]; # Bernardi--Raugel 
  
    #####################################################################################

    ## set testfunction operator for certain testfunctions
    ## (pressure-robustness chooses a reconstruction that can exploit the L2-orthogonality onto gradients)
    test_operator = reconstruct ? ReconstructionIdentity{HDIVBDM1{2}} : Identity

    ## negotiate data functions to the package
    ## note that dependencies "XT" marks the function to be x- and t-dependent
    ## that causes the solver to automatically reassemble associated operators in each time step
    u = DataFunction((result, x, t) -> (
            result[1] = (1+t)*cos(x[2]);
            result[2] = (1+t)*sin(x[1]);
        ), [2,2]; name = "u", dependencies = "XT", bonus_quadorder = 5)
    p = DataFunction((result, x) -> (
            result[1] = sin(x[1]+x[2]) - 2*sin(1)+sin(2)
        ), [1,2]; name = "p", dependencies = "X", bonus_quadorder = 5)
    dt_u = eval_dt(u)
    Δu = eval_Δ(u)
    ∇p = eval_∇(p)
    f = DataFunction((result, x, t) -> (
            result .= dt_u(x,t) .- ν*Δu(x,t) .+ view(∇p(x,t),:);
        ), [2,2]; name = "f", dependencies = "XT", bonus_quadorder = 5)
    ∇u = ∇(u)

    ## load Stokes problem prototype and assign data
    Problem = IncompressibleNavierStokesProblem(2; viscosity = ν, nonlinear = false)
    add_boundarydata!(Problem, 1, [1,2,3,4], BestapproxDirichletBoundary; data = u)
    add_rhsdata!(Problem, 1, LinearForm(test_operator, f))

    ## add grad-div stabilisation
    if graddiv > 0
        add_operator!(Problem, [1,1], BilinearForm("graddiv-stabilisation (div x div)", Divergence, Divergence; factor = graddiv))
    end

    ## define bestapproximation problems
    BAP_L2_p = L2BestapproximationProblem(p; bestapprox_boundary_regions = [])
    BAP_L2_u = L2BestapproximationProblem(u; bestapprox_boundary_regions = [1,2,3,4])
    BAP_H1_u = H1BestapproximationProblem(∇u, u; bestapprox_boundary_regions = [1,2,3,4])

    ## define ItemIntegrators for L2/H1 error computation and arrays to store them
    L2VelocityError = L2ErrorIntegrator(u, Identity; time = T)
    L2PressureError = L2ErrorIntegrator(p, Identity)
    H1VelocityError = L2ErrorIntegrator(∇u, Gradient; time = T) 
    Results = zeros(Float64, nlevels, 6)
    NDofs = zeros(Int, nlevels)
    
    ## loop over levels
    for level = 1 : nlevels

        ## refine grid
        xgrid = uniform_refine(xgrid)

        ## generate FESpaces
        FES = [FESpace{FETypes[1]}(xgrid), FESpace{FETypes[2]}(xgrid)]

        ## generate solution fector
        Solution = FEVector(FES)

        ## set initial solution ( = bestapproximation at time 0)
        BA_L2_u = FEVector("L2-Bestapproximation velocity",FES[1])
        solve!(BA_L2_u, BAP_L2_u; time = 0)
        Solution[1][:] = BA_L2_u[1][:]

        ## generate time-dependent solver and chance rhs data
        TCS = TimeControlSolver(Problem, Solution, CrankNicolson; timedependent_equations = [1], skip_update = [-1], dt_operator = [test_operator])
        advance_until_time!(TCS, dt, T)

        ## solve bestapproximation problems at final time for comparison
        BA_L2_p = FEVector("L2-Bestapproximation pressure",FES[2])
        BA_H1_u = FEVector("H1-Bestapproximation velocity",FES[1])
        solve!(BA_L2_u, BAP_L2_u; time = T)
        solve!(BA_L2_p, BAP_L2_p)
        solve!(BA_H1_u, BAP_H1_u; time = T)

        ## compute L2 and H1 errors and save data
        NDofs[level] = length(Solution.entries)
        Results[level,1] = sqrt(evaluate(L2VelocityError,Solution[1]))
        Results[level,2] = sqrt(evaluate(L2VelocityError,BA_L2_u[1]))
        Results[level,3] = sqrt(evaluate(L2PressureError,Solution[2]))
        Results[level,4] = sqrt(evaluate(L2PressureError,BA_L2_p[1]))
        Results[level,5] = sqrt(evaluate(H1VelocityError,Solution[1]))
        Results[level,6] = sqrt(evaluate(H1VelocityError,BA_H1_u[1]))
    end    

    ## print convergence history
    print_convergencehistory(NDofs, Results[:,1:2]; X_to_h = X -> X.^(-1/2), ylabels = ["||u-u_h||", "||u-Πu||"])
    print_convergencehistory(NDofs, Results[:,3:4]; X_to_h = X -> X.^(-1/2), ylabels = ["||p-p_h||", "||p-πp||"])
    print_convergencehistory(NDofs, Results[:,5:6]; X_to_h = X -> X.^(-1/2), ylabels = ["||∇(u-u_h)||", "||∇(u-Su)||"])
end

end