
#= 

# 2D Linear Elasticity
([source code](SOURCE_URL))

This example computes the solution ``\mathbf{u}`` of the linear elasticity problem
```math
\begin{aligned}
-\mathrm{div} (\mathbb{C} \epsilon(\mathbf{u})) & = \mathbf{f} \quad \text{in } \Omega\\
\mathbb{C} \epsilon(\mathbf{u}) \cdot \mathbf{n} & = \mathbf{g} \quad \text{along } \Gamma_N
\end{aligned}
```
with exterior force ``\mathbf{f}``, Neumann boundary force ``\mathbf{g}``, and the stiffness tensor
```math
\mathbb{C} \epsilon(\mathbf{u}) = 2 \mu \epsilon( \mathbf{u}) + \lambda \mathrm{tr}(\epsilon( \mathbf{u}))
```
for isotropic media.

The domain will be the Cook membrane and the displacement has homogeneous boundary conditions on the left side of the domain
and Neumann boundary conditions (i.e. a constant force that pulls the domain upwards) on the right side.

=#

module Example_2DCookMembrane

using GradientRobustMultiPhysics
using ExtendableGrids
using Printf

## everything is wrapped in a main function
function main(; verbosity = 0, Plotter = nothing)

    ## set log level
    set_verbosity(verbosity)

    ## load mesh and refine
    xgrid = simplexgrid("assets/2d_grid_cookmembrane.sg")
    xgrid = uniform_refine(xgrid,2)

    ## problem parameters
    elasticity_modulus = 1000 # elasticity modulus
    poisson_number = 1//3 # Poisson number
    shear_modulus = (1/(1+poisson_number))*elasticity_modulus
    lambda = (poisson_number/(1-2*poisson_number))*shear_modulus

    ## PDE description via prototype
    Problem = LinearElasticityProblem(2; shear_modulus = shear_modulus, lambda = lambda)

    ## add boundary data
    add_rhsdata!(Problem, 1, RhsOperator(Identity, [2], DataFunction([0,10]; name = "g"); AT = ON_BFACES))
    add_boundarydata!(Problem, 1, [4], HomogeneousDirichletBoundary)

    ## show and solve PDE
    @show Problem
    FEType = H1P1{2} # P1-Courant FEM will be used
    Solution = FEVector{Float64}("displacement",FESpace{FEType}(xgrid))
    solve!(Solution, Problem)

    ## plot stress on displaced mesh
    displace_mesh!(xgrid, Solution[1]; magnify = 4)
    GradientRobustMultiPhysics.plot(xgrid, [Solution[1], Solution[1]], [Identity, Gradient]; Plotter = Plotter)
end

end

#=
### Output of default main() run
=#
Example_2DCookMembrane.main()