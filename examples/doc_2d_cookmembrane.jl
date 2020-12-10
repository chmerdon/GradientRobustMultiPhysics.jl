
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
and Neumann boundary conditions (i.e. a force that pulls the domain upwards) on the right side.

=#

module Example_2DCookMembrane

using GradientRobustMultiPhysics
using ExtendableGrids
using Printf

## problem data for Neumann boundary
function neumann_force_right!(result)
    result[1] = 0
    result[2] = 10
end    

## everything is wrapped in a main function
function main(; verbosity = 1, Plotter = nothing)

    ## load mesh and refine
    xgrid = simplexgrid(IOStream;file = "assets/2d_grid_cookmembrane.sg")
    xgrid = uniform_refine(xgrid,2)

    ## problem parameters
    elasticity_modulus = 1000 # elasticity modulus
    poisson_number = 1//3 # Poisson number
    shear_modulus = (1/(1+poisson_number))*elasticity_modulus
    lambda = (poisson_number/(1-2*poisson_number))*shear_modulus

    ## negotiate data function to the package
    user_function_neumann_bnd = DataFunction(neumann_force_right!, [2,2]; name = "g", dependencies = "", quadorder = 0)

    ## choose finite element type
    FEType = H1P1{2} # P1-Courant
    #FEType = H1P2{2,2} # P2

    ## PDE description via prototype
    LinElastProblem = LinearElasticityProblem(2; shear_modulus = shear_modulus, lambda = lambda)

    ## add Neumann boundary data
    add_rhsdata!(LinElastProblem, 1,  RhsOperator(Identity, [2], user_function_neumann_bnd; on_boundary = true))

    ## add Dirichlet boundary data
    add_boundarydata!(LinElastProblem, 1, [4], HomogeneousDirichletBoundary)

    ## show problem definition
    show(LinElastProblem)

    ## generate FESpace
    FES = FESpace{FEType}(xgrid)

    ## solve PDE
    Solution = FEVector{Float64}("displacement",FES)
    solve!(Solution, LinElastProblem; verbosity = verbosity)

    ## plot stress
    GradientRobustMultiPhysics.plot(Solution, [1,1], [Identity, Gradient]; Plotter = Plotter, verbosity = verbosity, use_subplots = true)
end

end