
#= 

# 3D Linear Elasticity
([source code](SOURCE_URL))

This example computes the solution ``\mathbf{u}`` of the linear elasticity problem
```math
\begin{aligned}
-\mathrm{div} (\mathbb{C} \epsilon(\mathbf{u})) & = \mathbf{f} \quad \text{in } \Omega\\
\mathbb{C} \epsilon(\mathbf{u}) \cdot \mathbf{n} & = \mathbf{g} \quad \text{along } \Gamma_N
\end{aligned}
```
with exterior force ``\mathbf{f}``, Neumann boundary force ``\mathbf{g}``, and the stiffness tensor ``\mathbb{C}``.
=#

push!(LOAD_PATH, "../src")
using GradientRobustMultiPhysics

## problem data for Neumann boundary
function neumann_force_right!(result,x)
    result[1] = 0.0
    result[2] = 0.0
    result[3] = 10.0
end    

## everything is wrapped in a main function
function main()

    #####################################################################################
    #####################################################################################

    ## meshing parameters
    xgrid = uniform_refine(grid_unitcube(Parallelepiped3D; scale = [4//3,3//4,1//4]),3)

    ## problem parameters
    elasticity_modulus = 1000 # elasticity modulus
    poisson_number = 1//3 # Poisson number
    shear_modulus = (1/(1+poisson_number))*elasticity_modulus
    lambda = (poisson_number/(1-2*poisson_number))*shear_modulus

    ## choose finite element type
    FEType = H1P1{3} # P1-Courant

    #####################################################################################    
    #####################################################################################

    ## PDE description via prototype
    LinElastProblem = LinearElasticityProblem(3; shearmodulus = shear_modulus, lambda = lambda)

    ## add Neumann boundary data
    add_rhsdata!(LinElastProblem, 1,  RhsOperator(Identity, [3], neumann_force_right!, 3, 3; on_boundary = true, bonus_quadorder = 0))

    ## add Dirichlet boundary data
    add_boundarydata!(LinElastProblem, 1, [5], HomogeneousDirichletBoundary)

    ## show problem definition
    show(LinElastProblem)

    ## generate FESpace
    FES = FESpace{FEType}(xgrid)

    ## solve PDE
    Solution = FEVector{Float64}("displacement",FES)
    solve!(Solution, LinElastProblem; verbosity = 3)

    ## write to vtk
    writeVTK!("data_elasticity3d.vtk", Solution)

end


main()
