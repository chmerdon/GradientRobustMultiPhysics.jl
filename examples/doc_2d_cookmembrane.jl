
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
and Neumann boundary conditions (i.e. a force that pulls the domain downwards) on the right side.

After solving the solution is plotted on the displaced mesh and also the absolute value of the stress is plotted.
=#

push!(LOAD_PATH, "../src")
using GradientRobustMultiPhysics
using ExtendableGrids
using Triangulate
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
using Printf


## grid generator for the Cook membrane via Triangulate.jl/ExtendableGrids.jl
## generates triangles and four boundary regions (1 = bottom, 2 = right, 3 = top, 4 = left)
function grid_cookmembrane(maxarea::Float64)
    triin=Triangulate.TriangulateIO()
    triin.pointlist=Matrix{Cdouble}([0 0; 0 -44; 48 0; 48 16]');
    triin.segmentlist=Matrix{Cint}([1 2 ; 2 3 ; 3 4 ; 4 1 ]')
    triin.segmentmarkerlist=Vector{Int32}([4, 1, 2, 3])
    xgrid = simplexgrid("pALVa$(@sprintf("%.15f",maxarea))", triin)
    xgrid[CellRegions] = ones(Int32,num_sources(xgrid[CellNodes]))
    xgrid[CellGeometries] = VectorOfConstants(Triangle2D,num_sources(xgrid[CellNodes]))
    return xgrid
end

## problem data for Neumann boundary
function neumann_force_right!(result,x)
    result[1] = 0.0
    result[2] = 10.0
end    

## everything is wrapped in a main function
function main()

    #####################################################################################
    #####################################################################################

    ## meshing parameters
    maxarea = 0.1
    xgrid = grid_cookmembrane(maxarea) 

    ## problem parameters
    elasticity_modulus = 1000 # elasticity modulus
    poisson_number = 1//3 # Poisson number
    shear_modulus = (1/(1+poisson_number))*elasticity_modulus
    lambda = (poisson_number/(1-2*poisson_number))*shear_modulus

    ## choose finite element type
    FEType = H1P1{2} # P1-Courant
    #FEType = H1P2{2,2} # P2

    ## postprocess parameters
    plot_grid = false
    plot_stress = true
    factor_plotdisplacement = 4
    plot_displacement = true

    #####################################################################################    
    #####################################################################################

    ## PDE description via prototype
    LinElastProblem = LinearElasticityProblem(2; shear_modulus = shear_modulus, lambda = lambda)

    ## add Neumann boundary data
    add_rhsdata!(LinElastProblem, 1,  RhsOperator(Identity, [2], neumann_force_right!, 2, 2; on_boundary = true, bonus_quadorder = 0))

    ## add Dirichlet boundary data
    add_boundarydata!(LinElastProblem, 1, [4], HomogeneousDirichletBoundary)

    ## show problem definition
    show(LinElastProblem)

    ## generate FESpace
    FES = FESpace{FEType}(xgrid)

    ## solve PDE
    Solution = FEVector{Float64}("displacement",FES)
    solve!(Solution, LinElastProblem; verbosity = 1)

    ## plot triangulation
    if plot_grid
        PyPlot.figure("grid")
        ExtendableGrids.plot(xgrid, Plotter = PyPlot)
    end

    ## plot stress
    if plot_stress
        PyPlot.figure("|eps(u)|")
        nnodes = size(xgrid[Coordinates],2)
        nodevals = zeros(Float64,4,nnodes)
        nodevalues!(nodevals,Solution[1],FES,SymmetricGradient)
        ExtendableGrids.plot(xgrid, sqrt.(nodevals[1,:].^2 + nodevals[2,:].^2 + nodevals[3,:].^2 + nodevals[4,:].^2); Plotter = PyPlot)
    end

    ## plot displacement
    if plot_displacement
        PyPlot.figure("|u| on displaced grid")
        nnodes = size(xgrid[Coordinates],2)
        nodevals = zeros(Float64,2,nnodes)
        nodevalues!(nodevals,Solution[1],FES)
        xgrid[Coordinates] = xgrid[Coordinates] + factor_plotdisplacement*nodevals[[1,2],:]
        xCoordinates = xgrid[Coordinates]
        ExtendableGrids.plot(xgrid, sqrt.(nodevals[1,:].^2 + nodevals[2,:].^2); Plotter = PyPlot, isolines = 3)
    end

end


main()
