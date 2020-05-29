
using FEXGrid
using ExtendableGrids
using ExtendableSparse
using FiniteElements
using FEAssembly
using PDETools
using QuadratureRules
#using VTKView
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
using BenchmarkTools
using Printf


include("testgrids.jl")

# problem data
function neumann_force_right!(result,x)
    result[1] = 0.0
    result[2] = 10.0
end    

function main()

    #####################################################################################
    #####################################################################################

    # meshing parameters
    xgrid = testgrid_cookmembrane() # initial simplex grid
    for j=1:4
        xgrid = uniform_refine(xgrid)
    end

    # problem parameters
    elasticity_modulus = 1000 # elasticity modulus
    poisson_number = 1//3 # Poisson number
    shear_modulus = (1/(1+poisson_number))*elasticity_modulus
    lambda = (poisson_number/(1-2*poisson_number))*shear_modulus

    # choose finite element type
    #FEType = FiniteElements.H1P1{2} # P1-Courant
    FEType = FiniteElements.H1P2{2} # P2

    # other parameters
    verbosity = 1 # deepness of messaging (the larger, the more)
    factor_plotdisplacement = 4

    #####################################################################################    
    #####################################################################################

    # PDE description
    LinElastProblem = LinearElasticityProblem(2; shearmodulus = shear_modulus, lambda = lambda)
    append!(LinElastProblem.BoundaryOperators[1], [1], HomogeneousDirichletBoundary)
    push!(LinElastProblem.RHSOperators[1], RhsOperator(Identity, neumann_force_right!, 2, 2; regions = [3], on_boundary = true, bonus_quadorder = 0))
    show(LinElastProblem)

    # generate FESpace
    FESpace = FiniteElements.FESpace{FEType}(xgrid)

    # solve PDE
    Solution = FEVector{Float64}("Displacement",FESpace)
    solve!(Solution, LinElastProblem; verbosity = verbosity)
    
    # plot triangulation
    PyPlot.figure(1)
    xgrid = split_grid_into(xgrid,Triangle2D)
    ExtendableGrids.plot(xgrid, Plotter = PyPlot)

    # plot solution
    PyPlot.figure(2)
    nnodes = size(xgrid[Coordinates],2)
    nodevals = zeros(Float64,3,nnodes)
    nodevalues!(nodevals,Solution[1],FESpace)
    nodevals[3,:] = sqrt.(nodevals[1,:].^2 + nodevals[2,:].^2)
    ExtendableGrids.plot(xgrid, nodevals[3,:]; Plotter = PyPlot)

    # plot displaced triangulation
    xgrid[Coordinates] = xgrid[Coordinates] + factor_plotdisplacement*nodevals[[1,2],:]
    PyPlot.figure(3)
    ExtendableGrids.plot(xgrid, Plotter = PyPlot)

end


main()
