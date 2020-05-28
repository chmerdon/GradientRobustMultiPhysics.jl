
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
    result[2] = 1000.0
end    

function main()

    #####################################################################################
    #####################################################################################

    # meshing parameters
    xgrid = testgrid_cookmembrane() # initial simplex grid
    nlevels = 4 # number of refinement levels

    # problem parameters
    elasticity_modulus = 100000 # elasticity modulus
    poisson_number = 0.4999 # Poisson number
    shear_modulus = (1/(1+poisson_number))*elasticity_modulus
    lambda = (poisson_number/(1-2*poisson_number))*shear_modulus

    # fem/solver parameters
    #fem = "P1" # P1-Courant
    #fem = "CR" # Crouzeix--Raviart
    fem = "P2" # P2 element
    verbosity = 10 # deepness of messaging (the larger, the more)
    factor_plotdisplacement = 4

    #####################################################################################    
    #####################################################################################

    # PDE description
    LinElastProblem = LinearElasticityProblem(2; shearmodulus = shear_modulus, lambda = lambda)
    append!(LinElastProblem.BoundaryOperators[1], [1], HomogeneousDirichletBoundary)
    push!(LinElastProblem.RHSOperators[1], RhsOperator(Identity, neumann_force_right!, 2, 2; regions = [3], on_boundary = true, bonus_quadorder = 0))


    # define ItemIntegrators for L2/H1 error computation
    NDofs = []

    # loop over levels
    for level = 1 : nlevels

        # uniform mesh refinement
        if (level > 1) 
            xgrid = uniform_refine(xgrid)
        end

        # generate FE
        if fem == "P1"
            FE = FiniteElements.getH1P1FiniteElement(xgrid,2)
        elseif fem == "CR"
            FE = FiniteElements.getH1CRFiniteElement(xgrid,2)
        elseif fem == "P2"
            FE = FiniteElements.getH1P2FiniteElement(xgrid,2)
        end        
        if verbosity > 2
            FiniteElements.show(FE)
        end    

        # solve elasticity problem
        Solution = FEVector{Float64}("Displacement",FE)
        solve!(Solution, LinElastProblem; verbosity = verbosity - 1)
        push!(NDofs,length(Solution.entries))
        
        # plot final solution
        if (level == nlevels)

            # plot triangulation
            PyPlot.figure(1)
            xgrid = split_grid_into(xgrid,Triangle2D)
            ExtendableGrids.plot(xgrid, Plotter = PyPlot)

            # plot solution
            PyPlot.figure(2)
            nnodes = size(xgrid[Coordinates],2)
            nodevals = zeros(Float64,3,nnodes)
            nodevalues!(nodevals,Solution[1],FE)
            nodevals[3,:] = sqrt.(nodevals[1,:].^2 + nodevals[2,:].^2)
            ExtendableGrids.plot(xgrid, nodevals[3,:]; Plotter = PyPlot)

            # plot displaced triangulation
            xgrid[Coordinates] = xgrid[Coordinates] + factor_plotdisplacement*nodevals[[1,2],:]
            PyPlot.figure(3)
            ExtendableGrids.plot(xgrid, Plotter = PyPlot)
        end    
    end    


end


main()
