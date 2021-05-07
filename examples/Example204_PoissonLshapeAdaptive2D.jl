#= 

# 204 : Poisson L-shape Adaptive Mesh Refinement
([source code](SOURCE_URL))

This example computes the standard-residual error estimator for the $H^1$ error ``e = u - u_h`` of some $H^1$-conforming
approximation ``u_h`` to the solution ``u`` of some Poisson problem ``-\Delta u = f`` on an L-shaped domain, i.e.
```math
\eta^2(u_h) := \sum_{T \in \mathcal{T}} \lvert T \rvert \| f + \Delta u_h \|^2_{L^2(T)}
+ \sum_{F \in \mathcal{F}} \lvert F \rvert \| [[\nabla u_h \cdot \mathbf{n}]] \|^2_{L^2(F)}
```
This example script showcases the evaluation of 2nd order derivatives like the Laplacian and adaptive mesh refinement.

=#

module Example204_PoissonLshapeAdaptive2D

using GradientRobustMultiPhysics
using ExtendableGrids
using Printf

## exact solution u for the Poisson problem
function exact_function!(result,x::Array{<:Real,1})
    result[1] = atan(x[2],x[1])
    if result[1] < 0
        result[1] += 2*pi
    end
    result[1] = sin(2*result[1]/3)
    result[1] *= (x[1]^2 + x[2]^2)^(1/3)
end
## ... and its gradient
function exact_function_gradient!(result,x::Array{<:Real,1})
    result[1] = atan(x[2],x[1])
    if result[1] < 0
        result[1] += 2*pi
    end
    ## du/dy = du/dr * sin(phi) + (1/r) * du/dphi * cos(phi)
    result[2] = sin(2*result[1]/3) * sin(result[1]) + cos(2*result[1]/3) * cos(result[1])
    result[2] *= (x[1]^2 + x[2]^2)^(-1/6) * 2/3 
    ## du/dx = du/dr * cos(phi) - (1/r) * du/dphi * sin(phi)
    result[1] = sin(2*result[1]/3) * cos(result[1]) - cos(2*result[1]/3) * sin(result[1])
    result[1] *= (x[1]^2 + x[2]^2)^(-1/6) * 2/3 
end

## everything is wrapped in a main function
function main(; verbosity = 0, nlevels = 20, theta = 1//3, Plotter = nothing)

    ## set log level
    set_verbosity(verbosity)

    ## initial grid
    xgrid = grid_lshape(Triangle2D)

    ## choose some finite element
    FEType = H1P2{1,2}
    
    ## negotiate data functions to the package
    user_function = DataFunction(exact_function!, [1,2]; name = "u_exact", dependencies = "X", quadorder = 5)
    user_function_gradient = DataFunction(exact_function_gradient!, [2,2]; name = "grad(u_exact)", dependencies = "X", quadorder = 4)

    ## setup Poisson problem
    Problem = PoissonProblem()
    add_boundarydata!(Problem, 1, [2,3,4,5,6,7], BestapproxDirichletBoundary; data = user_function)
    add_boundarydata!(Problem, 1, [1,8], HomogeneousDirichletBoundary)

    ## setup exact error evaluations
    L2ErrorEvaluator = L2ErrorIntegrator(Float64, user_function, Identity)
    H1ErrorEvaluator = L2ErrorIntegrator(Float64, user_function_gradient, Gradient)

    ## define error estimator
    ## kernel for jump term : |F| ||[[grad(u_h)*n_F]]||^2_L^2(F)
    xFaceVolumes::Array{Float64,1} = xgrid[FaceVolumes]
    xFaceNormals::Array{Float64,2} = xgrid[FaceNormals]
    xCellVolumes::Array{Float64,1} = xgrid[CellVolumes]
    function L2jump_integrand(result, input, item)
        result[1] = ((input[1]*xFaceNormals[1,item])^2 + (input[2]*xFaceNormals[2,item])^2) * xFaceVolumes[item]
        return nothing
    end
    ## kernel for volume term : |T| * ||f + Laplace(u_h)||^2_L^2(T)
    ## note: f = 0 here, but integrand can also be made x-dpendent to allow for non-homogeneous rhs
    function L2vol_integrand(result, input, item)
        result[1] = 0
        for j = 1 : length(input)
            result[1] += input[j]^2 * xCellVolumes[item]
        end
        return nothing
    end
    ## ... which generates an action...
    eta_jumps_action = Action(Float64, L2jump_integrand, [1,2]; name = "estimator kernel jumps", dependencies = "I", quadorder = 2)
    eta_vol_action = Action(Float64, L2vol_integrand, [1,2]; name = "estimator kernel volume", dependencies = "I", quadorder = 1)
    ## ... which is used inside an ItemIntegrator
    jumpIntegrator = ItemIntegrator(Float64,ON_IFACES,[Jump(Gradient)],eta_jumps_action; name = "η_F")
    volIntegrator = ItemIntegrator(Float64,ON_CELLS,[Laplacian],eta_vol_action; name = "η_T")
          
    ## refinement loop
    NDofs = zeros(Int, nlevels)
    Results = zeros(Float64, nlevels, 3)
    Solution = nothing
    for level = 1 : nlevels

        ## create a solution vector and solve the problem
        println("------- LEVEL $level")
        FES = FESpace{FEType}(xgrid)
        Solution = FEVector{Float64}("u_h",FES)
        solve!(Solution, Problem)
        NDofs[level] = length(Solution[1])

        ## calculate local error estimator contributions
        xFaceVolumes = xgrid[FaceVolumes]
        xFaceNormals = xgrid[FaceNormals]
        xCellVolumes = xgrid[CellVolumes]
        vol_error = zeros(Float64,1,num_sources(xgrid[CellNodes]))
        jump_error = zeros(Float64,1,num_sources(xgrid[FaceNodes]))
        evaluate!(vol_error,volIntegrator,[Solution[1]])
        evaluate!(jump_error,jumpIntegrator,[Solution[1]])

        ## calculate exact L2 error, H1 error and total estimator
        Results[level,1] = sqrt(evaluate(L2ErrorEvaluator,[Solution[1]]))
        Results[level,2] = sqrt(evaluate(H1ErrorEvaluator,[Solution[1]]))
        Results[level,3] = sqrt(sum(jump_error) + sum(vol_error))
        println("\tη = $(Results[level,3])\n\te = $(Results[level,2])")

        if level == nlevels
            break;
        end

        ## mesh refinement
        if theta >= 1
            ## uniform mesh refinement
            xgrid = uniform_refine(xgrid)
        else
            ## adaptive mesh refinement
            ## compute refinement indicators
            nfaces = num_sources(xgrid[FaceNodes])
            refinement_indicators = sum(jump_error, dims = 1)
            xFaceCells = xgrid[FaceCells]
            cell::Int = 0
            for face = 1 : nfaces, k = 1 : 2
                cell = xFaceCells[k,face]
                if cell > 0
                    refinement_indicators[face] += vol_error[1,cell]
                end
            end

            ## refine by red-green-blue refinement (incl. closuring)
            facemarker = bulk_mark(xgrid, refinement_indicators, theta; indicator_AT = ON_FACES)
            xgrid = RGB_refine(xgrid, facemarker)
        end
    end
    
    ## plot
    GradientRobustMultiPhysics.plot(xgrid, [Solution[1]], [Identity]; add_grid_plot = true, Plotter = Plotter)
    
    ## print results
    @printf("\n  NDOFS  |   L2ERROR      order   |   H1ERROR      order   | H1-ESTIMATOR   order   ")
    @printf("\n=========|========================|========================|========================\n")
    order = 0
    for j=1:nlevels
        @printf("  %6d |",NDofs[j]);
        for k = 1 : 3
            if j > 1
                order = log(Results[j-1,k]/Results[j,k]) / (log(NDofs[j]/NDofs[j-1])/2)
            end
            @printf(" %.5e ",Results[j,k])
            @printf("   %.3f   |",order)
        end
        @printf("\n")
    end
    
end

end