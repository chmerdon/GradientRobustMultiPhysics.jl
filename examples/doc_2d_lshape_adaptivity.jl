#= 

# 2D Adaptive Mesh Refinement (L-shape)
([source code](SOURCE_URL))

This example computes the standard-residual error estimator for the $H^1$ error of some (vector-valued) $H^1$-conforming
approximation ``\mathbf{u}_h`` to the solution ``\mathbf{u}`` of some vector Poisson problem ``-\Delta \mathbf{u} = 0`` on the L-shaped domain, i.e.
```math
\eta^2(\mathbf{u}_h) := \sum_{T \in \mathcal{T}} \lvert T \rvert \| \mathbf{f} + \Delta \mathbf{u}_h \|^2_{L^2(T)}
+ \sum_{F \in \mathcal{F}} \lvert F \rvert \| [[D\mathbf{u}_h \mathbf{n}]] \|^2_{L^2(F)}
```
This example script showcases the evaluation of 2nd order derivatives like the Laplacian and adaptive mesh refinement.

=#

module Example_Lshape

using GradientRobustMultiPhysics
using ExtendableGrids
using Printf

## exact solution u for the Poisson problem
function exact_function!(result,x)
    result[1] = atan(x[2],x[1])
    if result[1] < 0
        result[1] += 2*pi
    end
    result[1] = sin(2*result[1]/3)
    result[1] *= (x[1]^2 + x[2]^2)^(1/3)
end
## ... and its gradient
function exact_function_gradient!(result,x)
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
function main(; verbosity = 1, nlevels = 20, theta = 1//3, Plotter = nothing)

    ## initial grid
    xgrid = grid_lshape(Triangle2D)

    ## choose some finite element
    FEType = H1P2{1,2}
    
    ## setup a bestapproximation problem via a predefined prototype
    Problem = PoissonProblem(2; ncomponents = 1, diffusion = 1.0)
    add_boundarydata!(Problem, 1, [2,3,4,5,6,7], BestapproxDirichletBoundary; data = exact_function!, bonus_quadorder = 8)
    add_boundarydata!(Problem, 1, [1,8], HomogeneousDirichletBoundary)

    ## setup exact error evaluations
    L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 2, 1; bonus_quadorder = 10)
    H1ErrorEvaluator = L2ErrorIntegrator(exact_function_gradient!, Gradient, 2, 2; bonus_quadorder = 8)

    ## define error estimator
    ## kernel for jump term : |F| ||[[grad(u_h)*n_F]]||^2_L^2(F)
    xFaceVolumes = xgrid[FaceVolumes]
    xFaceNormals = xgrid[FaceNormals]
    xCellVolumes = xgrid[CellVolumes]
    function L2jump_integrand(result, input, item)
        result[1] = (input[1]*xFaceNormals[1,item])^2 + (input[2]*xFaceNormals[2,item])^2
        result .*= xFaceVolumes[item]
        return nothing
    end
    ## kernel for volume term : |T| * ||f + Laplace(u_h)||^2_L^2(T)
    ## note: f = 0 here, but integrand can depend on x to allow for non-homogeneous rhs
    function L2vol_integrand(result, input, x, item)
        for j = 1 : length(input)
            input[j] += result[j]
            result[j] = input[j]^2 * xCellVolumes[item]
        end
        return nothing
    end
    jumpIntegrator = ItemIntegrator{Float64,ON_IFACES}(GradientDisc{Jump},ItemWiseFunctionAction(L2jump_integrand, [1,2]; bonus_quadorder = 2), [0])
    volIntegrator = ItemIntegrator{Float64,ON_CELLS}(Laplacian,ItemWiseXFunctionAction(L2vol_integrand, [1,1], 2; bonus_quadorder = 2), [0])
          
    ## refinement loop (only uniform for now)
    NDofs = zeros(Int, nlevels)
    Results = zeros(Float64, nlevels, 3)
    Solution = nothing
    for level = 1 : nlevels

        ## create a solution vector and solve the problem
        FES = FESpace{FEType}(xgrid)
        Solution = FEVector{Float64}("Discrete Solution",FES)
        solve!(Solution, Problem; verbosity = verbosity)
        NDofs[level] = length(Solution[1])

        ## error estimator jump term 
        ## complete error estimator
        xFaceVolumes = xgrid[FaceVolumes]
        xFaceNormals = xgrid[FaceNormals]
        xCellVolumes = xgrid[CellVolumes]
        vol_error = zeros(Float64,2,num_sources(xgrid[CellNodes]))
        jump_error = zeros(Float64,2,num_sources(xgrid[FaceNodes]))
        evaluate!(vol_error,volIntegrator,[Solution[1]])
        evaluate!(jump_error,jumpIntegrator,[Solution[1]])

        ## calculate L2 error, H1 error, estimator and H2 error Results and write to results
        Results[level,1] = sqrt(evaluate(L2ErrorEvaluator,[Solution[1]]))
        Results[level,2] = sqrt(evaluate(H1ErrorEvaluator,[Solution[1]]))
        Results[level,3] = sqrt(sum(jump_error) + sum(vol_error))

        ## mesh refinement
        if theta >= 1
            ## uniform mesh refinement
            xgrid = uniform_refine(xgrid)
        else
            ## adaptive mesh refinement
            ## mark faces with largest errors
            nfaces = num_sources(xgrid[FaceNodes])
            refinement_indicators = sum(jump_error, dims = 1)
            xFaceCells = xgrid[FaceCells]
            cell::Int = 0
            for face = 1 : nfaces, k = 1 : 2
                cell = xFaceCells[k,face]
                if cell > 0
                    refinement_indicators[face] += vol_error[1,cell] + vol_error[2,cell]
                end
            end
            p = Base.sortperm(refinement_indicators[1,:], rev = true)
            totalsum = sum(refinement_indicators)
            csum = 0
            j = 0
            facemarker = zeros(Bool,nfaces)
            while csum <= theta*totalsum
                j += 1
                csum += refinement_indicators[1,p[j]]
                facemarker[p[j]] = true
            end

            ## refine by red-green-blue refinement (incl. closuring)
            xgrid = RGB_refine(xgrid, facemarker)
        end
    end
    
    ## plot
    GradientRobustMultiPhysics.plot(Solution, [0,1], [Identity,Identity]; Plotter = Plotter, verbosity = verbosity, use_subplots = false)
    
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