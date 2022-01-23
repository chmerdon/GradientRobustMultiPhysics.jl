#= 

# 207 : Nonlinear Elasticity Bimetal 2D
([source code](SOURCE_URL))

This example computes the displacement field ``u`` of the nonlinear elasticity problem
```math
\begin{aligned}
-\mathrm{div}(\mathbb{C} (\epsilon(u)-\epsilon_T)) & = 0 \quad \text{in } \Omega
\end{aligned}
```
where an isotropic stress tensor ``\mathbb{C}`` is applied to the nonlinear strain ``\epsilon(u) := \frac{1}{2}(\nabla u + (\nabla u)^T + (\nabla u)^T \nabla u)``
and a misfit strain  ``\epsilon_T := \Delta T \alpha`` due to thermal load caused by temperature(s) ``\Delta T`` and thermal expansion coefficients ``\alpha`` (that may be different)
in the two regions of the bimetal.

This example demonstrates how to setup a nonlinear expression with automatic differentiation and how to assign it to the problem description.
=#

module Example207_NonlinearElasticityBimetal2D

using GradientRobustMultiPhysics
using ExtendableGrids
using GridVisualize

## callable structs to reduce allocations
mutable struct nonlinear_operator{T}
    λ::T
    μ::T
    ϵT::T
end
const ops = [nonlinear_operator(0.0,0.0,0.0), nonlinear_operator(0.0,0.0,0.0)] # one for each region

function strain!(result, input)
    result[1] = input[1]
    result[2] = input[4]
    result[3] = input[2] + input[3]

    ## add nonlinear part of the strain 1/2 * (grad(u)'*grad(u))
    result[1] += 1//2 * (input[1]^2 + input[3]^2)
    result[2] += 1//2 * (input[2]^2 + input[4]^2)
    result[3] += input[1]*input[2] + input[3]*input[4]
    return nothing
end

## kernel for nonlinear operator
(op::nonlinear_operator)(result, input) = (
        ## input = grad(u) written as a vector
        ## compute strain and subtract thermal strain (all in Voigt notation)
        strain!(result, input);
        result[1] -= op.ϵT;
        result[2] -= op.ϵT;

        ## multiply with isotropic stress tensor
        ## (stored in input[5:7] using Voigt notation)
        input[5] = op.λ*(result[1]+result[2]) + 2*op.μ*result[1];
        input[6] = op.λ*(result[1]+result[2]) + 2*op.μ*result[2];
        input[7] = 2*op.μ*result[3];
    
        ## write strain into result
        result[1] = input[5];
        result[2] = input[7];
        result[3] = input[7];
        result[4] = input[6];
        return nothing
)

## everything is wrapped in a main function
function main(; ν = [0.3,0.3], E = [2.1,1.1], ΔT = [580,580], α = [1.3e-5,2.4e-5], scale = [20,500], nref = 0, material_border = 0.5, store = Threads.nthreads() > 1, verbosity = 0, Plotter = nothing)

    ## set log level
    set_verbosity(verbosity)

    ## compute Lame' coefficients μ and λ from ν and E
    μ = E ./ (2  .* (1 .+ ν.^(-1)))
    λ = E .* ν ./ ( (1 .- 2*ν) .* (1 .+ ν))

    ## change coefficients of operators
    for region = 1 : 2
        ops[region].μ = μ[region]
        ops[region].λ = λ[region]
        ops[region].ϵT = ΔT[region] * α[region]
    end

    ## generate bimetal mesh
    xgrid = bimetal_strip2D(; scale = scale, n = 2*(nref+1))

    ## prepare nonlinear operator (one for each bimetal region)
    nonlin_operator_1 = NonlinearForm(Gradient, [Gradient], [1], ops[1], [4,4,7]; name = "C(ϵ(u)-ϵT):∇v", regions = [1], bonus_quadorder = 3, newton = true, store = store, sparse_jacobian = true) 
    nonlin_operator_2 = NonlinearForm(Gradient, [Gradient], [1], ops[2], [4,4,7]; name = "C(ϵ(u)-ϵT):∇v", regions = [2], bonus_quadorder = 3, newton = true, store = store, sparse_jacobian = true) 
    
    ## generate problem description and assign nonlinear operators
    Problem = PDEDescription("nonlinear elasticity problem")
    add_unknown!(Problem; unknown_name = "u", equation_name = "displacement equation")
    add_operator!(Problem, 1, nonlin_operator_1)
    add_operator!(Problem, 1, nonlin_operator_2)
    add_boundarydata!(Problem, 1, [1], HomogeneousDirichletBoundary)
    @show Problem

    ## create finite element space and solution vector
    FEType = H1P2{2,2}
    FES = FESpace{FEType}(xgrid)
    Solution = FEVector("u_h",FES)

    ## solve
    solve!(Solution, Problem; maxiterations = 10, target_residual = 1e-9, show_statistics = true)

    ## displace mesh and plot
    p = GridVisualizer(; Plotter = Plotter, layout = (3,1), clear = true, resolution = (800,600))
    grad_nodevals = nodevalues(Solution[1], Gradient)
    strain_nodevals = zeros(Float64,3,num_nodes(xgrid))
    for j = 1 : num_nodes(xgrid)
        strain!(view(strain_nodevals,:,j), view(grad_nodevals,:,j))
    end
    scalarplot!(p[1,1], xgrid, view(strain_nodevals,1,:), levels = 3, colorbarticks = 7, xlimits = [0, scale[2]+10], ylimits = [-100,scale[1]], title = "ϵ(u)_xx + displacement")
    scalarplot!(p[2,1], xgrid, view(strain_nodevals,2,:), levels = 3, colorbarticks = 7, xlimits = [0, scale[2]+10], ylimits = [-100,scale[1]], title = "ϵ(u)_yy + displacement")
    vectorplot!(p[1,1], xgrid, evaluate(PointEvaluator(Solution[1], Identity)), spacing = [50,25], clear = false)
    vectorplot!(p[2,1], xgrid, evaluate(PointEvaluator(Solution[1], Identity)), spacing = [50,25], clear = false)
    displace_mesh!(xgrid, Solution[1])
    gridplot!(p[3,1], xgrid, linewidth = 1, title = "displaced mesh")
end


function bimetal_strip2D(; scale = [1,1], n = 2, anisotropy_factor::Int = Int(ceil(scale[2]/(4*scale[1]))), ε_fix = scale[1]/(4*n))
    X=linspace(0,scale[2],(n+1)*anisotropy_factor)
    ## add some epsilon layer around material border
    if ε_fix>0.0
        yfix=[scale[1]/2-1.5*ε_fix,scale[1]/2-ε_fix/2,scale[1]/2,scale[1]/2+ε_fix/2,scale[1]/2+1.5*ε_fix]
	    Y=glue(glue(linspace(0,yfix[1],n),yfix),linspace(yfix[end],scale[1],n))
    else
        Y=linspace(0, scale[1], 2*n+1)
    end
    xgrid=simplexgrid(X,Y)
    cellmask!(xgrid,[0.0,0.0],[scale[2],scale[1]/2],1)
    cellmask!(xgrid,[0.0,scale[1]/2],[scale[2],scale[1]],2)
    bfacemask!(xgrid,[0.0,0.0],[0.0,scale[1]/2],1)
    bfacemask!(xgrid,[0.0,scale[1]/2],[0.0,scale[1]],11)
    bfacemask!(xgrid,[0.0,0.0],[scale[2],0.0],2)
    bfacemask!(xgrid,[scale[2],0.0],[scale[2],scale[1]],2)
    bfacemask!(xgrid,[0.0,scale[1]],[scale[2],scale[1]],2)
    return xgrid
end
end