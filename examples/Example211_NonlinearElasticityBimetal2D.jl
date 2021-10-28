#= 

# 211 : Nonlinear Elasticity Bimetal 2D
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

Note: For the mesh generation the additional packages SimplexGridFactory and Triangulate are needed.
=#

module Example211_NonlinearElasticityBimetal2D

using GradientRobustMultiPhysics
using ExtendableGrids
using SimplexGridFactory
using Triangulate
using GridVisualize

## kernel for nonlinear operator
function nonlinear_operator_kernel!(λ,μ,ΔT,α)

    ## thermal misfit strain
    ϵT::Float64 = ΔT*α

    function closure(result, input)
        ## input = grad(u) written as a vector
        ## compute linear part of strain, subtract thermal strain (all in Voigt notation)
        result[1] = input[1] - ϵT
        result[2] = input[4] - ϵT
        result[3] = input[2] + input[3]

        ## add nonlinear part of the strain 1/2 * (grad(u)'*grad(u))
        result[1] += 1//2 * (input[1]^2 + input[3]^2)
        result[2] += 1//2 * (input[2]^2 + input[4]^2)
        result[3] += input[1]*input[2] + input[3]*input[4]

        ## multiply with isotropic stress tensor
        ## (stored in input[5:7] using Voigt notation)
        input[5] = λ*(result[1]+result[2]) + 2*μ*result[1]
        input[6] = λ*(result[1]+result[2]) + 2*μ*result[2]
        input[7] = 2*μ*result[3]
    
        ## write strain into result
        result[1] = input[5]
        result[2] = input[7]
        result[3] = input[7]
        result[4] = input[6]

        return nothing
    end
    return closure
end 

## everything is wrapped in a main function
function main(; ν = [0.3,0.3], E = [2.1,1.1], ΔT = [580,580], α = [1.3e-5,2.4e-5], scale = [20,500], nrefinements = 1, material_border = 0.5, verbosity = 0, Plotter = nothing)

    ## set log level
    set_verbosity(verbosity)

    ## compute Lame' coefficients μ and λ from ν and E
    μ = E ./ (2  .* (1 .+ ν.^(-1)))
    λ = E .* ν ./ ( (1 .- 2*ν) .* (1 .+ ν))

    ## generate bimetal mesh
    xgrid = bimetal_strip2D(; material_border = material_border, scale = scale)
    xgrid = uniform_refine(xgrid,nrefinements)

    ## prepare nonlinear operator (one for each bimetal region)
    nonlin_operator_1 = GenerateNonlinearForm("C(ϵ(u)-ϵT):∇v", [Gradient], [1], Gradient, nonlinear_operator_kernel!(λ[1],μ[1],ΔT[1],α[1]), [4,4,7]; regions = [1], quadorder = 3, ADnewton = true) 
    nonlin_operator_2 = GenerateNonlinearForm("C(ϵ(u)-ϵT):∇v", [Gradient], [1], Gradient, nonlinear_operator_kernel!(λ[2],μ[2],ΔT[2],α[2]), [4,4,7]; regions = [2], quadorder = 3, ADnewton = true) 
    
    ## generate problem description and assign nonlinear operators
    Problem = PDEDescription("nonlinear elasticity problem")
    add_unknown!(Problem; unknown_name = "u", equation_name = "displacement equation")
    add_operator!(Problem, 1, nonlin_operator_1)
    add_operator!(Problem, 1, nonlin_operator_2)
    add_boundarydata!(Problem, 1, [1,11], HomogeneousDirichletBoundary)
    @show Problem

    ## create finite element space and solution vector
    FEType = H1P2{2,2}
    FES = FESpace{FEType}(xgrid)
    Solution = FEVector{Float64}("u_h",FES)

    ## solve
    solve!(Solution, Problem; maxiterations = 10, target_residual = 1e-9, show_statistics = true)

    ## displace mesh and plot
    p = GridVisualizer(; Plotter = Plotter, layout = (2,1), clear = true, resolution = (800,600))
    scalarplot!(p[1,1], xgrid, view(nodevalues(Solution[1]; abs = true),1,:), levels = 0, colorbarticks = 7)
    vectorplot!(p[1,1], xgrid, evaluate(PointEvaluator(Solution[1], Identity)), spacing = [50,25], xlimits = [0, scale[2]+10], ylimits = [-100,scale[1]], clear = false, title = "u_h (abs + quiver)")
    displace_mesh!(xgrid, Solution[1])
    gridplot!(p[2,1], xgrid, linewidth = "1", title = "displaced mesh")
end


function bimetal_strip2D(; material_border = 0.5, scale = [1,1], maxvol = prod(scale)/4)
    builder=SimplexGridBuilder(Generator=Triangulate)
    @info "Generating 2d bimetal grid for scale = $scale"

    p1=point!(builder,0,0)                                                
    p2=point!(builder,scale[2],0)                                         
    p3=point!(builder,scale[2],scale[1])                                 
    p4=point!(builder,0,scale[1])                                      
    p5=point!(builder,0,material_border*scale[1])                         
    p6=point!(builder,scale[2],material_border*scale[1])

    facetregion!(builder,1) # left (material A)
    facet!(builder,p5 ,p1)
    facetregion!(builder,11) # left (material B)
    facet!(builder,p4,p5)
    facetregion!(builder,2) # bottom
    facet!(builder,p1 ,p2)
    facetregion!(builder,3) # top
    facet!(builder,p3, p4)
    facetregion!(builder,4) # right
    facet!(builder,p2, p6)
    facet!(builder,p6, p3)
    facetregion!(builder,99) # interior facet to split regions
    facet!(builder,p5, p6)

    cellregion!(builder,1)
    maxvolume!(builder,maxvol)
    regionpoint!(builder,(0.5*scale[2],0.5*material_border*scale[1]))
    cellregion!(builder,2)
    maxvolume!(builder,maxvol)
    regionpoint!(builder,(0.5*scale[2],0.5*(material_border+1)*scale[1]))

    xgrid = simplexgrid(builder)
    return xgrid
end

end