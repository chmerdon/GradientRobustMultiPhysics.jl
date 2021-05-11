#= 

# 230 : Stokes Hdiv-DG (2D)
([source code](SOURCE_URL))

This example computes a velocity ``\mathbf{u}`` and pressure ``\mathbf{p}`` of the incompressible Navier--Stokes problem
```math
\begin{aligned}
- \mu \Delta \mathbf{u} + (\mathbf{u} \cdot \nabla) \mathbf{u} + \nabla p & = \mathbf{f}\\
\mathrm{div}(u) & = 0
\end{aligned}
```
with exterior force ``\mathbf{f}`` and some viscosity parameter ``\mu`` and inhomogeneous Dirichlet boundary data.

The problem will be solved by a dicontinuous Galerkin method with Hdiv-conforming ansatz space (e.g. BDM1). The normal components of the velocity are fixed by the boundary data, while
the tangential boundary fluxes are handled by the DG discretisation of the Laplacian that involves several discontinuous terms on faces ``\mathcal{F}``, i.e.

```math
\begin{aligned}
a_h(u_h,v_h) = \mu \Bigl( \int \nabla_h u_h : \nabla_h v_h dx +  \sum_{F \in \mathcal{F}} \frac{\lambda}{h_F} \int_F [[u_h]] \cdot [[v_h]] ds - \int_F {{\nabla_h u_h}} n_F \cdot [[v_h]] ds
 - \int_F [[u_h]] \cdot {{\nabla_h v_h}} n_F ds \Bigr)
\end{aligned}
```
and similar terms on the right-hand side for the inhomogeneous Dirichlet data. The qunatity ``\lambda`` is the SIP parameter.
=#

module Example230_StokesHdivDG2D

using GradientRobustMultiPhysics
using Printf

## functions that define the exact solution and the data
    function exact_pressure!(result,x::Array{<:Real,1},t::Real)
        result[1] = cos(t)*(sin(x[1])*cos(x[2]) + (cos(1) -1)*sin(1))
    end
    function user_function_velocity!(result,x::Array{<:Real,1},t::Real)
        result[1] = cos(t)*(sin(pi*x[1]-0.7)*sin(pi*x[2]+0.2))
        result[2] = cos(t)*(cos(pi*x[1]-0.7)*cos(pi*x[2]+0.2))
    end
    function exact_velogradient!(result,x::Array{<:Real,1},t::Real)
        result[1] = pi*cos(t)*(cos(pi*x[1]-0.7)*sin(pi*x[2]+0.2))
        result[2] = pi*cos(t)*(sin(pi*x[1]-0.7)*cos(pi*x[2]+0.2))
        result[3] = -pi*cos(t)*(sin(pi*x[1]-0.7)*cos(pi*x[2]+0.2))
        result[4] = -pi*cos(t)*(cos(pi*x[1]-0.7)*sin(pi*x[2]+0.2))
    end
    function rhs(nu)
        function closure!(result,x::Array{<:Real,1},t::Real)
            ## exact Laplacian
            result[1] = 2*pi*pi*nu*cos(t)*(sin(pi*x[1]-0.7)*sin(pi*x[2]+0.2))
            result[2] = 2*pi*pi*nu*cos(t)*(cos(pi*x[1]-0.7)*cos(pi*x[2]+0.2))
            ## exact pressure gradient
            result[1] += cos(t)*cos(x[1])*cos(x[2])
            result[2] -= cos(t)*sin(x[1])*sin(x[2])
        end
    end

## everything is wrapped in a main function
function main(;viscosity = 1e-3, nlevels = 5, Plotter = nothing, verbosity = 0, T = 1, lambda = 4)

    ## set log level
    set_verbosity(verbosity)

    ## FEType (Hdiv-conforming)
    FETypes = [HDIVBDM1{2}, H1P0{1}]
    
    ## initial grid
    xgrid = grid_unitsquare(Triangle2D)
    xBFaces = xgrid[BFaces]
    xFaceVolumes = xgrid[FaceVolumes]
    xFaceNormals = xgrid[FaceNormals]
    
    ## load exact flow data
    user_function_velocity = DataFunction(user_function_velocity!, [2,2]; dependencies = "XT", quadorder = 8)
    user_function_pressure = DataFunction(exact_pressure!, [1,2]; dependencies = "XT", quadorder = 4)
    user_function_velocity_gradient = DataFunction(exact_velogradient!, [4,2]; dependencies = "XT", quadorder = 4)
    user_function_rhs = DataFunction(rhs(viscosity), [2,2]; dependencies = "XT", quadorder = 8)

    ## prepare error calculation
    L2VelocityErrorEvaluator = L2ErrorIntegrator(Float64, user_function_velocity, Identity; time = T)
    L2PressureErrorEvaluator = L2ErrorIntegrator(Float64, user_function_pressure, Identity; time = T)
    H1VelocityErrorEvaluator = L2ErrorIntegrator(Float64, user_function_velocity_gradient, Gradient; time = T)

    ## load Stokes problem prototype and assign data
    Problem = IncompressibleNavierStokesProblem(2; viscosity = viscosity, nonlinear = false)
    add_rhsdata!(Problem, 1, RhsOperator(Identity, [1], user_function_rhs))

    ## add boundary data (fixes normal components of along boundary)
    add_boundarydata!(Problem, 1, [1,2,3,4], BestapproxDirichletBoundary; data = user_function_velocity)

    ## define additional operators for DG terms for Laplacian and Dirichlet data
    ## (in order of their appearance in the documentation above)
    function hdiv_laplace2_kernel(result, input, item)
        result[1] = input[1]
        result[2] = input[2]
        result .*= lambda*viscosity / xFaceVolumes[item]
        return nothing
    end
    function hdiv_laplace3_kernel(result, input, item)
        result[1] = input[1] * xFaceNormals[1,item]
        result[2] = input[1] * xFaceNormals[2,item]
        result[3] = input[2] * xFaceNormals[1,item]
        result[4] = input[2] * xFaceNormals[2,item]
        result .*= -viscosity
        return nothing
    end
    function hdiv_laplace4_kernel(result, input, item)
        result[1] = input[1] * xFaceNormals[1,item] + input[2] * xFaceNormals[2,item]
        result[2] = input[3] * xFaceNormals[1,item] + input[4] * xFaceNormals[2,item]
        result .*= -viscosity
        return nothing
    end
    HdivLaplace2 = AbstractBilinearForm([Jump(Identity), Jump(Identity)], Action(Float64, hdiv_laplace2_kernel, [2,2]; dependencies = "I", quadorder = 0); name = "nu/h_F [u] [v]", AT = ON_FACES)
    HdivLaplace3 = AbstractBilinearForm([Jump(Identity), Average(Gradient)], Action(Float64, hdiv_laplace3_kernel, [4,2]; dependencies = "I", quadorder = 0); name = "-nu [u] {grad(v)*n}", AT = ON_FACES)
    HdivLaplace4 = AbstractBilinearForm([Average(Gradient), Jump(Identity)], Action(Float64, hdiv_laplace4_kernel, [2,4]; dependencies = "I", quadorder = 0); name = "-nu {grad(u)*n} [v] ", AT = ON_FACES)

    ## additional terms for tangential part at boundary
    ## note: we use average operators here to force evaluation of all basis functions and not only of the face basis functions
    ## (which in case of Hdiv would be only the ones with nonzero normal fluxes)
    veloeval = zeros(Float64,2)
    function hdiv_boundary_kernel(result, input, x, t, item)
        eval!(veloeval, user_function_velocity, x, t)
        result[1] = input[1] * veloeval[1] + input[2] * veloeval[2]
        result[1] *= lambda*viscosity / xFaceVolumes[xBFaces[item]]
        return nothing
    end
    function hdiv_boundary_kernel2(result, input, x, t, item)
        eval!(veloeval, user_function_velocity, x, t)
        result[1] = (input[1] * xFaceNormals[1,xBFaces[item]] + input[2] * xFaceNormals[2,xBFaces[item]]) * veloeval[1]
        result[1] += (input[3] * xFaceNormals[1,xBFaces[item]] + input[4] * xFaceNormals[2,xBFaces[item]]) * veloeval[2]
        result[1] *= -viscosity
        return nothing
    end
    HdivBoundary1 = RhsOperator(Average(Identity), Action(Float64, hdiv_boundary_kernel, [1,2]; dependencies = "XTI", quadorder = user_function_velocity.quadorder); name = "- nu lambda/h_F u_D v", AT = ON_BFACES)
    HdivBoundary2 = RhsOperator(Average(Gradient), Action(Float64, hdiv_boundary_kernel2, [1,4]; dependencies = "XTI", quadorder = user_function_velocity.quadorder); name = "- nu u_D grad(v)*n", AT = ON_BFACES)

    ## assign DG operators to problem descriptions
    add_operator!(Problem, [1,1], HdivLaplace2)       
    add_operator!(Problem, [1,1], HdivLaplace3)       
    add_operator!(Problem, [1,1], HdivLaplace4)  
    add_rhsdata!(Problem, 1, HdivBoundary1)
    add_rhsdata!(Problem, 1, HdivBoundary2)

    ## show final problem description
    @show Problem

    ## loop over levels
    Results = zeros(Float64,nlevels,3); NDofs = zeros(Int,nlevels)
    for level = 1 : nlevels

        ## refine grid and update grid component references
        xgrid = uniform_refine(xgrid)
        xBFaces = xgrid[BFaces]
        xFaceVolumes = xgrid[FaceVolumes]
        xFaceNormals = xgrid[FaceNormals]

        ## generate FESpaces
        FES = [FESpace{FETypes[1]}(xgrid), FESpace{FETypes[2]}(xgrid)]

        ## generate solution vector
        Solution = FEVector{Float64}(["v_h", "p_h"],FES)

        ## solve
        solve!(Solution, Problem; time = T)

        ## plot
        GradientRobustMultiPhysics.plot(xgrid, [Solution[1], Solution[1], Solution[2]], [IdentityComponent{1}, IdentityComponent{2}, Identity]; Plotter = Plotter)

        ## compute L2 and H1 errors and save data
        NDofs[level] = length(Solution.entries)
        Results[level,1] = sqrt(evaluate(L2VelocityErrorEvaluator,Solution[1]))
        Results[level,2] = sqrt(evaluate(L2PressureErrorEvaluator,Solution[2]))
        Results[level,3] = sqrt(evaluate(H1VelocityErrorEvaluator,Solution[1]))
    end    

    ## print/show convergence history
    print_convergencehistory(NDofs, Results; X_to_h = X -> X.^(-1/2), labels = ["|| u - u_h ||", "|| p - p_h ||", "|| ∇(u - u_h) ||"])
    plot_convergencehistory(NDofs, Results; add_h_powers = [1,2], X_to_h = X -> X.^(-1/2), Plotter = Plotter, labels = ["|| u - u_h ||", "|| p - p_h ||", "|| ∇(u - u_h) ||"])
end
end