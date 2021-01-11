#= 

# 2D Lid-driven cavity (AD-Newton)
([source code](SOURCE_URL))

This example solves the lid-driven cavity problem where one seeks
a velocity ``\mathbf{u}`` and pressure ``\mathbf{p}`` of the incompressible Navier--Stokes problem
```math
\begin{aligned}
- \mu \Delta \mathbf{u} + (\mathbf{u} \cdot \nabla) \mathbf{u} + \nabla p & = 0\\
\mathrm{div}(u) & = 0
\end{aligned}
```
where ``\mathbf{u} = (1,0)`` along the top boundary of a square domain.

This examples highlights the use of automatic differentation to obtain Newton derivatives of nonlinear PDEOperators. The user can 
switch between a Newton obtained by automatic differentation (ADnewton = true) and a 'manual' Newton scheme (ADnewton = false). The runtime
of the manual newton is slightly faster, but requires much more code input from the user that is more prone to errors.

=#

module Example_2DLidDrivenCavityADNewton

using GradientRobustMultiPhysics
using ExtendableGrids
using Printf

## data
function boundary_data_top!(result)
    result[1] = 1;
    result[2] = 0;
end

## everything is wrapped in a main function
function main(; verbosity = 2, Plotter = nothing, ADnewton = true)

    ## grid
    xgrid = uniform_refine(grid_unitsquare(Triangle2D), 6);

    ## problem parameters
    viscosity = 1e-2
    maxIterations = 50  # termination criterion 1 for nonlinear mode
    maxResidual = 1e-12 # termination criterion 2 for nonlinear mode
    broken_p = false

    ## choose one of these (inf-sup stable) finite element type pairs
    #FETypes = [H1P2{2,2}, H1P1{1}] # Taylor--Hood
    #FETypes = [H1P2B{2,2}, H1P1{1}]; broken_p = true # P2-bubble
    #FETypes = [H1CR{2}, H1P0{1}] # Crouzeix--Raviart
    #FETypes = [H1MINI{2,2}, H1P1{1}] # MINI element on triangles only
    FETypes = [H1BR{2}, H1P0{1}]; broken_p = true # Bernardi--Raugel

    #####################################################################################    
    #####################################################################################

    ## negotiate data functions to the package
    user_function_bnd = DataFunction(boundary_data_top!, [2,2]; name = "u_bnd", dependencies = "", quadorder = 0)

    ## load linear Stokes problem prototype and assign data
    ## we are adding the nonlinar convection term ourself below
    ## to discuss the details
    StokesProblem = IncompressibleNavierStokesProblem(2; viscosity = viscosity, nonlinear = false)
    add_boundarydata!(StokesProblem, 1, [1,2,4], HomogeneousDirichletBoundary)
    add_boundarydata!(StokesProblem, 1, [3], BestapproxDirichletBoundary; data = user_function_bnd)

    ## store matrix of Laplace operator for nonlinear solver
    StokesProblem.LHSOperators[1,1][1].store_operator = true   

    ## add Newton for convection term
    if ADnewton
        ## AUTOMATIC DIFFERENTATION
        ## requries kernel function for NonlinearForm action (u * grad) u
        function ugradu_kernel_AD(result, input)
            ## input = [u, grad(u)]
            ## compute (u * grad) u = grad(u)*u
            for j = 1 : 2
                result[j] = 0.0
                for k = 1 : 2
                    result[j] += input[k]*input[2 + (j-1)*2+k]
                end
            end
            return nothing
        end 
        action_kernel = ActionKernel(ugradu_kernel_AD, [2,6]; dependencies = "", quadorder = 1)

        ## generate and add nonlinear PDEOperator (modifications to RHS are taken care of automatically)
        NLConvectionOperator = GenerateNonlinearForm("(u * grad) u  * v", [Identity, Gradient], [1,1], Identity, action_kernel; ADnewton = true)            
        add_operator!(StokesProblem, [1,1], NLConvectionOperator)
    else
        ## MANUAL DIFFERENTATION
        ## uses the following kernel function for the linearised convection operator
        function ugradu_kernel_nonAD(result, input_current, input_ansatz)
            ## input_current = [current, grad(current)]
            ## input_ansatz = [ansatz, grad(ansatz)]
            ## compute (current * grad) ansatz + (current * grad) ansatz
            for j = 1 : 2
                result[j] = 0.0
                for k = 1 : 2
                    result[j] += input_current[k]*input_ansatz[2 + (j-1)*2+k]
                    result[j] += input_ansatz[k]*input_current[2 + (j-1)*2+k]
                end
            end
            return nothing
        end
        ## and a similar kernel function for the right-hand side
        function ugradu_kernel_rhs(result, input_current)
            ## input_current = [current, grad(current)]
            ## input_ansatz = [ansatz, grad(ansatz)]
            ## compute (current * grad) current
            for j = 1 : 2
                result[j] = 0.0
                for k = 1 : 2
                    result[j] += input_current[k]*input_current[2 + (j-1)*2+k]
                end
            end
            return nothing
        end
        newton_action_kernel = NLActionKernel(ugradu_kernel_nonAD, [2,6]; dependencies = "", quadorder = 1)
        action_kernel_rhs = ActionKernel(ugradu_kernel_rhs, [2,6]; dependencies = "", quadorder = 1)

        ## generate and add nonlinear PDEOperator (modifications to RHS are taken care of by optional arguments)
        NLConvectionOperator = GenerateNonlinearForm("(u * grad) u  * v", [Identity, Gradient], [1,1], Identity, newton_action_kernel; ADnewton = false, action_kernel_rhs = action_kernel_rhs)            
        add_operator!(StokesProblem, [1,1], NLConvectionOperator)         
    end

    ## generate FESpaces
    FESpaceVelocity = FESpace{FETypes[1]}(xgrid)
    FESpacePressure = FESpace{FETypes[2]}(xgrid; broken = broken_p)
    Solution = FEVector{Float64}("Stokes velocity",FESpaceVelocity)
    append!(Solution,"Stokes pressure",FESpacePressure)

    ## show configuration and solve Stokes problem
    Base.show(StokesProblem)
    solve!(Solution, StokesProblem; verbosity = verbosity, maxIterations = maxIterations, maxResidual = maxResidual)

    ## plot
    GradientRobustMultiPhysics.plot(Solution, [1,2], [Identity, Identity]; Plotter = Plotter, verbosity = verbosity)

end

end