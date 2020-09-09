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

This examples highlights the use of automatic differentation to obtain Newton derivatives of nonlinear PDEOperators.

=#

module Example_2DLidDrivenCavityADNewton

using GradientRobustMultiPhysics
using Printf
using ExtendableGrids
using Triangulate

## data
function boundary_data_top!(result,x)
    result[1] = 1.0;
    result[2] = 0.0;
end

## grid generator that generates  unstructured simplex mesh
function grid_square(maxarea::Float64)
    triin=Triangulate.TriangulateIO()
    triin.pointlist=Matrix{Cdouble}([-1 -1; 1 -1; 1 1; -1 1]');
    triin.segmentlist=Matrix{Cint}([1 2 ; 2 3 ; 3 4 ; 4 1 ]')
    triin.segmentmarkerlist=Vector{Int32}([1, 2, 3, 4])
    xgrid = simplexgrid("pALVa$(@sprintf("%.15f",maxarea))", triin)
    xgrid[CellRegions] = VectorOfConstants(Int32(1),num_sources(xgrid[CellNodes]))
    xgrid[CellGeometries] = VectorOfConstants(Triangle2D,num_sources(xgrid[CellNodes]))
    return xgrid
end

## everything is wrapped in a main function
function main(; verbosity = 1, Plotter = nothing)
    ## grid
    xgrid = grid_square(1e-3)

    ## problem parameters
    viscosity = 1e-2
    ADnewton = true # use automatic differentation to setup Newton (otherwise hardcoded Newton terms)
    maxIterations = 50  # termination criterion 1 for nonlinear mode
    maxResidual = 1e-12 # termination criterion 2 for nonlinear mode

    barycentric_refinement = false # do not change
    ## choose one of these (inf-sup stable) finite element type pairs
    #FETypes = [H1P2{2,2}, H1P1{1}] # Taylor--Hood
    #FETypes = [H1CR{2}, L2P0{1}] # Crouzeix--Raviart
    #FETypes = [H1MINI{2,2}, H1P1{1}] # MINI element on triangles only
    #FETypes = [H1MINI{2,2}, H1CR{1}] # MINI element on triangles/quads
    FETypes = [H1BR{2}, L2P0{1}] # Bernardi--Raugel
    #FETypes = [H1P2{2,2}, L2P1{1}]; barycentric_refinement = true # Scott-Vogelius 

    #####################################################################################    
    #####################################################################################

    ## load linear Stokes problem prototype and assign data
    ## we are adding the nonlinar convection term below
    StokesProblem = IncompressibleNavierStokesProblem(2; viscosity = viscosity, nonlinear = false)
    add_boundarydata!(StokesProblem, 1, [1,2,4], HomogeneousDirichletBoundary)
    add_boundarydata!(StokesProblem, 1, [3], BestapproxDirichletBoundary; data = boundary_data_top!, bonus_quadorder = 0)

    ## store matrix of Laplace operator for nonlinear solver
    StokesProblem.LHSOperators[1,1][1].store_operator = true   

    ## add Newton for convection term
    if ADnewton
        ## AUTOMATIC DIFFERENTATION
        ## requries AD kernel function for NonlinearForm action (u * grad) u
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

        ## generate and add nonlinear PDEOperator (modifications to RHS are taken care of automatically)
        NLConvectionOperator = GenerateNonlinearForm("(u * grad) u  * v", [Identity, Gradient], [1,1], Identity, ugradu_kernel_AD, [2, 6], 2; ADnewton = true)            
        add_operator!(StokesProblem, [1,1], NLConvectionOperator)
    else
        ## MANUAL DIFFERENTATION
        ## uses the following kernel function
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
        ## and a similar kernel function for the RHS
        function ugradu_kernel_rhs(result, input_current, input_ansatz)
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

        ## generate and add nonlinear PDEOperator (modifications to RHS are taken care of by optional arguments)
        NLConvectionOperator = GenerateNonlinearForm("(u * grad) u  * v", [Identity, Gradient], [1,1], Identity, ugradu_kernel_nonAD, [2, 6], 2; ADnewton = false, action_kernel_rhs = ugradu_kernel_rhs)            
        add_operator!(StokesProblem, [1,1], NLConvectionOperator)         
    end

    ## uniform mesh refinement, in case of Scott-Vogelius we use barycentric refinement
    if barycentric_refinement == true
        xgrid = barycentric_refine(xgrid)
    end

    ## generate FESpaces
    FESpaceVelocity = FESpace{FETypes[1]}(xgrid)
    FESpacePressure = FESpace{FETypes[2]}(xgrid)
    Solution = FEVector{Float64}("Stokes velocity",FESpaceVelocity)
    append!(Solution,"Stokes pressure",FESpacePressure)

    ## show configuration and solve Stokes problem
    Base.show(StokesProblem)
    solve!(Solution, StokesProblem; verbosity = verbosity, maxIterations = maxIterations, maxResidual = maxResidual)

    ## plot
    GradientRobustMultiPhysics.plot(Solution, [1,2], [Identity, Identity]; Plotter = Plotter, verbosity = verbosity, use_subplots = true)

end

end