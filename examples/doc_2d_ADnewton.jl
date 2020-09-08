#= 

# 2D Lid-driven cavity (Navier--Stokes + Newton)
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

using GradientRobustMultiPhysics
using ExtendableGrids
using Triangulate
using DiffResults
using ForwardDiff
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
using Printf


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
function main()
    #####################################################################################
    #####################################################################################

    ## grid
    xgrid = grid_square(1e-3)

    ## problem parameters
    viscosity = 1e-2
    barycentric_refinement = false # do not change
    maxIterations = 50  # termination criterion 1 for nonlinear mode
    maxResidual = 1e-10 # termination criterion 2 for nonlinear mode
    ADnewton = true # use Newton iteration for nonlinearity

    ## choose one of these (inf-sup stable) finite element type pairs
    ReconstructionOperator = Identity
    #FETypes = [H1P2{2,2}, H1P1{1}] # Taylor--Hood
    #FETypes = [H1CR{2}, L2P0{1}] # Crouzeix--Raviart
    #FETypes = [H1MINI{2,2}, H1P1{1}] # MINI element on triangles only
    #FETypes = [H1MINI{2,2}, H1CR{1}] # MINI element on triangles/quads
    FETypes = [H1BR{2}, L2P0{1}] # Bernardi--Raugel
    #FETypes = [H1P2{2,2}, L2P1{1}]; barycentric_refinement = true # Scott-Vogelius 

    ## postprocess parameters
    plot_grid = false
    plot_pressure = false
    plot_velocity = true

    #####################################################################################    
    #####################################################################################

    ## load linear Stokes problem prototype and assign data
    StokesProblem = IncompressibleNavierStokesProblem(2; viscosity = viscosity, nonlinear = false)
    add_boundarydata!(StokesProblem, 1, [1,2,4], HomogeneousDirichletBoundary)
    add_boundarydata!(StokesProblem, 1, [3], BestapproxDirichletBoundary; data = boundary_data_top!, bonus_quadorder = 0)

    ## uniform mesh refinement
    ## in case of Scott-Vogelius we use barycentric refinement
    if barycentric_refinement == true
        xgrid = barycentric_refine(xgrid)
    end

    ## generate FESpaces
    FESpaceVelocity = FESpace{FETypes[1]}(xgrid)
    FESpacePressure = FESpace{FETypes[2]}(xgrid)
    Solution = FEVector{Float64}("Stokes velocity",FESpaceVelocity)
    append!(Solution,"Stokes pressure",FESpacePressure)


    ## set nonlinear options and Newton terms
    ## store matrix of Laplace operator for nonlinear solver
    StokesProblem.LHSOperators[1,1][1].store_operator = true   

    ## add Newton for convection term
    if ADnewton
        ## AUTOMATIC DIFFERENTATION
        ## requries AD kernel function for NonlinearForm action (u * grad) u
        function ugradu_kernel_AD(result, input)
            # input = [u, grad(u)]
            # compute (u * grad) u = grad(u)*u
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
            # input_current = [current, grad(current)]
            # input_ansatz = [ansatz, grad(ansatz)]
            # compute (current * grad) ansatz + (current * grad) ansatz
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
            # input_current = [current, grad(current)]
            # input_ansatz = [ansatz, grad(ansatz)]
            # compute (current * grad) current
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
    Base.show(StokesProblem)

    ## solve Stokes problem
    solve!(Solution, StokesProblem; verbosity = 1, maxIterations = maxIterations, maxResidual = maxResidual)

    
    ## split grid into triangles for plotter
    xgrid = split_grid_into(xgrid,Triangle2D)
    xCoordinates = xgrid[Coordinates]
    nnodes = size(xgrid[Coordinates],2)

    ## plot triangulation
    if plot_grid
        PyPlot.figure("grid")
        ExtendableGrids.plot(xgrid, Plotter = PyPlot)
    end

    ## plot pressure
    if plot_pressure
        nodevals = zeros(Float64,1,nnodes)
        PyPlot.figure("pressure")
        nodevalues!(nodevals,Solution[2],FESpacePressure)
        ExtendableGrids.plot(xgrid, nodevals[1,:]; Plotter = PyPlot)
    end

    ## plot velocity (speed + quiver)
    if plot_velocity
        nodevals = zeros(Float64,2,nnodes)
        nodevalues!(nodevals,Solution[1],FESpaceVelocity)
        PyPlot.figure("velocity")
        ExtendableGrids.plot(xgrid, sqrt.(nodevals[1,:].^2+nodevals[2,:].^2); Plotter = PyPlot, isolines = 3)
        quiver(xCoordinates[1,:],xCoordinates[2,:],nodevals[1,:],nodevals[2,:])
    end
end


main()
