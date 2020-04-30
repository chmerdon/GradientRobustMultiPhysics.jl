
using FEXGrid
using ExtendableGrids
using ExtendableSparse
using FiniteElements
using FEOperator
using QuadratureRules
using VTKView
#ENV["MPLBACKEND"]="qt5agg"
#using PyPlot
using BenchmarkTools

function gridgen_mixedEG()

    NumberType = Float64
    xgrid=ExtendableGrid{NumberType,Int32}()
    xgrid[Coordinates]=Array{NumberType,2}([0 0; 4//10 0; 1 0; 0 6//10; 4//10 6//10; 1 6//10;0 1; 4//10 1; 1 1]')
    xCellNodes=VariableTargetAdjacency(Int32)
    xCellGeometries=[Triangle2D, Triangle2D, Parallelogram2D, Parallelogram2D, Triangle2D, Triangle2D];
    
    append!(xCellNodes,[1,5,4])
    append!(xCellNodes,[1,2,5])
    append!(xCellNodes,[2,3,6,5])
    append!(xCellNodes,[4,5,8,7]) 
    append!(xCellNodes,[5,6,8])
    append!(xCellNodes,[8,6,9])

    xgrid[CellNodes] = xCellNodes
    xgrid[CellGeometries] = xCellGeometries
    ncells = num_sources(xCellNodes)
    xgrid[CellRegions]=VectorOfConstants{Int32}(1,ncells)
    xgrid[BFaceRegions]=Array{Int32,1}([1,1,2,2,3,3,4,4])
    xBFaceNodes=Array{Int32,2}([1 2; 2 3; 3 6; 6 9; 9 8; 8 7; 7 4; 4 1]')
    xgrid[BFaceNodes]=xBFaceNodes
    nbfaces = num_sources(xBFaceNodes)
    xgrid[BFaceGeometries]=VectorOfConstants(Edge1D,nbfaces)
    xgrid[CoordinateSystem]=Cartesian2D

    return xgrid
end


function main()

    nlevels = 5

    xgrid = gridgen_mixedEG()

    L2error = []
    H1error = []
    for level = 1 : nlevels
        xgrid = uniform_refine(xgrid)
        
        # solve Poisson problem
        function exact_solution!(result,x)
            result[1] = x[1]*x[2]*(x[1]-1)*(x[2]-1)
        end    
        function exact_solution_gradient!(result,x)
            result[1] = x[2]*(2*x[1]-1)*(x[2]-1)
            result[2] = x[1]*(2*x[2]-1)*(x[1]-1)
        end    
        function exact_solution_laplacian!(result,x)
            result[1] = 2*x[2]*(x[2]-1) + 2*x[1]*(x[1]-1)
        end    

        # generate FE
        FE = FiniteElements.getH1P1FiniteElement(xgrid,1)
        FiniteElements.show_new(FE)

        # compute stiffness matrix
        action = MultiplyMatrixAction([1.0 0.0;0.0 1.0])
        A = StiffnessMatrix!(FE, action; talkative = true)

        # compute right hand side
        function rhs_function(result,input,x)
            exact_solution_laplacian!(result,x)
            result[1] = -result[1]*input[1] 
        end    
        action = XFunctionAction(rhs_function,1)
        b = RightHandSide!(FE, action; talkative = true, bonus_quadorder = 2)[:]
        
        # apply homogeneous boundary data
        bdofs = []
        xBFaces = xgrid[BFaces]
        nbfaces = length(xBFaces)
        xFaceDofs = FE.FaceDofs
        for bface = 1 : nbfaces
            append!(bdofs,xFaceDofs[:,xBFaces[bface]])
        end
        bdofs = unique(bdofs)
        for j = 1 : length(bdofs)
            b[bdofs[j]] = 0.0
            A[bdofs[j],bdofs[j]] = 1e60
        end

        # solve
        Solution = FEFunction{Float64}("solution",FE,A\b)

        # compute L2 and H1 error
        function L2error_function(result,input,x)
            exact_solution!(result,x)
            result[1] = (result[1] - input[1])^2
        end    
        L2error_action = XFunctionAction(L2error_function,1)
        error4cell = zeros(Float64,num_sources(xgrid[CellNodes]),1)
        assemble!(error4cell, ItemIntegrals, AbstractAssemblyTypeCELL, Identity, Solution, L2error_action; talkative = true, bonus_quadorder = 5)
        append!(L2error, sqrt(sum(error4cell[:])))

        function H1error_function(result,input,x)
            exact_solution_gradient!(result,x)
            result[1] = (result[1] - input[1])^2 + (result[2] - input[2])^2
            result[2] = 0 
        end    
        H1error_action = XFunctionAction(H1error_function,2)
        error4cell = zeros(Float64,num_sources(xgrid[CellNodes]),2)
        assemble!(error4cell, ItemIntegrals, AbstractAssemblyTypeCELL, Gradient, Solution, H1error_action; talkative = true, bonus_quadorder = 3)
        append!(H1error, sqrt(sum(error4cell[:])))
    end    

    println("\nL2error")
    Base.show(L2error)

    println("\nH1error")
    Base.show(H1error)

end


main()
