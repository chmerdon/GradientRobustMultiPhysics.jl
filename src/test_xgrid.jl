
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


function main()

    NumberType = Float64
    mixed_geometries = true

    xgrid=ExtendableGrid{NumberType,Int32}()
    xgrid[Coordinates]=Array{NumberType,2}([0 0; 4//10 0; 1 0; 0 6//10; 4//10 6//10; 1 6//10;0 1; 4//10 1; 1 1]')
    xCellNodes=VariableTargetAdjacency(Int32)
    if mixed_geometries
        xCellGeometries=[Triangle2D, Triangle2D, Parallelogram2D, Parallelogram2D, Triangle2D, Triangle2D];
    else
        xCellGeometries=VectorOfConstants(Triangle2D,8)
    end    

    append!(xCellNodes,[1,5,4])
    append!(xCellNodes,[1,2,5])

    if mixed_geometries
        append!(xCellNodes,[2,3,6,5])
        append!(xCellNodes,[4,5,8,7])
    else    
        append!(xCellNodes,[4,5,8])
        append!(xCellNodes,[4,8,7])
        append!(xCellNodes,[2,3,5])
        append!(xCellNodes,[5,3,6])
    end    

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

    function f(result,x)
        result[1] = x[1] - 1//2
    end    

    # integral of f over cells
    cellvalues = zeros(Real,num_sources(xgrid[CellNodes]),1)
    @time integrate!(cellvalues, xgrid, AbstractAssemblyTypeCELL, f, 1, 1, NumberType; talkative = true)
    show(sum(cellvalues, dims = 1))

    # integral of f over faces
    facevalues = zeros(Real,num_sources(xgrid[FaceNodes]),1)
    @time integrate!(facevalues, xgrid, AbstractAssemblyTypeFACE, f, 1, 1, NumberType; talkative = true)
    show(sum(facevalues[xgrid[BFaces]], dims = 1))

    # integral of f over bfaces
    bfacevalues = zeros(Real,num_sources(xgrid[BFaceNodes]),1)
    @time integrate!(facevalues, xgrid, AbstractAssemblyTypeBFACE, f, 1, 1, NumberType; talkative = true)
    show(sum(bfacevalues, dims = 1))


    # solve Poisson problem
    function exact_solution!(result,x)
        result[1] = x[1]*x[2]*(x[1]-1)*(x[2]-1)
    end    
    function exact_solution_laplacian!(result,x)
        result[1] = 2*x[2]*(x[2]-1) + 2*x[1]*(x[1]-1)
    end    


    println("")
    FE = FiniteElements.getH1P1FiniteElement(xgrid,1)
    ndofs = FE.ndofs
    FiniteElements.show_new(FE)

    # stiffness matrix
    action = MultiplyMatrixAction([1.0 0.0;0.0 1.0])
    A = ExtendableSparseMatrix{NumberType,Int32}(ndofs,ndofs)
    @time StiffnessMatrix!(A,FE,action)


    # righ hand side
    b = zeros(NumberType,FE.ndofs,1)
    function rhs_function(result,input,x)
        exact_solution_laplacian!(result,x)
        result[1] = -result[1]*input[1] 
    end    
    action = XFunctionAction(rhs_function,1)
    @time assemble!(b, LinearForm, AbstractAssemblyTypeCELL, Identity, FE, action; talkative = true, bonus_quadorder = 2)
    
    # homogeneous boundary data
    bdofs = []
    xBFaces = xgrid[BFaces]
    nbfaces = length(xBFaces)
    xFaceDofs = FE.FaceDofs
    for bface = 1 : nbfaces
        append!(bdofs,xFaceDofs[:,xBFaces[bface]])
    end
    bdofs = unique(bdofs)
    
    # fix boundary dofs
    for j = 1 : length(bdofs)
        b[bdofs[j]] = 0.0
        A[bdofs[j],bdofs[j]] = 1e60
    end
    x = A\b

    Solution = FEVector{Float64}("solution",FE,x[:])

    # compute L2 error
    function L2error(result,input,x)
        exact_solution!(result,x)
        result[1] = (result[1] - input[1])^2
    end    
    L2error_action = XFunctionAction(L2error,1)
    error4cell = zeros(Float64,ncells)
    @time assemble!(error4cell, ItemIntegrals, AbstractAssemblyTypeCELL, Identity, Solution, L2error_action; talkative = true, bonus_quadorder = 4)
    
    show(sum(error4cell[:]))

    #xgrid = split_grid_into(xgrid,Triangle2D)
    #ExtendableGrids.plot(xgrid; Plotter = VTKView)

end


main()
