using Triangulate

function testgrid_mixedEG()

    xgrid=ExtendableGrid{Float64,Int32}()
    xgrid[Coordinates]=Array{Float64,2}([0 0; 4//10 0; 1 0; 0 6//10; 4//10 6//10; 1 6//10;0 1; 4//10 1; 1 1]')
    xCellNodes=VariableTargetAdjacency(Int32)
    xCellGeometries=[Triangle2D, Triangle2D, Parallelogram2D, Parallelogram2D, Triangle2D, Triangle2D];
    
    append!(xCellNodes,[1,5,4])
    append!(xCellNodes,[1,2,5])
    append!(xCellNodes,[2,3,6,5])
    append!(xCellNodes,[4,5,8,7]) 
    append!(xCellNodes,[5,6,9])
    append!(xCellNodes,[8,5,9])

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


function testgrid_cookmembrane()
    triin=Triangulate.TriangulateIO()
    triin.pointlist=Matrix{Cdouble}([0 0; 0 -44; 48 0; 48 16]');
    triin.segmentlist=Matrix{Cint}([1 2 ; 2 3 ; 3 4 ; 4 1 ]')
    triin.segmentmarkerlist=Vector{Int32}([1, 2, 3, 4])

    xgrid = simplexgrid("pALVa1000.0", triin)
    
    # why do I have to do these things below ?
    xgrid[CellRegions] = VectorOfConstants(Int32(1),num_sources(xgrid[CellNodes]))
    xgrid[CellGeometries] = VectorOfConstants(Triangle2D,num_sources(xgrid[CellNodes]))
    xgrid[BFaceGeometries] = VectorOfConstants(Edge1D,num_sources(xgrid[BFaceNodes]))
    xgrid[BFaceNodes] = xgrid[BFaceNodes][[2,1],:]

    return xgrid
end


function testgrid_tire(k::Int = 1, S::Int = 2)
    
    N = 4^k # number of points on circle

    r1 = 1.0 # outer tire radius
    r2 = 0.9 # innter tire radius
    r3 = 0.1 # innter circle radius


    xgrid=ExtendableGrid{Float64,Int32}()

    npoints = 3*N + 1# = three circles
    xCoordinates = zeros(Float64,2,npoints)
    for j = 1 : N
        xCoordinates[1,j] = r1*cos(2*pi*j/N)
        xCoordinates[2,j] = r1*sin(2*pi*j/N)
        xCoordinates[1,N+j] = r2*cos(2*pi*j/N)
        xCoordinates[2,N+j] = r2*sin(2*pi*j/N)
        xCoordinates[1,2*N+j] = r3*cos(2*pi*j/N)
        xCoordinates[2,2*N+j] = r3*sin(2*pi*j/N)
    end
    xCoordinates[:,end] = [0.01,0.01]
    xgrid[Coordinates]=xCoordinates


    ntriangles = 3*N
    nspokes = Int(ceil(N/S))

    xCellNodes=VariableTargetAdjacency(Int32)
    xCellRegions=ones(Int32,ntriangles+nspokes)
    xCellGeometries=Array{DataType,1}(undef,ntriangles+nspokes)
    for j = 1 : ntriangles
        xCellGeometries[j] = Triangle2D
    end
    for j = 1 : N
        if j==N
            append!(xCellNodes,[j 1 N+1])
            append!(xCellNodes,[j N+1 N+j])
            append!(xCellNodes,[2*N+j 2*N+1 3*N+1])
        else
            append!(xCellNodes,[j j+1 N+j+1])
            append!(xCellNodes,[j N+j+1 N+j])
            append!(xCellNodes,[2*N+j 2*N+j+1 3*N+1])
        end
    end
    for j = 1 : nspokes
        xCellGeometries[ntriangles+j] = Edge1D
        append!(xCellNodes,[N+j*S,2*N+j*S])
        xCellRegions[ntriangles+j] = 2
    end
    xgrid[CellNodes] = xCellNodes
    xgrid[CellGeometries] = xCellGeometries
    ncells = num_sources(xCellNodes)
    xgrid[CellRegions]=xCellRegions



    nbfaces = 3*N+nspokes
    xBFaceRegions=zeros(Int32,nbfaces)
    xBFaceNodes=zeros(Int32,2,nbfaces)
    for j = 1 : N
        xBFaceRegions[j] = 1
        xBFaceRegions[N+j] = 2
        xBFaceRegions[2*N+j] = 3
        if j == N
            xBFaceNodes[:,j] = [j, 1]
            xBFaceNodes[:,N+j] = [N+1,N+j]
            xBFaceNodes[:,2*N+j] = [2*N+j,2*N+1]
        else
            xBFaceNodes[:,j] = [j, j+1]
            xBFaceNodes[:,N+j] = [N+j+1,N+j]
            xBFaceNodes[:,2*N+j] = [2*N+j,2*N+j+1]
        end
    end
    for j = 1 : nspokes
        xBFaceNodes[:,3*N+j] = [N+j*S,2*N+j*S]
        xBFaceRegions[3*N+j] = 4
    end
    xgrid[BFaceNodes]=xBFaceNodes
    xgrid[BFaceRegions]=xBFaceRegions
    xgrid[BFaceGeometries]=VectorOfConstants(Edge1D,nbfaces)

    xgrid[CoordinateSystem]=Cartesian2D


    return xgrid
end