
struct CellFinder{Tv,Ti,EG,CS}
    xgrid::ExtendableGrid{Tv,Ti}
    xCellFaces::GridAdjacencyTypes
    xFaceCells::GridAdjacencyTypes
    node2oppositeface::Array{Int,1}
    previous_cells::Array{Int,1}
    L2G::L2GTransformer{Tv,EG,CS}
    invA::Matrix{Tv}
    xreftest::Array{Tv,1}
    cx::Vector{Tv}
end

function postprocess_xreftest!(CF::CellFinder{Tv,Ti,EG,CS}) where{Tv,Ti,EG <: AbstractElementGeometry, CS}
    CF.xreftest[end] = 1 - sum(CF.xreftest[1:end-1])
end

function postprocess_xreftest!(CF::CellFinder{Tv,Ti,EG,CS}) where{Tv,Ti,EG <: Parallelogram2D, CS}
    CF.xreftest[3] = 1 - CF.xreftest[1]
    CF.xreftest[4] = 1 - CF.xreftest[2]
end

function postprocess_xreftest!(CF::CellFinder{Tv,Ti,EG,CS}) where{Tv,Ti,EG <: Parallelepiped3D, CS}
    CF.xreftest[4] = 1 - CF.xreftest[1]
    CF.xreftest[5] = 1 - CF.xreftest[2]
    CF.xreftest[6] = 1 - CF.xreftest[3]
end


function CellFinder(xgrid::ExtendableGrid{Tv,Ti}, EG) where {Tv,Ti}
    CS = xgrid[CoordinateSystem]
    L2G = L2GTransformer{Tv,EG,CS}(xgrid, ON_CELLS)
    if EG <: AbstractElementGeometry1D
        A = zeros(Tv,1,1)
        node2oppositeface = [1, 2]
        xreftest = zeros(Tv,2)
    elseif EG <: Triangle2D
        A = zeros(Tv,2,2)
        node2oppositeface = [3, 1, 2]
        xreftest = zeros(Tv,3)
    elseif EG <: Tetrahedron3D
        A = zeros(Tv,3,3)
        node2oppositeface = [4, 2, 1, 3]
        xreftest = zeros(Tv,4)
    elseif EG <: Parallelogram2D
        A = zeros(Tv,2,2)
        node2oppositeface = [4, 1, 2, 3]
        xreftest = zeros(Tv,4)
    elseif EG <: Parallelepiped3D
        A = zeros(Tv,3,3)
        node2oppositeface = [5, 2, 1, 3, 4, 6]
        xreftest = zeros(Tv,6)
    else
        @error "ElementGeometry not supported by CellFinder"
    end
    return CellFinder{Tv,Ti,EG,CS}(xgrid, xgrid[CellFaces], xgrid[FaceCells], node2oppositeface, zeros(Ti,3), L2G, A, xreftest, zeros(Tv,size(A,1)))
end


# modification of pdelib function in gfind.cxx
function gFindLocal!(xref, CF::CellFinder{Tv,Ti,EG,CS}, x; icellstart::Int = 1, eps = 1e-14) where{Tv,Ti,EG,CS}

    # works for convex domainsand simplices only !
    xCellFaces::GridAdjacencyTypes = CF.xCellFaces
    xFaceCells::GridAdjacencyTypes = CF.xFaceCells
    L2G::L2GTransformer{Tv,EG,CS} = CF.L2G
    cx::Vector{Tv} = CF.cx
    node2oppositeface::Array{Int,1} = CF.node2oppositeface
    icell::Int = icellstart
    previous_cells::Array{Int,1} = CF.previous_cells
    fill!(previous_cells,0)
    xreftest::Array{Tv,1} = CF.xreftest

    invA::Matrix{Tv} = CF.invA
    L2Gb::Vector{Tv} = L2G.b
    imin::Int = 0

    while (true)
        # compute barycentric coordinates of point in current simplex
        update!(L2G, icell)

        # compute barycentric coordinates of node
        for j = 1 : length(x)
            cx[j] = x[j] - L2Gb[j]
        end
        mapderiv!(invA,L2G,xref)
        fill!(xreftest,0)
        for j = 1 : length(x), k = 1 : length(x)
            xreftest[k] += invA[j,k] * cx[j]
        end
        postprocess_xreftest!(CF)

        # find minimal barycentric coordinate with
        imin = 1
        for i = 2 : length(xreftest)
            if xreftest[imin] >= xreftest[i]
                imin = i
            end
        end

        # if all barycentric coordinates are within [0,1] the including cell is found
        if xreftest[imin] >= -eps
            xref .= view(xreftest,1:length(xref))
            return icell
        end

        # otherwise: go into direction of minimal barycentric coordinates
        for j = 1 : length(previous_cells)-1
            previous_cells[j] = previous_cells[j+1]
        end
        previous_cells[end] = icell
        icell = xFaceCells[1,xCellFaces[node2oppositeface[imin],icell]]
        if icell == previous_cells[end]
            icell = xFaceCells[2,xCellFaces[node2oppositeface[imin],icell]]
            if icell == 0
                @debug  "could not find point in any cell and ended up at boundary of domain (maybe x lies outside of the domain ?)"
                return 0
            end
        end

        if icell == previous_cells[end-1]
            @debug  "could not find point in any cell and ended up going in circles (better try brute force search)"
            return 0
        end
    end
    
    return 0
end

function gFindBruteForce!(xref, CF::CellFinder{Tv,Ti,EG,CS}, x; eps = 1e-14) where{Tv,Ti,EG,CS}

    L2G::L2GTransformer{Tv,EG,CS} = CF.L2G
    cx::Vector{Tv} = CF.cx
    previous_cells::Array{Int,1} = CF.previous_cells
    fill!(previous_cells,0)

    invA::Matrix{Tv} = CF.invA
    L2Gb::Vector{Tv} = L2G.b
    imin::Int = 0

    for icell = 1 : num_sources(CF.xgrid[CellNodes])
        # compute barycentric coordinates of point in current simplex
        update!(L2G, icell)

        # compute barycentric coordinates of node
        for j = 1 : length(x)
            cx[j] = x[j] - L2Gb[j]
        end
        mapderiv!(invA,L2G,xref)
        fill!(xreftest,0)
        for j = 1 : length(x), k = 1 : length(x)
            xreftest[k] += invA[j,k] * cx[j]
        end
        postprocess_xreftest!(CF)

        # find minimal barycentric coordinate with
        imin = 1
        for i = 2 : length(xreftest)
            if xreftest[imin] >= xreftest[i]
                imin = i
            end
        end

        # if all barycentric coordinates are within [0,1] the including cell is found
        if xreftest[imin] >= -eps
            xref .= view(xreftest,1:length(xref))
            return icell
        end
    end

    @debug "gFindBruteForce did not find any cell that contains x = $x (make sure that x is inside the domain, or try reducing $eps)"
    
    return 0
end
