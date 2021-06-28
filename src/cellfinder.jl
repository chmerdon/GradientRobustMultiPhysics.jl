
struct CellFinder{Tv,Ti,EG,CS}
    xgrid::ExtendableGrid{Tv,Ti}
    xCellFaces::GridAdjacencyTypes
    xFaceCells::GridAdjacencyTypes
    node2oppositeface::Array{Int,1}
    previous_cells::Array{Int,1}
    L2G::L2GTransformer{Tv,EG,CS}
    invA::Matrix{Tv}
    cx::Vector{Tv}
end


function CellFinder(xgrid::ExtendableGrid{Tv,Ti}, EG) where {Tv,Ti}
    CS = xgrid[CoordinateSystem]
    L2G = L2GTransformer{Tv,EG,CS}(xgrid, ON_CELLS)
    if EG <: AbstractElementGeometry1D
        A = zeros(Tv,1,1)
    elseif EG <: Triangle2D
        A = zeros(Tv,2,2)
        node2oppositeface = [2, 3, 1]
    elseif EG <: Tetrahedron3D
        A = zeros(Tv,3,3)
        node2oppositeface = [3, 4, 2,1]
    else
        @error "ElementGeometry not supported by CellFinder"
    end
    return CellFinder{Tv,Ti,EG,CS}(xgrid, xgrid[CellFaces], xgrid[FaceCells], node2oppositeface, zeros(Ti,3), L2G, A, zeros(Tv,size(A,1)))
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

    invA::Matrix{Tv} = CF.invA
    L2Gb::Vector{Tv} = L2G.b
    xrefmin::Tv = 1e30
    imin::Int = 0

    while (true)
        # compute barycentric coordinates of point in current simplex
        update!(L2G, icell)

        # compute barycentric coordinates of node
        for j = 1 : length(x)
            cx[j] = x[j] - L2Gb[j]
        end
        mapderiv!(invA,L2G,xref)
        fill!(xref,0)
        for j = 1 : length(x), k = 1 : length(x)
            xref[k] += invA[j,k] * cx[j]
        end
        xrefmin = 1e30
        for i = 1 : length(xref)
            if xrefmin >= xref[i]
                xrefmin = xref[i]
                imin = i+1
            end
        end
        if xrefmin >= (1 - sum(xref))
            xrefmin = (1 - sum(xref))
            imin = 1
        end

        # if all barycentric coordinates are within [0,1] the including cell is found
        if  xrefmin >= -eps
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
    xrefmin::Tv = 1e30
    imin::Int = 0

    for icell = 1 : num_sources(CF.xgrid[CellNodes])
        # compute barycentric coordinates of point in current simplex
        update!(L2G, icell)

        # compute barycentric coordinates of node
        for j = 1 : length(x)
            cx[j] = x[j] - L2Gb[j]
        end
        mapderiv!(invA,L2G,xref)
        fill!(xref,0)
        for j = 1 : length(x), k = 1 : length(x)
            xref[k] += invA[j,k] * cx[j]
        end
        xrefmin = 1e30
        for i = 1 : length(xref)
            if xrefmin >= xref[i]
                xrefmin = xref[i]
                imin = i+1
            end
        end
        if xrefmin >= (1 - sum(xref))
            xrefmin = (1 - sum(xref))
            imin = 1
        end

        # if all barycentric coordinates are within [0,1] the including cell is found
        if  xrefmin >= -eps
            return icell
        end
    end

    @debug "gFindBruteForce did not find any cell that contains x = $x (make sure that x is inside the domain, or try reducing $eps)"
    
    return 0
end
