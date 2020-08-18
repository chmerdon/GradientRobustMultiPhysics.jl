vtkcelltype(::Type{<:AbstractElementGeometry1D}) = 3
vtkcelltype(::Type{<:Triangle2D}) = 5
vtkcelltype(::Type{<:Quadrilateral2D}) = 9
vtkcelltype(::Type{<:Tetrahedron3D}) = 10
vtkcelltype(::Type{<:Hexahedron3D}) = 12 # maybe needs renumbering of cellnodes !!!

function writeVTK!(filename::String, Data::FEVector)
    # open grid
    xgrid = Data[1].FES.xgrid

    # open file and write VTK header
    io = open(filename, "w")
    @printf(io, "# vtk DataFile Version 3.0\n")
    @printf(io, "vtk output\n")
    @printf(io, "ASCII\n")
    @printf(io, "DATASET UNSTRUCTURED_GRID\n")
    
    # write POINTS
    xCoordinates = xgrid[Coordinates]
    nnodes = size(xCoordinates,2)
    xdim = size(xCoordinates,1)
    @printf(io, "POINTS %d float\n", nnodes)
    for node = 1 : nnodes
        for k = 1 : xdim
            if k > 1
                @printf(io, " ")
            end
            @printf(io, "%f", xCoordinates[k,node])
        end
        for k = xdim+1:3
            @printf(io, " %f", 0)
        end
        @printf(io, "\n")
    end

    @printf(io, "\n")
    # write CELLS
    xCellNodes = xgrid[CellNodes]
    ncells = num_sources(xCellNodes)
    nlinks = num_links(xCellNodes) + ncells
    @printf(io, "CELLS %d %d\n",ncells,nlinks)
    for cell = 1 : ncells
        @printf(io, "%d ", num_targets(xCellNodes,cell))
        for k = 1 : num_targets(xCellNodes,cell)
            if k > 1
                @printf(io, " ")
            end
            @printf(io, "%d", xCellNodes[k,cell]-1)
        end
        @printf(io, "\n")
    end

    @printf(io, "\n")
    # write CELLTYPES
    xCellGeometries = xgrid[CellGeometries]
    @printf(io, "CELL_TYPES %d\n",ncells)
    for cell = 1 : ncells
        @printf(io, "%s\n", vtkcelltype(xCellGeometries[cell]))
    end
    
    @printf(io, "\n")
    # write data from Data FEVectorBlocks
    nblocks = length(Data)
    ncomponents = 0
    maxcomponents = 0
    nfields = 0
    for block = 1 : nblocks
        ncomponents = get_ncomponents(eltype(Data[block].FES))
        if ncomponents > maxcomponents
            maxcomponents = ncomponents
        end
        nfields += ncomponents
    end
    nodedata = zeros(Float64, maxcomponents, nnodes)
    @printf(io, "CELL_DATA %d\n", ncells)
    @printf(io, "POINT_DATA %d\n", nnodes)
    @printf(io, "FIELD FieldData %d\n", nfields)
    fieldname = ""
    for block = 1 : nblocks
        # get node values
        nodevalues!(nodedata, Data[block], Data[block].FES)
        for c = 1 : get_ncomponents(eltype(Data[block].FES))
            #fieldname = "data.b$block.c$c"
            fieldname = "$(Data[block].name[1]).c$c"
            @printf(io, "%s 1 %d float\n", fieldname, nnodes)
            for node = 1 : nnodes
                @printf(io, "%f ", nodedata[c,node])
            end
            @printf(io, "\n")
        end
    end
    close(io)
end