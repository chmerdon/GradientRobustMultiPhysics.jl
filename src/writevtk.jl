vtkcelltype(::Type{<:AbstractElementGeometry1D}) = 3
vtkcelltype(::Type{<:Triangle2D}) = 5
vtkcelltype(::Type{<:Quadrilateral2D}) = 9
vtkcelltype(::Type{<:Tetrahedron3D}) = 10
vtkcelltype(::Type{<:Hexahedron3D}) = 12 # maybe needs renumbering of cellnodes !!!

"""
$(TYPEDSIGNATURES)

Writes the specified FEVector into a vtk datafile with the given filename. Each component of each FEVectorBlock
(or the subset specified by blocks) is saved separately. Vector-valued quantities also generate a data field
that represents the absolute value of the vector field at each grid point.
"""
function writeVTK!(filename::String, Data::FEVector; blocks = [], names = [], vectorabs::Bool = true)
    if blocks == []
        blocks = 1:length(Data)
    end
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
    for block in blocks
        ncomponents = get_ncomponents(eltype(Data[block].FES))
        if ncomponents > maxcomponents
            maxcomponents = ncomponents
        end
        nfields += ncomponents
        if vectorabs && ncomponents > 1
            nfields += 1
        end
    end
    nodedata = zeros(Float64, maxcomponents, nnodes)
    @printf(io, "CELL_DATA %d\n", ncells)
    @printf(io, "POINT_DATA %d\n", nnodes)
    @printf(io, "FIELD FieldData %d\n", nfields)
    fieldname = ""
    absval = 0
    for b = 1 : length(blocks)
        block = blocks[b]
        ncomponents = get_ncomponents(eltype(Data[block].FES))
        # get node values
        nodevalues!(nodedata, Data[block], Data[block].FES)
        for c = 1 : ncomponents
            if length(names) >= b
                fieldname = names[b]
            else
                if ncomponents == 1
                    fieldname = Data[block].name[1:min(30,length(Data[block].name))]
                    fieldname = replace(String(fieldname), " " => "_")
                    fieldname = "$(block)_$fieldname"
                else
                    fieldname = Data[block].name[1:min(28,length(Data[block].name))]
                    fieldname = replace(String(fieldname), " " => "_")
                    fieldname = "$(block)_$fieldname.$c"
                end
            end    
            @printf(io, "%s 1 %d float\n", fieldname, nnodes)
            for node = 1 : nnodes
                @printf(io, "%f ", nodedata[c,node])
            end
            @printf(io, "\n")
        end
        # add data for absolute value of vector quantity
        if vectorabs && ncomponents > 1
            nfields += 1
            fieldname = Data[block].name[1:min(28,length(Data[block].name))]
            fieldname = replace(String(fieldname), " " => "_")
            fieldname = "$(block)_$fieldname.a"
            @printf(io, "%s 1 %d float\n", fieldname, nnodes)
            for node = 1 : nnodes
                absval = 0
                for c = 1 : ncomponents
                    absval += nodedata[c,node]^2
                end
                @printf(io, "%f ", sqrt(absval))
            end
        end
    end
    close(io)
end