vtkcelltype(::Type{<:AbstractElementGeometry1D}) = 3
vtkcelltype(::Type{<:Triangle2D}) = 5
vtkcelltype(::Type{<:Quadrilateral2D}) = 9
vtkcelltype(::Type{<:Tetrahedron3D}) = 10
vtkcelltype(::Type{<:Hexahedron3D}) = 12

"""
$(TYPEDSIGNATURES)

Writes the specified FEVector into a vtk datafile with the given filename. Each component of each FEVectorBlock
(or the subset specified by blocks) is saved separately. Vector-valued quantities also generate a data field
that represents the absolute value of the vector field at each grid point.
"""
function writeVTK!(filename::String, Data::FEVector; blocks = [], operators = [], names = [], vectorabs::Bool = true)
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
    xCoordinates::Array{Float64,2} = xgrid[Coordinates]
    nnodes::Int = size(xCoordinates,2)
    xdim::Int = size(xCoordinates,1)
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
    xCellNodes::GridAdjacencyTypes = xgrid[CellNodes]
    ncells::Int = num_sources(xCellNodes)
    nlinks::Int = num_links(xCellNodes) + ncells
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
    nblocks::Int = length(Data)
    ncomponents::Int = 0
    maxcomponents::Int = 0
    nfields::Int = 0
    block::Int = 0
    for j = 1 : length(blocks)
        block = blocks[j]
        while length(operators) < j
            push!(operators, Identity)
        end
        ncomponents = Length4Operator(operators[j], xdim, get_ncomponents(eltype(Data[block].FES))) 
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
    absval::Float64 = 0.0
    for b = 1 : length(blocks)
        block = blocks[b]
        ncomponents = Length4Operator(operators[b], xdim, get_ncomponents(eltype(Data[block].FES))) 
        # get node values
        nodevalues!(nodedata, Data[block], Data[block].FES, operators[b])
        for c = 1 : ncomponents
            if length(names) >= b
                fieldname = names[b]
            else
                fieldname = "$(operators[b])" * "(" * Data[block].name * ")"
                if ncomponents == 1
                    fieldname = fieldname[1:min(30,length(fieldname))]
                    fieldname = replace(String(fieldname), " " => "_")
                    fieldname = "$(b)_$fieldname"
                else
                    fieldname = fieldname[1:min(28,length(fieldname))]
                    fieldname = replace(String(fieldname), " " => "_")
                    fieldname = "$(b)_$fieldname.$c"
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
            fieldname = "$(operators[b])" * "(" * Data[block].name * ")"
            fieldname = fieldname[1:min(28,length(fieldname))]
            fieldname = replace(String(fieldname), " " => "_")
            fieldname = "$(b)_$fieldname.a"
            @printf(io, "%s 1 %d float\n", fieldname, nnodes)
            for node = 1 : nnodes
                absval = 0.0
                for c = 1 : ncomponents
                    absval += nodedata[c,node]^2
                end
                @printf(io, "%f ", sqrt(absval))
            end
        end
    end
    close(io)
end

"""
$(TYPEDSIGNATURES)

Writes the specified FEVector into a CSV datafile with the given filename. First d colomuns are the grid coordinates, the remaining column are filled
with the evaluations of the operators where operator[j] is applied to Source[blockids[j]].

"""
function writeCSV!(filename::String, Source::FEVector; blockids = [], operators = [], names = [], seperator::String = "\t")
    if blockids == []
        blockids = 1:length(Data)
    end

    # open file and write VTK header
    io = open(filename, "w")
    @logmsg MoreInfo "data export to csv file $filename"
    
    # open grid
    xgrid = Source[1].FES.xgrid
    xCoordinates = xgrid[Coordinates]
    nnodes = size(xCoordinates,2)
    xdim = size(xCoordinates,1)
    for j = 1 : xdim
        @printf(io, "X%d",j)
        @printf(io, "%s", seperator)
    end

    # collect data
    offsets = zeros(Int, length(blockids)+1)
    for j = 1 : length(blockids)
        if blockids[j] > 0
            FEType = eltype(Source[blockids[j]].FES)
            offsets[j+1] = offsets[j] + Length4Operator(operators[j], xdim, get_ncomponents(FEType))
        end
    end

    nodevals = zeros(Float64,offsets[end],nnodes)
    name = ""
    for j = 1 : length(blockids)
        try
            name = names[j]
        catch
            name = "$(operators[j])(" * Source[blockids[j]].name * ")"
        end
        @printf(io, "%s %s",name,seperator)
        ## evaluate operator
        if blockids[j] > 0
            @debug "collecting data for block $j : " * "$(operators[j])(" * Source[blockids[j]].name * ")"
            nodevalues!(nodevals, Source[blockids[j]], operators[j]; target_offset = offsets[j], zero_target = false)
        end
    end
    @printf(io, "\n")

    # write data
    for n = 1 : nnodes
        for j = 1 : xdim
            @printf(io, "%.6f",xCoordinates[j,n])
            @printf(io, "%s", seperator)
        end
        for j = 1 : offsets[end]
            @printf(io, "%.6f",nodevals[j,n])
            @printf(io, "%s", seperator)
        end
        @printf(io, "\n")
    end
    close(io)
end
