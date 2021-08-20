
# conversion from AbstractElementGeometry to WriteVTK.VTKCellTypes
VTKCellType(::Type{<:AbstractElementGeometry1D}) = VTKCellTypes.VTK_LINE
VTKCellType(::Type{<:Triangle2D}) = VTKCellTypes.VTK_TRIANGLE
VTKCellType(::Type{<:Quadrilateral2D}) = VTKCellTypes.VTK_QUAD
VTKCellType(::Type{<:Tetrahedron3D}) = VTKCellTypes.VTK_TETRA
VTKCellType(::Type{<:Hexahedron3D}) = VTKCellTypes.VTK_HEXAHEDRON

"""
$(TYPEDSIGNATURES)

Writes the specified FEVector into a vtk datafile with the given filename. Each FEVectorBlock in the Data array
is saved as separate VTKPointData. Vector-valued quantities also generate a data field
that represents the absolute value of the vector field at each grid point (if vectorabs is true).
"""
function writeVTK!(filename::String, Data::Array{<:FEVectorBlock,1}; operators = [], names = [], vectorabs::Bool = true, add_regions = false, caplength::Int = 40)
    # open grid
    xgrid = Data[1].FES.xgrid
    xCoordinates = xgrid[Coordinates]
    xdim = size(xCoordinates,1)
    nnodes = size(xCoordinates,2)
    xCellNodes = xgrid[CellNodes]
    xCellGeometries = xgrid[CellGeometries]
    xCellRegions = xgrid[CellRegions]
    ncells = num_sources(xCellNodes)

    ## add grid to file
    vtk_cells = Array{MeshCell,1}(undef,ncells)
    for item = 1 : ncells
        vtk_cells[item] = MeshCell(VTKCellType(xCellGeometries[item]), xCellNodes[:,item])
    end
    vtkfile = vtk_grid(filename, xCoordinates, vtk_cells)

    if add_regions
        vtkfile["grid_regions", VTKCellData()] = xCellRegions
    end

    ## add data
    nblocks::Int = length(Data)
    ncomponents::Int = 0
    maxcomponents::Int = 0
    nfields::Int = 0
    block::Int = 0
    for d = 1 : nblocks
        while length(operators) < d
            push!(operators, Identity)
        end
        ncomponents = Length4Operator(operators[d], xdim, get_ncomponents(eltype(Data[d].FES))) 
        if ncomponents > maxcomponents
            maxcomponents = ncomponents
        end
    end
    nodedata = zeros(Float64, maxcomponents, nnodes)
    
    for d = 1 : length(Data)
        # get node values
        ncomponents = Length4Operator(operators[d], xdim, get_ncomponents(eltype(Data[d].FES))) 
        nodevalues!(nodedata, Data[d], Data[d].FES, operators[d])
        if length(names) >= d
            fieldname = names[d]
        else
            fieldname = "$(operators[d])" * "(" * Data[d].name * ")"
            fieldname = fieldname[1:min(caplength,length(fieldname))]
            fieldname = replace(String(fieldname), " " => "_")
            fieldname = "$(d)_$fieldname"
        end    
        for c = 1 : ncomponents
            vtkfile["$fieldname.$c", VTKPointData()] = view(nodedata,c,:)
        end
        # add data for absolute value of vector quantity
        if vectorabs && ncomponents > 1
            vtkfile["$fieldname.a", VTKPointData()] = sqrt.(sum(view(nodedata,1:ncomponents,:).^2, dims = 1))
        end
    end

    ## save file
    outfiles = vtk_save(vtkfile)
    return nothing
end



"""
$(TYPEDSIGNATURES)
Writes the specified FEVectorBlocks into a CSV datafile with the given filename. First d colomuns are the grid coordinates, the remaining columns are filled
with the evaluations of the operators where operator[j] is applied to Data[j].
"""
function writeCSV!(filename::String, Data::Array{<:FEVectorBlock,1}; operators = [], names = [], seperator::String = "\t")

    # open file and write VTK header
    io = open(filename, "w")
    @logmsg MoreInfo "data export to csv file $filename"
    
    # open grid
    xgrid = Data[1].FES.xgrid
    xCoordinates = xgrid[Coordinates]
    nnodes = size(xCoordinates,2)
    xdim = size(xCoordinates,1)
    for j = 1 : xdim
        @printf(io, "X%d",j)
        @printf(io, "%s", seperator)
    end

    # collect data
    offsets = zeros(Int, length(Data)+1)
    for j = 1 : length(Data)
        if length(operators) < j
            push!(operators, Identity)
        end
        FEType = eltype(Data[j].FES)
        offsets[j+1] = offsets[j] + Length4Operator(operators[j], xdim, get_ncomponents(FEType))
    end

    nodevals = zeros(Float64,offsets[end],nnodes)
    name = ""
    for j = 1 : length(Data)
        try
            name = names[j]
        catch
            name = "$(operators[j])(" * Data[j].name * ")"
        end
        @printf(io, "%s %s",name,seperator)
        ## evaluate operator
        @debug "collecting data for block $j : " * "$(operators[j])(" * Data[j].name * ")"
        nodevalues!(nodevals, Data[j], operators[j]; target_offset = offsets[j], zero_target = false)
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


function print_convergencehistory(X, Y; X_to_h = X -> X, ylabels = [], xlabel = "ndofs")
    xlabel = center_string(xlabel,12)
    @printf("\n%s|",xlabel)
    for j = 1 : size(Y,2)
        if length(ylabels) < j
            push!(ylabels, "DATA $j")
        end
        @printf("%s  order  |", center_string(ylabels[j],20))
    end
    @printf("\n")
    @printf("============|")
    for j = 1 : size(Y,2)
        @printf("=============================|")
    end
    @printf("\n")
    order = 0
    for j=1:length(X)
        @printf("   %7d  |",X[j]);
        for k = 1 : size(Y,2)
            if j > 1
                order = -log(Y[j-1,k]/Y[j,k]) / (log(X_to_h(X[j])/X_to_h(X[j-1])))
            end
            @printf("     %.5e      %.3f  |",Y[j,k], order)
        end
        @printf("\n")
    end
end

function print_table(X, Y; ylabels = [], xlabel = "ndofs")
    xlabel = center_string(xlabel,12)
    @printf("\n%s|",xlabel)
    for j = 1 : size(Y,2)
        if length(ylabels) < j
            push!(ylabels, "DATA $j")
        end
        @printf(" %s |", center_string(ylabels[j],20))
    end
    @printf("\n")
    @printf("============|")
    for j = 1 : size(Y,2)
        @printf("======================|")
    end
    @printf("\n")
    for j=1:length(X)
        if eltype(X) <: Int
            @printf("   %7d  |",X[j]);
        else
            @printf("  %.2e  |",X[j]);
        end
        for k = 1 : size(Y,2)
            @printf("    %.8e    |",Y[j,k])
        end
        @printf("\n")
    end
end
