

function center_string(S::String, L::Int = 8)
    if length(S) > L
        S = S[1:L]
    end
    while length(S) < L-1
        S = " " * S * " "
    end
    if length(S) < L
        S = " " * S
    end
    return S
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


function print_convergencehistory(X, Y; X_to_h = X -> X, ylabels = [], xlabel = "ndofs", latex_mode = false, seperator = latex_mode ? "&" : "|", order_seperator = latex_mode ? "&" : "")
    xlabel = center_string(xlabel,12)
    if latex_mode
        tabular_argument = "c"
        for j = 1 : size(Y,2)
            tabular_argument *= "|cc"
        end
        @printf("\\begin{tabular}{%s}",tabular_argument)
    end
    @printf("\n%s%s",xlabel,seperator)
    for j = 1 : size(Y,2)
        if length(ylabels) < j
            push!(ylabels, "DATA $j")
        end
        if j == size(Y,2)
            @printf("%s %s order %s", center_string(ylabels[j],18), order_seperator, latex_mode ? "" : seperator)
        else
            @printf("%s %s order %s", center_string(ylabels[j],18), order_seperator, seperator)
        end
    end
    @printf("\n")
    if latex_mode
        @printf("\\\\\\hline")
    else
        @printf("============|")
        for j = 1 : size(Y,2)
            @printf("==========================|")
        end
    end
    @printf("\n")
    order = 0
    for j=1:length(X)
        @printf("   %7d  %s",X[j],seperator);
        for k = 1 : size(Y,2)
            if j > 1
                order = -log(Y[j-1,k]/Y[j,k]) / (log(X_to_h(X[j])/X_to_h(X[j-1])))
            end
            if k == size(Y,2)
                @printf("     %.3e  %s    %.2f  %s",Y[j,k], order_seperator, order, latex_mode ? "" : seperator)
            else
                @printf("     %.3e  %s    %.2f  %s",Y[j,k], order_seperator, order, seperator)
            end
        end
        if latex_mode
            @printf("\\\\")
        end
        @printf("\n")
    end
    if latex_mode
        @printf("\\end{tabular}")
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
