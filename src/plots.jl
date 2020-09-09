
"""
$(TYPEDSIGNATURES)

Plots operators applied to components of a given FEVector via ExtendablePlots.plot, either in subplots or several single plots.
"""
function plot(
    Source::FEVector,
    blockids::Array{Int,1},
    operators::Array{DataType,1};
    Plotter = nothing,
    use_subplots = false,
    colorlevels=51,
    isolines=11,
    aspect=1,
    show=true,
    clear=true,
    cbar=true,
    verbosity::Int = 0,
    cmap = "hot",
    fsize = 10)

    if Plotter != nothing
        # extract grid
        xgrid = Source[1].FES.xgrid

        if verbosity > 0
            println("\nPLOTTING")
            println("========\n")
        end

        if xgrid[UniqueCellGeometries] != [Triangle2D]
            if verbosity > 0
                println("   need triangles, splitting into triangles")
            end
            xgrid = deepcopy(Source[1].FES.xgrid)
            xgrid = split_grid_into(xgrid,Triangle2D)
        end
        xCoordinates = xgrid[Coordinates]
        nnodes = num_sources(xCoordinates)
        spacedim = size(xCoordinates,1)

        # collect data
        offsets = zeros(Int, length(blockids)+1)
        for j = 1 : length(blockids)
            FEType = eltype(Source[blockids[j]].FES)
            offsets[j+1] = offsets[j] + Length4Operator(operators[j], spacedim, get_ncomponents(FEType))
        end



        nodevals = zeros(Float64,offsets[end],nnodes)
        for j = 1 : length(blockids)
            ## evaluate operator
            if verbosity > 0
                println("   collecting data for plot $j : " * "$(operators[j])(" * Source[blockids[j]].name * ")")
            end
            nodevalues!(nodevals, Source[blockids[j]], operators[j]; target_offset = offsets[j], zero_target = false)
        end

        ## plot
        if use_subplots
            if ExtendableGrids.ispyplot(Plotter)
                fig = Plotter.figure("Subplots",figsize=(fsize/length(blockids),fsize))
                if clear
                    Plotter.clf()
                end
            end
        end
        Z = zeros(Float64,nnodes)
        for j = 1 : length(blockids)
            if offsets[j+1] - offsets[j] > 1
                Z[:] = sqrt.(sum(view(nodevals,(offsets[j]+1):offsets[j+1],:).^2, dims = 1))
                title = "| $(operators[j])(" * Source[blockids[j]].name * ") |"
            else
                Z[:] = view(nodevals,offsets[j]+1,:)
                title = "$(operators[j])(" * Source[blockids[j]].name * ")"
            end
            if verbosity > 0
                println("   plotting data into plot $j : " * title)
            end
            if use_subplots
                if ExtendableGrids.ispyplot(Plotter)
                    Plotter.subplot(length(blockids),1,j)
                    Plotter.title(title)
                    ExtendableGrids.plot(xgrid, Z; Plotter = Plotter, cmap = cmap, clear = false, show = show, cbar = cbar, isolines = isolines, colorlevels = colorlevels, aspect = aspect)
                end
            else
                if ExtendableGrids.ispyplot(Plotter)
                    Plotter.figure(title)
                    ExtendableGrids.plot(xgrid, Z; Plotter = Plotter, cmap = cmap, clear = clear, show = show, cbar = cbar, isolines = isolines, colorlevels = colorlevels, aspect = aspect)
                end
            end
            
        end
    end

end