
"""
$(TYPEDSIGNATURES)

Plots operators applied to components of a given FEVector Source (operator[j9] is applied to component with blockids[j])
via ExtendablePlots.plot, either in one window with subplots or several single plots.

If blockids[j] == 0, a plot of the grid is generated at the j-th subplot.

Plotter = PyPlot should work in 2D
Plotter = Makie should work in 3D (but only with a single plot currently)
"""
function plot(
    Source::FEVector,
    blockids::Array{Int,1},
    operators::Array{DataType,1};
    Plotter = nothing,
    use_subplots = false,
    subplots_per_column = 2,
    colorlevels=51,
    isolines=11,
    aspect=1,
    show=true,
    clear=true,
    cbar=true,
    verbosity::Int = 0,
    cmap = "hot",
    maintitle = "",
    fsize = 10)

    if Plotter != nothing
        # extract grid
        xgrid = Source[1].FES.xgrid
        spacedim = size( xgrid[Coordinates],1)

        if verbosity > 0
            println("\nPLOTTING")
            println("========\n")
        end

        if xgrid[UniqueCellGeometries] != [Triangle2D] && xgrid[UniqueCellGeometries] != [Tetrahedron3D]
            if verbosity > 0
                println("   need simplices, splitting into simplices")
            end
            xgrid = deepcopy(Source[1].FES.xgrid)
            if spacedim == 2
                xgrid = split_grid_into(xgrid,Triangle2D)
            elseif spacedim == 3
                xgrid = split_grid_into(xgrid,Tetrahedron3D)
            end
        end
        xCoordinates = xgrid[Coordinates]
        nnodes = num_sources(xCoordinates)

        # collect data
        offsets = zeros(Int, length(blockids)+1)
        for j = 1 : length(blockids)
            if blockids[j] > 0
                FEType = eltype(Source[blockids[j]].FES)
                offsets[j+1] = offsets[j] + Length4Operator(operators[j], spacedim, get_ncomponents(FEType))
            end
        end

        nodevals = zeros(Float64,offsets[end],nnodes)
        for j = 1 : length(blockids)
            ## evaluate operator
            if blockids[j] > 0
                if verbosity > 0
                    println("   collecting data for plot $j : " * "$(operators[j])(" * Source[blockids[j]].name * ")")
                end
                nodevalues!(nodevals, Source[blockids[j]], operators[j]; target_offset = offsets[j], zero_target = false)
            end
        end

        ## plot
        if use_subplots
            # prepare plots
            layout = [1]
            while length(blockids) >= layout[end]
                push!(layout, layout[end] + subplots_per_column)
            end
            layout_aspect = length(layout) / subplots_per_column
            if ispyplot(Plotter)
                if clear
                    Plotter.clf()
                end
            end
            if plottertype(Plotter) == PyPlotType
                fig = Plotter.figure(maintitle,figsize=(layout_aspect*fsize,fsize))
            elseif plottertype(Plotter) == MakieType
                # not supported yet
            end
            ctx = Array{SubVis,1}(undef, length(blockids))
            # figure=fig, cbar=true
            vis=GridVisualizer(Plotter=Plotter, layout=(subplots_per_column,length(blockids)÷subplots_per_column), clear=false, legend=false, edges=true,cmap = cmap, show = show, isolines = isolines, colorlevels = colorlevels, aspect = aspect)
            for j = 1 : length(blockids)
                #                subplot = subplots_per_column * 100 + (length(layout)-1) * 10 + j
                ctx[j] = vis[(j-1)%subplots_per_column+1,(j-1)÷subplots_per_column+1]
            end
        else
            ctx = Array{SubVis,1}(undef, length(blockids))
            for j = 1 : length(blockids)
                #cbar = cbar, 
                vis=GridVisualizer(Plotter=Plotter, fignumber=j, clear=true, legend=false, edges=true, cmap = cmap, show = show,isolines = isolines, colorlevels = colorlevels, aspect = aspect)
                ctx[j] = vis[1,1]
            end
        end

        # fill plots
        Z = zeros(Float64,nnodes)
        for j = 1 : length(blockids)
            if blockids[j] == 0
                if verbosity > 0
                    println("   plotting grid into plot $j")
                end
                visualize!(ctx[j], xgrid,show=true) 
                if plottertype(Plotter) == PyPlotType
                    Plotter.title("grid")
                end
            else
                if offsets[j+1] - offsets[j] > 1
                    Z[:] = sqrt.(sum(view(nodevals,(offsets[j]+1):offsets[j+1],:).^2, dims = 1))
                    title = "| $(DefaultName4Operator(operators[j]))(" * Source[blockids[j]].name * ") |"
                else
                    Z[:] = view(nodevals,offsets[j]+1,:)
                    title = "$(DefaultName4Operator(operators[j]))(" * Source[blockids[j]].name * ")"
                end
                if use_subplots == false
                    title = maintitle * " " * title
                end
                if minimum(Z) == maximum(Z)
                    Z[1] += 1e-16
                end
                if verbosity > 0
                    println("   plotting data into plot $j : " * title)
                end
                visualize!(ctx[j], xgrid, Z) 
                if plottertype(Plotter) == PyPlotType
                    Plotter.title(title)
                end
            end
        end
        reveal(vis)
        
    end
end
