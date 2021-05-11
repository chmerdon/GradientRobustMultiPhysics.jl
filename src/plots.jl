
"""
$(TYPEDSIGNATURES)

Plots operators applied to components of a given vector of FEVectorBlocks Sources (operator[j] is applied to block j)
via ExtendablePlots.plot, either in one window with subplots or several single plots.

Plotter = PyPlot should work in 2D
Plotter = Makie should work in 3D (but only with a single plot currently)
"""
function plot(
    xgrid::ExtendableGrid,
    Sources::Array{<:FEVectorBlock,1},
    operators::Array{DataType,1};
    add_grid_plot::Bool = false,
    Plotter = nothing,
    subplots_per_column = 2,
    resolution = "auto",
    use_subplots::Bool = true,
    colorlevels=51,
    isolines=9,
    aspect=1,
    show=true,
    clear=true,
    cbar=true,
    cmap = "hot",
    maintitle = "")

    nplots = length(Sources) + add_grid_plot

    if nplots == 1
        use_subplots = false
    end

    if Plotter != nothing
        # extract grid (assumed to be the same for all FEVectorBlocks)
        spacedim = size( xgrid[Coordinates],1)

        @info "Plotting $(length(Sources)) quantities $(add_grid_plot ? "and the grid" : "")"

        if xgrid[UniqueCellGeometries] != [Triangle2D] && xgrid[UniqueCellGeometries] != [Tetrahedron3D]
            @debug "need simplices, splitting into simplices"
            xgrid = deepcopy(xgrid)
            if spacedim == 2
                xgrid = split_grid_into(xgrid,Triangle2D)
            elseif spacedim == 3
                xgrid = split_grid_into(xgrid,Tetrahedron3D)
            end
        end
        xCoordinates = xgrid[Coordinates]
        nnodes = num_sources(xCoordinates)

        # collect data
        offsets = zeros(Int, nplots+1-add_grid_plot)
        for j = 1 : nplots - add_grid_plot
            if typeof(Sources[j]) <: FEVectorBlock
                FEType = eltype(Sources[j].FES)
                offsets[j+1] = offsets[j] + Length4Operator(operators[j], spacedim, get_ncomponents(FEType))
            end
        end

        nodevals = zeros(Float64,offsets[end],nnodes)
        for j = 1 : nplots - add_grid_plot
            ## evaluate operator
            if typeof(Sources[j]) <: FEVectorBlock
                @logmsg DeepInfo "collecting data for plot $j : " * "$(operators[j])(" * Sources[j].name * ")"
                nodevalues!(nodevals, Sources[j], operators[j]; target_offset = offsets[j], zero_target = false)
            end
        end

        ## plot
        if use_subplots
            # prepare plots
            layout = [1]
            while nplots >= layout[end]
                push!(layout, layout[end] + subplots_per_column)
            end
            layout_aspect = length(layout) / subplots_per_column
            if ispyplot(Plotter)
                if clear
                    Plotter.clf() 
                end
                # fig = Plotter.figure(figsize=(layout_aspect*fsize,fsize))
            elseif plottertype(Plotter) == MakieType
                # not supported yet
            end
            ctx = Array{SubVisualizer,1}(undef, nplots)
            if resolution == "auto"
                resolution = (subplots_per_column * 500, (length(layout)-1)*500)
            end
            # figure=fig, cbar=true
            vis=GridVisualizer(Plotter=Plotter, layout=((nplots-1)÷subplots_per_column+1,subplots_per_column), clear=true, edges=true, cmap = cmap, show = show, isolines = isolines, colorlevels = colorlevels, aspect = aspect, resolution = resolution)
            if ispyplot(Plotter)
                Plotter.figure(vis.context[:fignumber], figsize = resolution)
            end
            for j = 1 : nplots
                #                subplot = subplots_per_column * 100 + (length(layout)-1) * 10 + j
                ctx[j] = vis[(j-1)÷subplots_per_column+1,(j-1)%subplots_per_column+1]
            end
        else
            ctx = Array{SubVisualizer,1}(undef, nplots)
            if resolution == "auto"
                resolution = (500, 500)
            end
            for j = 1 : nplots
                #cbar = cbar, 
                vis=GridVisualizer(Plotter=Plotter, fignumber=j, clear=true, edges=true, cmap = cmap, show = show, isolines = isolines, colorlevels = colorlevels, aspect = aspect, resolution = resolution)
                ctx[j] = vis[1,1]
            end
        end

        # fill plots
        Z = zeros(Float64,nnodes)
        for j = 1 : nplots - add_grid_plot
            if offsets[j+1] - offsets[j] > 1
                Z[:] = sqrt.(sum(view(nodevals,(offsets[j]+1):offsets[j+1],:).^2, dims = 1))
                title = maintitle * " | $(operators[j])(" * Sources[j].name * ") |"
            else
                Z[:] = view(nodevals,offsets[j]+1,:)
                title = maintitle * " $(operators[j])(" * Sources[j].name * ")"
            end
            if minimum(Z) == maximum(Z)
                Z[1] += 1e-16
            end
            @logmsg DeepInfo "plotting data into plot $j : " * title
            scalarplot!(ctx[j], xgrid, Z) 
            if plottertype(Plotter) == PyPlotType
                Plotter.title(title)
            end
        end
        if add_grid_plot
            @logmsg DeepInfo "plotting grid"
            gridplot!(ctx[nplots], xgrid,show=true; linewidth = 1) 
            if plottertype(Plotter) == PyPlotType
                Plotter.title(maintitle * " grid")
            end
        end
        reveal(vis) 
    end
end

function print_convergencehistory(X, Y; add_h_powers = [2], X_to_h = X -> X, name = "convergence history", labels = [], xlabel = "ndofs")

    ## print results
    xlabel = center_string(xlabel,12)
    @printf("\n%s|",xlabel)
    for j = 1 : size(Y,2)
        if length(labels) < j
            push!(labels, "DATA $j")
        end
        @printf("%s  order  |", center_string(labels[j],16))
    end
    @printf("\n")
    @printf("============|")
    for j = 1 : size(Y,2)
        @printf("=========================|")
    end
    @printf("\n")
    order = 0
    for j=1:length(X)
        @printf("   %7d  |",X[j]);
        for k = 1 : size(Y,2)
            if j > 1
                order = -log(Y[j-1,k]/Y[j,k]) / (log(X_to_h(X[j])/X_to_h(X[j-1])))
            end
            @printf("   %.5e    %.3f  |",Y[j,k], order)
        end
        @printf("\n")
    end
end


function plot_convergencehistory(X, Y; add_h_powers = [2], X_to_h = X -> X, Plotter = nothing, name = "convergence history", resolution = (800,600), labels = [], xlabel = "ndofs")
    if plottertype(Plotter) == PyPlotType
        Plotter.figure(name, figsize = resolution ./ 100)
        Plotter.loglog(X, Y; marker = "o");
        Plotter.xlabel(xlabel) 
        while length(labels) < size(Y,2)
            push!(labels, "Data $(length(labels)+1)")
        end
        for p in add_h_powers
            Plotter.loglog(X, X_to_h(X).^p; linestyle = "dotted"); 
            push!(labels, "h^$p")
        end
        Plotter.legend(labels)
    end
end