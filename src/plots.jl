
"""
$(TYPEDSIGNATURES)

DEPRECATED (use GridVisualize interfaces directly in combination with nodalvalues and PointEvaluator)

Plots scalar plots of nodval values of operators applied to the given vector of FEVectorBlocks (Sources) (this meands operator[j] is applied to block j)
via GridVisualizer (see documentation there for possible kwargs), either in one window with subplots (default) or several single plots.
If the operator evaluation is vector-valued the absolute value of this vector is plotted (quiver plots not supported by this interface yet).

Plotter = PyPlot should work in 2D
Plotter = GLMakie should work in 3D (but only with a single plot currently)
"""
function plot(
    xgrid::ExtendableGrid,
    Sources::Vector{<:FEVectorBlock},
    operators::Vector{DataType};
    add_grid_plot::Bool = false,
    Plotter = nothing,
    subplots_per_column = 2,
    use_subplots::Bool = true,
    resolution = "auto",
    maintitle = "",
    kwargs...)

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

        nodevals = Array{Array{Float64,2},1}(undef,nplots - add_grid_plot)
        for j = 1 : nplots - add_grid_plot
            ## evaluate operator
            if typeof(Sources[j]) <: FEVectorBlock
                nodevals[j] = zeros(Float64,offsets[j+1]-offsets[j],size(Sources[j].FES.xgrid[Coordinates],2))
                @logmsg DeepInfo "collecting data for plot $j : " * "$(operators[j])(" * Sources[j].name * ")"
                nodevalues!(nodevals[j], Sources[j], operators[j]; zero_target = false)
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
            ctx = Array{SubVisualizer,1}(undef, nplots)
            if resolution == "auto"
                resolution = (subplots_per_column * 500, (length(layout)-1)*500)
            end
            # figure=fig, cbar=true
            vis=GridVisualizer(Plotter=Plotter, layout=((nplots-1)÷subplots_per_column+1,subplots_per_column), clear=true, edges=true, resolution = resolution)
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
                vis=GridVisualizer(Plotter=Plotter, fignumber=j, clear=true, edges=true, resolution = resolution)
                ctx[j] = vis[1,1]
            end
        end

        # fill plots    
        temp::Float64 = 0
        for j = 1 : nplots - add_grid_plot
            if offsets[j+1] - offsets[j] > 1
                for n = 1 : size(nodevals[j],2)
                    temp = 0
                    for k = 1 : size(nodevals[j],1)
                        temp += nodevals[j][k,n].^2
                    end
                    nodevals[j][1,n] = sqrt(temp)
                end
                title = maintitle * " | $(operators[j])(" * Sources[j].name * ") |"
            else
                title = maintitle * " $(operators[j])(" * Sources[j].name * ")"
            end
            @logmsg DeepInfo "plotting data into plot $j : " * title
            scalarplot!(ctx[j], Sources[j].FES.xgrid, view(nodevals[j],1,:); kwargs...) 
            if plottertype(Plotter) == PyPlotType
                Plotter.title(title)
            end
        end
        if add_grid_plot
            @logmsg DeepInfo "plotting grid"
            gridplot!(ctx[nplots], xgrid, clear=true, show=true; linewidth = 1, kwargs...) 
            if plottertype(Plotter) == PyPlotType
                Plotter.title(maintitle * " grid")
            end
        end
        reveal(vis) 
    end
end


function plot_convergencehistory(X, Y; add_h_powers = [], X_to_h = X -> X, Plotter = nothing, name = "convergence history", resolution = (800,600), ylabel = "", ylabels = [], xlabel = "ndofs", clear = true, legend_fontsize = 14, label_fontsize = 14)
    labels = deepcopy(ylabels)
    if plottertype(Plotter) == PyPlotType
        Plotter.figure(name, figsize = resolution ./ 100)
        if clear
            Plotter.clf()
        end
        for j = 1 : size(Y,2)
            Xk = []
            Yk = []
            for k = 1 : length(X)
                if Y[k,j] > 0
                    push!(Xk,X[k])
                    push!(Yk,Y[k,j])
                end
            end
            Plotter.loglog(Xk,Yk; marker = "o");
        end
        Plotter.ylabel(ylabel; fontsize = label_fontsize) 
        Plotter.xlabel(xlabel; fontsize = label_fontsize) 
        Plotter.xticks(fontsize = label_fontsize) 
        Plotter.yticks(fontsize = label_fontsize) 
        Plotter.grid("on")
        while length(labels) < size(Y,2)
            push!(labels, "Data $(length(labels)+1)")
        end
        for p in add_h_powers
            Plotter.loglog(X, X_to_h(X).^p; linestyle = "dotted"); 
            push!(labels, "h^$p")
        end
        Plotter.legend(labels; fontsize = legend_fontsize)
    end
end