function convergencehistory!(target, X, Y; add_h_powers = [], X_to_h = X -> X, colors = [:blue, :green, :red, :magenta, :lightblue], title = "convergence history", legend = :best, ylabel = "", ylabels = [], xlabel = "ndofs", clear = true, fontsize = 14)
    for j = 1 : size(Y,2)
        Xk = []
        Yk = []
        for k = 1 : length(X)
            if Y[k,j] > 0
                push!(Xk,X[k])
                push!(Yk,Y[k,j])
            end
        end
        if length(ylabels) >= j
            label = ylabels[j]
        else
            label = "Data $j"
        end
        scalarplot!(target, simplexgrid(Xk), Yk; fontsize = fontsize, xlabel = xlabel, ylabel = ylabel, color = length(colors) >= j ? colors[j] : :black, clear = j == 1 ? clear : false, xscale = :log, yscale = :log, markershape = :circle, markevery = 1, label = label, legend = legend, title = title)
    end
    for p in add_h_powers
        label = "h^$p"
        scalarplot!(target, simplexgrid(X), X_to_h(X).^p; fontsize = fontsize, linestyle = :dot, xlabel = xlabel, ylabel = ylabel, color = :gray, clear = false, markershape = :none, xscale = :log, yscale = :log, label = label, legend = legend, title = title)
    end
end

function plot_convergencehistory(X, Y; Plotter = nothing, resolution = (800,600), add_h_powers = [], X_to_h = X -> X, colors = [:blue, :green, :red, :magenta, :lightblue], title = "convergence history", legend = :best, ylabel = "", ylabels = [], xlabel = "ndofs", clear = true, fontsize = 14)
    p=GridVisualizer(; Plotter = Plotter, layout = (1,1), clear = true, resolution = resolution)
    plot_convergencehistory!(p[1,1], X, Y; add_h_powers = add_h_powers, X_to_h = X_to_h, colors = colors, title = title, legend = legend, ylabel = ylabel, ylabels = ylabels, xlabel = xlabel, clear = clear, fontsize = fontsize)
end