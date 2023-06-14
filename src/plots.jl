function convergencehistory!(target, X, Y; add_h_powers = [], X_to_h = X -> X, colors = [:blue, :green, :red, :magenta, :lightblue], title = "convergence history", legend = :best, ylabel = "", ylabels = [], xlabel = "ndofs", markershape = :circle, markevery = 1, clear = true, args...)
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
        scalarplot!(target, simplexgrid(Xk), Yk; xlabel = xlabel, ylabel = ylabel, color = length(colors) >= j ? colors[j] : :black, clear = j == 1 ? clear : false, markershape = markershape, markevery = markevery, xscale = :log, yscale = :log, label = label, legend = legend, title = title, args...)
    end
    for p in add_h_powers
        label = "h^$p"
        scalarplot!(target, simplexgrid(X), X_to_h(X).^p; linestyle = :dot, xlabel = xlabel, ylabel = ylabel, color = :gray, clear = false, markershape = :none, xscale = :log, yscale = :log, label = label, legend = legend, title = title, args...)
    end
end

function plot_convergencehistory(X, Y; Plotter = nothing, resolution = (800,600), add_h_powers = [], X_to_h = X -> X, colors = [:blue, :green, :red, :magenta, :lightblue], legend = :best, ylabel = "", ylabels = [], xlabel = "ndofs", clear = true, args...)
    p=GridVisualizer(; Plotter = Plotter, layout = (1,1), clear = true, resolution = resolution)
    convergencehistory!(p[1,1], X, Y; add_h_powers = add_h_powers, X_to_h = X_to_h, colors = colors, legend = legend, ylabel = ylabel, ylabels = ylabels, xlabel = xlabel, clear = clear, args...)
end