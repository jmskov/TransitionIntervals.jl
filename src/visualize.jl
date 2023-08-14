# Tools for plotting

function create_shape(state)
    shape = Plots.Shape([state[1][1], state[1][2], state[1][2], state[1][1]], 
                        [state[2][1], state[2][2], state[2][2], state[2][1]])
    return shape
end

function plot_with_alpha(state, alpha)
    plt = plot()
    plot_with_alpha!(plt, state, alpha)
    return plt
end

function plot_with_alpha!(plt, state, alpha)
    plot!(plt, create_shape(state), fillcolor=:blue, fillalpha=alpha, alpha=0.1, label="")
end

function plot_with_classifications(state, classifications)
    plt = plot()
    plot_with_classifications!(plt, state, classifications)
    return plt
end

function plot_with_classifications!(plt, state, classifications)
    color = :yellow
    if classifications == 1
        color = :green
    elseif classifications == 2
        color = :red
    end
    plot!(plt, create_shape(state), fillcolor=color, fillalpha=1.0, alpha=0.1, label="")
end

function save_figure_files(plt, filename)
    savefig(plt, filename * ".png")
    serialize(filename * ".plt", plt)
end

