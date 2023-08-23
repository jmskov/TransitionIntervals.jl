# Tools for plotting

function create_shape(state)
    shape = Plots.Shape([state[1,1], state[1,2], state[1,2], state[1,1]], 
                        [state[2,1], state[2,1], state[2,2], state[2,2]])
    return shape
end

function plot_with_alpha(states, alpha)
    plt = plot(aspect_ratio=1)
    plot_with_alpha!(plt, states, alpha)
    return plt
end

function plot_with_alpha!(plt, states, alpha)
        plot!(plt, create_shape.(states), fillcolor=:blue, fillalpha=alpha, alpha=0.1, label="")
end

function plot_with_classifications(states, classifications)
    plt = plot(aspect_ratio=1)
    plot_with_classifications!(plt, states, classifications)
    return plt
end

function plot_with_classifications!(plt, states, classifications)
    # make an array with just yellow
    colors = fill(:yellow, length(classifications))
    # change the colors of the states that are classified
    colors[classifications .== 1] .= :green
    colors[classifications .== 2] .= :red
    plot!(plt, create_shape.(states), fillcolor=colors, fillalpha=1.0, alpha=0.1, label="")
end

function plot_state_labels!(plt, states)
    for (i, state) in enumerate(states)
        state_mean = calculate_state_mean(state)
        annotate!(plt, state_mean[1], state_mean[2], text("$i", :black, :center, 5,))
    end
end

function save_figure_files(plt, filename)
    savefig(plt, filename * ".png")
    serialize(filename * ".plt", plt)
end

