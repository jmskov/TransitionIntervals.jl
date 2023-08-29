# Tools for plotting

function create_shape(state)
    shape = Plots.Shape([state[1,1], state[1,2], state[1,2], state[1,1]], 
                        [state[2,1], state[2,1], state[2,2], state[2,2]])
    return shape
end

function plot_with_alpha(states, alpha)
    plt = plot(aspect_ratio=1, dpi=300)
    plot_with_alpha!(plt, states, alpha)
    return plt
end

function plot_with_alpha!(plt, states, alpha)
        plot!(plt, create_shape.(states), fillcolor=:blue, fillalpha=permutedims(alpha), alpha=0.1, label="")
end

function plot_with_classifications(states, classifications)
    plt = plot(aspect_ratio=1, dpi=300)
    plot_with_classifications!(plt, states, classifications)
    return plt
end

function plot_with_classifications!(plt, states, classifications)
    # make an array with just yellow
    colors = fill(:yellow, length(classifications))
    # change the colors of the states that are classified
    colors[classifications .== 1] .= :green
    colors[classifications .== 2] .= :red
    plot!(plt, create_shape.(states), fillcolor=permutedims(colors), fillalpha=1.0, linewidth=0.0, label="")
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

function plot_all_results(results_dir, states, results_matrix; threshold=0.9, xlims=:none, ylims=:none, state_labels=nothing)
    figure_filename = "$results_dir/sat-lower-bound"
    plt = plot(aspect_ratio=1, dpi=300, xlims=xlims, ylims=ylims)
    plot_with_alpha!(plt, states, results_matrix[:,3])
    if !isnothing(state_labels)
        plot_labelled_states_outline!(plt, state_labels)
    end
    save_figure_files(plt, figure_filename)
    figure_filename = "$results_dir/sat-upper-bound"
    plt = plot(aspect_ratio=1, dpi=300, xlims=xlims, ylims=ylims)
    plot_with_alpha!(plt, states, results_matrix[:,4])
    if !isnothing(state_labels)
        plot_labelled_states_outline!(plt, state_labels)
    end
    save_figure_files(plt, figure_filename)
    classifications = classify_results(results_matrix, threshold)
    plt = plot(aspect_ratio=1, dpi=300, xlims=xlims, ylims=ylims)
    plot_with_classifications!(plt, states, classifications)
    if !isnothing(state_labels)
        plot_labelled_states_outline!(plt, state_labels)
    end
    figure_filename = "$results_dir/sat-classification"
    save_figure_files(plt, figure_filename)
end

function plot_labelled_states_outline!(plt, state_label_dict; color=:black, fontsize=16)
    for (label, states) in state_label_dict
        for state in states
            plot_labelled_state_outline!(plt, state, label, color=color, fontsize=fontsize)
        end
    end
end

function plot_labelled_state_outline!(plt, state, label; color=:black, fontsize=16)
    shape = create_shape(state)
    Plots.plot!(plt, shape, fillalpha=0, linecolor=:black, linewidth=1.0, label="")
    state_mean = calculate_state_mean(state)
    Plots.annotate!(plt, state_mean[1], state_mean[2], text(label, color, :center, fontsize,))
end
