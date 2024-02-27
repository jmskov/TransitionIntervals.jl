# Tools for plotting

function create_shape(state::DiscreteState)
    shape = Plots.Shape([state.lower[1], state.upper[1], state.upper[1], state.lower[1]], 
                        [state.lower[2], state.lower[2], state.upper[2], state.upper[2]])
    return shape
end

function plot!(plt, state::DiscreteState; kwargs...)
    shp = create_shape(state)
    plt = Plots.plot!(plt, shp; kwargs...)
    return plt
end

function plot(state::DiscreteState; kwargs...)
    plt = Plots.plot(;kwargs...)
    plot!(plt, state; kwargs...)
    return plt
end

function plot!(plt, states::Vector{DiscreteState}; kwargs...)
    for state in states
        plt = plot!(plt, state; kwargs...)
    end
    return plt
end

function plot(states::Vector{DiscreteState}; kwargs...)
    plt = Plots.plot(;kwargs...)
    for state in states
        plt = plot!(plt, state; kwargs...)
    end
    return plt
end

function plot(states::Vector{DiscreteState}, alphas::Vector{Float64}; kwargs...)
    plt = Plots.plot(;kwargs...)
    for i in eachindex(states)
        plt = plot!(plt, states[i]; kwargs..., fillalpha=alphas[i])
    end
    return plt
end 

function annotate(states::Vector{DiscreteState}, state_labels::Dict{Int, String}; kwargs...)
    plt = plot(state; kwargs...)
    for (key, val) in state_labels
        annotate!(plt, states[key], val; kwargs...)
    end
    return plt
end

function annotate!(plt, state::DiscreteState, label::String; fontsize=5, kwargs...)
    state_mean = mean(state)
    Plots.annotate!(plt, state_mean[1], state_mean[2], text("$label", :black, :center, fontsize))
    return plt
end

function annotate_state_indeces!(plt::Plots.Plot, states::Vector{DiscreteState}; kwargs...)
    for (i, state) in enumerate(states)
        annotate!(plt, state, string(i); kwargs...)
    end
end

#     indet_color = colorant"#D6FAFF"
#     sat_color = colorant"#00AFF5"
#     unsat_color = colorant"#D55672"

function save_figure_files(plt, filename)
    savefig(plt, filename * ".png")
    serialize(filename * ".plt", plt)
end
