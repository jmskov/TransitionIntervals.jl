using IntervalMDP
using IMDPs

using TransitionIntervals

function verify_and_plot(abstraction, spec_filename)
    reach, avoid = terminal_states(spec_filename, abstraction.states)
    result = verify(abstraction, reach, avoid)
    sat_states = findall(result[:,3] .> 0.99)
    plt = plot(abstraction.states[sat_states]; fillcolor=:blue, label="", aspect_ratio=1, linewidth=0, fillalpha=0.7)
    unsat_states = findall(result[:,4] .< 0.99)[1:end-1]
    plot!(plt, abstraction.states[unsat_states]; fillcolor=:red, label="", linewidth=0, fillalpha=0.7)
    amb_states = setdiff(1:length(result[:,1])-1, union(sat_states, unsat_states))
    plot!(plt, abstraction.states[amb_states]; fillcolor=:grey, label="", linewidth=0, fillalpha=0.5)
    display(plt)
    return amb_states
end

# outer function for IMC verification
function verify(abstraction::TransitionIntervals.Abstraction, terminal_states::Vector{Int}, avoid_set::Vector{Int}; tolerance=1e-6)
    
    prob = IntervalProbabilities(;
        lower = abstraction.Plow,
        upper = abstraction.Phigh,
    )
    mc = IntervalMarkovChain(prob, [-1])
    prop = InfiniteTimeReachAvoid(terminal_states, avoid_set, tolerance) 
    spec_low = Specification(prop, Pessimistic, Minimize)
    problem_low = Problem(mc, spec_low)
    Vlow, k, residual = value_iteration(problem_low)
    spec_high = Specification(prop, Optimistic, Maximize)
    problem_high = Problem(mc, spec_high)
    Vupp, k, residual = value_iteration(problem_high)

    result_matrix = zeros(length(Vlow), 4)
    result_matrix[:,1] = collect(1:length(Vlow))
    # todo: actions?
    result_matrix[:,3] = Vlow
    result_matrix[:,4] = Vupp

    # save the results
    # result_filename = "$results_dir/results-explicit.bson"
    # serialize(result_filename, result_matrix)

    return result_matrix
end

function get_reachability_labels(state_label_fcn, state_means, target_label, unsafe_label)
    terminal_states = Vector{Int64}()
    unsafe_states = Vector{Int64}()

    for (i, pt) in enumerate(state_means)
        if state_label_fcn(pt) == target_label 
            push!(terminal_states, i)
        end
        if state_label_fcn(pt) == unsafe_label 
            push!(unsafe_states, i)
        end
    end
    return terminal_states, unsafe_states
end

function terminal_states(spec_file::String, states::Vector{DiscreteState})
    # get the terminal states
    state_label_fcn, _, _, target_label, unsafe_label, _, _ = IMDPs.load_PCTL_specification(spec_file) 
    state_means = TransitionIntervals.mean.(states)
    terminal_state_idxs, unsafe_state_idxs = get_reachability_labels(state_label_fcn, state_means, target_label, unsafe_label)
    return terminal_state_idxs, unsafe_state_idxs
end


