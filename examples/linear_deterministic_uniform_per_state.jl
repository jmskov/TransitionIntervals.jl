using Distributions
using TransitionIntervals

include("utilities.jl")

# abstraction
linear_system_mat = [0.9 0.1; -0.2 0.8]
process_noise_dist = nothing # Normal(0.0, 0.01) 
# do learning here

function uncertainty_dist(lower, upper; thread_idx=1)
    return GaussianUniformError(0.01, 1.0)
end

discretization = UniformDiscretization(DiscreteState([-1.0, -1.0], [1.0, 1.0]), [0.5, 0.5])
abstraction = transition_intervals(discretization, linear_system_mat, uncertainty_dist)

spec_filename = "$(@__DIR__)/specs/bdd_until2.toml"
reach, avoid = terminal_states(spec_filename, abstraction.states)

amb_states = verify_and_plot(abstraction, spec_filename)
refinement_steps = 5
for _ in 1:refinement_steps
    global amb_states, abstraction
    abstraction = refine_abstraction(abstraction, linear_system_mat, discretization.compact_space, amb_states, uncertainty_dist)
    amb_states = verify_and_plot(abstraction, spec_filename)
end
