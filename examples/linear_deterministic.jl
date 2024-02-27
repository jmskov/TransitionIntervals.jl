using Distributions
using TransitionIntervals

include("utilities.jl")

# abstraction
linear_system_mat = [0.9 0.1; -0.2 0.8]
discretization = UniformDiscretization(DiscreteState([-1.0, -1.0], [1.0, 1.0]), [0.25, 0.25])
abstraction = transition_intervals(discretization, linear_system_mat)

spec_filename = "$(@__DIR__)/specs/bdd_until2.toml"
reach, avoid = terminal_states(spec_filename, abstraction.states)

amb_states = verify_and_plot(abstraction, spec_filename)
refinement_steps = 5
for _ in 1:refinement_steps
    global amb_states, abstraction
    abstraction = refine_abstraction(abstraction, linear_system_mat, discretization.compact_space, amb_states)
    amb_states = verify_and_plot(abstraction, spec_filename)
end
