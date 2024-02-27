using Distributions
using TransitionIntervals
import TransitionIntervals.cdf

include("utilities.jl")

# abstraction
linear_system_mat = [0.9 0.1; -0.2 0.8]

mutable struct UniformRKHSError <: UniformError 
    sigma::Float64
    info_bound::Float64
    f_sup::Float64
    kernel_length::Float64
    norm_bound::Float64
    log_noise::Float64
end

function cdf(d::UniformRKHSError, x::Real)
    if x < 0.0
        return 0.0
    end
    R = exp(d.log_noise)
    frac = x/(d.sigma)
    dbound = exp(-0.5*(1/R*(frac - d.norm_bound))^2 + d.info_bound + 1.)
    return 1. - min(dbound, 1.) # is this min needed?
end

# create a dummy distribution
uncertainty_dist = UniformRKHSError(0.001, 1.0, 1.0, 1.0, 1.0, 0.0)

discretization = UniformDiscretization(DiscreteState([-1.0, -1.0], [1.0, 1.0]), [0.5, 0.5])
abstraction = transition_intervals(discretization, linear_system_mat, uncertainty_dist)

spec_filename = "$(@__DIR__)/specs/bdd_until2.toml"
reach, avoid = terminal_states(spec_filename, abstraction.states)

amb_states = verify_and_plot(abstraction, spec_filename)
refinement_steps = 4
for _ in 1:refinement_steps
    global amb_states, abstraction
    abstraction = refine_abstraction(abstraction, linear_system_mat, discretization.compact_space, amb_states, uncertainty_dist)
    amb_states = verify_and_plot(abstraction, spec_filename)
end
