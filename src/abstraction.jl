# Functions for abstraction
Base.@kwdef struct AbstractionProblem
    discreization::Discreteization
    image_map::Function # function
    process_noise_distribution::Union{Nothing, Distribution} = nothing
    uniform_error_distribution::Union{Nothing, Function, Distribution} = nothing
end

Base.@kwdef struct Abstraction
    states::Vector{DiscreteState}
    images::Vector{DiscreteState}
    Plow::SparseMatrixCSC{Float64,Int64}
    Phigh::SparseMatrixCSC{Float64,Int64}
end

function transition_intervals(problem::AbstractionProblem)
    return transition_intervals(problem.discreization, problem.image_map, problem.process_noise_distribution, problem.uniform_error_distribution)
end

function transition_intervals(discretization::Discreteization, image_map::Union{Function, Matrix{Float64}})
    return transition_intervals(discretization, image_map, nothing, nothing)
end

function transition_intervals(discretization::Discreteization, image_map::Union{Function, Matrix{Float64}}, noise_distribution::Distribution)
    return transition_intervals(discretization, image_map, noise_distribution,  nothing)
end

function transition_intervals(discretization::Discreteization, image_map::Union{Function, Matrix{Float64}}, uniform_error_distribution::Union{Nothing, Function, Distribution})
    grid = grid_generator(discretization)
    states = explicit_states(grid) 
    return transition_intervals(states, discretization, image_map, nothing, uniform_error_distribution)
end

function transition_intervals(discretization::Discreteization, image_map::Union{Function, Matrix{Float64}}, noise_distribution::Union{Nothing, Distribution}, uniform_error_distribution::Union{Nothing, Function, Distribution})
    grid = grid_generator(discretization)
    states = explicit_states(grid) 
    return transition_intervals(states, discretization, image_map, noise_distribution, uniform_error_distribution)
end

#== Given an Image Map ==#
function transition_intervals(states::Vector{DiscreteState}, discretization::Discreteization, image_map::Function, noise_distribution::Union{Nothing, Distribution}, uniform_error_distribution::Union{Nothing, Function, Distribution})
    images = state_images(states, image_map) 
    return transition_intervals(states, images, discretization, noise_distribution, uniform_error_distribution)
end

#== Given a Linear System ==#
function transition_intervals(states::Vector{DiscreteState}, discretization::Discreteization, system_matrix::Matrix{Float64}, noise_distribution::Union{Nothing, Distribution}, uniform_error_distribution::Union{Nothing, Function, Distribution})
    images = state_images(states, system_matrix) 
    return transition_intervals(states, images, discretization, noise_distribution, uniform_error_distribution)
end

function transition_intervals(states::Vector{DiscreteState}, images::Vector{DiscreteState}, discretization::Discreteization,  noise_distribution::Union{Nothing, Distribution}, uniform_error_distribution::Union{Nothing, Function, Distribution}) 

    if isnothing(noise_distribution) && isnothing(uniform_error_distribution)
        Plow, Phigh = calculate_transition_probabilities(states, images, discretization.compact_space)
    elseif isnothing(uniform_error_distribution)
        Plow, Phigh = calculate_transition_probabilities(states, images, discretization.compact_space, noise_distribution)
    elseif isnothing(noise_distribution)
        Plow, Phigh = calculate_transition_probabilities(states, images, discretization.compact_space, uniform_error_distribution)
    else
        Plow, Phigh = calculate_transition_probabilities(states, images, discretization.compact_space, noise_distribution, uniform_error_distribution)
    end

    return Abstraction(states, images, Plow, Phigh)
end
