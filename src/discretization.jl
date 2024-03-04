# functions related to discretization

# state structure
struct DiscreteState
    lower::Vector{Float64}
    upper::Vector{Float64}
end

abstract type Discretization end;

# components that define a uniform discretization
struct UniformDiscretization <: Discretization
    compact_space::DiscreteState 
    spacing::Vector{Float64}    # todo: better name?
end

#= 
generate grids
=#
function grid_generator(discretization::UniformDiscretization)
    return grid_generator(discretization.compact_space.lower, discretization.compact_space.upper, discretization.spacing)
end

function grid_generator(L::Vector{Float64}, U::Vector{Float64}, δ::Vector{Float64})
	dim_generators = []
	for i in eachindex(L)
        push!(dim_generators, dimension_grid(L[i], U[i], δ[i]))
	end
	return Iterators.product(dim_generators...)
end

function dimension_grid(L::Float64, U::Float64, δ_desired::Float64)
    N_tight = Int(floor((U - L)/δ_desired))
    δ_tight = (U - L)/N_tight
    return L:δ_tight:(U-δ_tight) 
end

# 1-D
function explicit_states(grid::StepRangeLen)
    states = Vector{DiscreteState}(undef, length(grid))
    for (i, grid_point) in enumerate(grid)
        states[i] = DiscreteState([grid_point], [grid_point + grid.step.hi]) 
    end
    return states
end

#n-D
function explicit_states(grid::Iterators.ProductIterator)
    n = length(grid.iterators)
    states = Vector{DiscreteState}(undef, length(grid))
    for (i, grid_point) in enumerate(grid)
        state = DiscreteState(zeros(Float64, n), zeros(Float64, n)) 
        for j in 1:n
            state.lower[j] = grid_point[j]
            state.upper[j] = grid_point[j] + grid.iterators[j].step.hi
        end
        states[i] = state
    end
    return states
end

# both
function discretize(discretization::Discretization)
    grid = grid_generator(discretization)
    return explicit_states(grid)
end

"""
    intersects
"""
function intersects(state1::DiscreteState, state2::DiscreteState)
    # check if the two states intersect
    dims = length(state2.lower)
    for dim = 1:dims
        if state1.lower[dim] >= state2.upper[dim] || state1.upper[dim] <= state2.lower[dim]
            return false
        end
    end
    return true
end

"""
    contains
"""
function contains(state1::DiscreteState, state2::DiscreteState)
    # check if state1 contains state2
    dims = length(state2.lower)
    for dim = 1:dims
        if state1.lower[dim] > state2.lower[dim] || state1.upper[dim] < state2.upper[dim]
            return false
        end
    end
    return true
end

"""
    mean
"""
function mean(state::DiscreteState)
    state_mean = 0.5*(state.upper + state.lower)
    return state_mean
end

"""
    radius
"""
function radius(state::DiscreteState)
    return 0.5*sqrt(sum((state.upper - state.lower).^2))
end