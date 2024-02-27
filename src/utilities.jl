# Utility functions 

abstract type UniformError <: Distribution{Univariate, Continuous} end

# custom distribution
struct GaussianUniformError <: UniformError 
    sigma::Float64
    scale::Float64
end

# generic function for uniform error prototyping, based on the normal distribution
function cdf(d::GaussianUniformError, x::Real)
    if x < 0.0
        return 0.0
    end
    return cdf(Normal(0.0, d.scale*d.sigma), x) - cdf(Normal(0.0, d.scale*d.sigma), -x)
end

function initialize_buffers(num_states::Int, num_dims::Int)
    num_threads = Threads.nthreads()
    P_low_buffers = [spzeros(num_states, num_states) for _=1:num_threads]
    P_high_buffers = [spzeros(num_states, num_states) for _=1:num_threads]
    p_buffers = [zeros(2) for _=1:num_threads]
    distance_buffers = [zeros(num_dims, 4) for _=1:num_threads]
    return P_low_buffers, P_high_buffers, p_buffers, distance_buffers
end

# checks if a buffer is initialized and returns a zero array if not
function check_buffer(buffer::Union{Nothing, AbstractArray{Float64}}, size::Tuple{Vararg{Int}})
    if isnothing(buffer)
        return zeros(size)
    else
        return buffer
    end
end

function cdf_interval(distribution::Distribution, a::Real, b::Real)
    @assert a <= b
    return cdf(distribution, b) - cdf(distribution, a)
end

function check_zero_numerical(val, eps=1e-10)
    if val < 0.0 && val > -eps
        return 0.0
    else
        @assert val >= -eps
        return val
    end
end

function check_one_numerical(val, eps=1e-10)
    if val > 1.0 && val < 1.0+eps
        return 1.0
    else
        @assert val <= 1.0+eps
        return val
    end
end

function check_zero_one_numerical(val, eps=1e-10)
    val = check_zero_numerical(val, eps)
    val = check_one_numerical(val, eps)
    return val
end

function validate_transition_matrices(Plow, Phigh)
    for col in eachcol(Plow)
        @assert sum(col) <= 1.0
    end
    for col in eachcol(Phigh)
        @assert sum(col) >= 1.0
    end
end

"""
    true_transition_probabilities
"""
function true_transition_probabilities(pmin::AbstractVector, pmax::AbstractVector, indeces::AbstractVector)

    @assert length(pmin) == length(pmax) == length(indeces)

    p = zeros(size(indeces))
    used = sum(pmin[indeces])
    remain = 1 - used

    for i in indeces
        if pmax[i] <= (remain + pmin[i])
            p[i] = pmax[i]
        else
            p[i] = pmin[i] + remain
        end
        remain = max(0, remain - (pmax[i] - pmin[i]))
    end

    if !(sum(p)≈1)
        @info remain
        @info pmin
        @info pmax
    end
    @assert sum(p) ≈ 1

    return p
end
