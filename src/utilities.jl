# Utility functions 

# custom distribution
struct UniformError <: Distribution{Univariate, Continuous} 
    sigma::Float64
    constant::Float64
end

function cdf(d::UniformError, x::Real)
    if x < 0.0
        return 0.0
    end
    return cdf(Normal(0.0, d.sigma), x) - cdf(Normal(0.0, d.sigma), -x)
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
    for row in eachrow(Plow)
        @assert sum(row) <= 1.0
    end
    for row in eachrow(Phigh)
        @assert sum(row) >= 1.0
    end
end
