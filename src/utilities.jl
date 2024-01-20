# Utility functions 

# cdf table for epsilon values
EPSILON_DEFAULT = 0:0.01:2.0
CDF_DICT = Dict{Float64, Float64}()

function initialize_cdf_table(dist::Distribution)
    for eps in EPSILON_DEFAULT
        CDF_DICT[eps] = cdf(dist, eps)
    end
end

# custom distribution
# ! update for RKHS error
struct UniformError <: Distribution{Univariate, Continuous} 
    sigma::Float64
    scale::Float64
    constant::Float64
end

function cdf(d::UniformError, x::Real)
    if x < 0.0
        return 0.0
    end
    return cdf(Normal(0.0, d.scale*d.sigma), x) - cdf(Normal(0.0, d.scale*d.sigma), -x)
end

# todo: prototype the RKHS uniform error...
mutable struct UniformRKHSError <: Distribution{Univariate, Continuous} 
    sigma::Float64
    scale_factor::Float64
    constant::Float64
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

    R = d.scale_factor*exp(d.log_noise)
    frac = x/(d.scale_factor*d.sigma)
	if frac > d.norm_bound
    	dbound = exp(-0.5*(1/R*(frac - d.norm_bound))^2 + d.info_bound + 1.)
	else
		dbound = 1.0
	end

    # @assert dbound < 1.
    return 1. - min(dbound, 1.) # is this min needed?
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
