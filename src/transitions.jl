# gets the distance function
function get_distance_fcn()
    MULTIPLICATIVE_NOISE_FLAG = false
    if MULTIPLICATIVE_NOISE_FLAG
        distances_function = multiplicative_noise_distances
    else
        distances_function = Stochascape.additive_noise_distances!
    end

    # todo: add static partitions here
    # todo: this is a hack. I need to do this better, out of here
    # if USE_STATIC_PARTITIONS
    #     if !(all(STATIC_PARTITION_BOUNDS[1] .>= dis_comps[:,4]) && all(STATIC_PARTITION_BOUNDS[2] .<= dis_comps[:,3]))
    #         dis_comps[:,4] .= 0 
    #         dis_comps[:,3] .= 0
    #     else
    #         dis_comps[:,4] .= STATIC_PARTITION_BOUNDS[1]
    #         dis_comps[:,3] .= STATIC_PARTITION_BOUNDS[2]
    #     end

    #     if all(STATIC_PARTITION_BOUNDS[1] .<= dis_comps[:,2]) && all(STATIC_PARTITION_BOUNDS[2] .>= dis_comps[:,1])
    #         dis_comps[:,2] .= STATIC_PARTITION_BOUNDS[1]
    #         dis_comps[:,1] .= STATIC_PARTITION_BOUNDS[2]
    #     else
    #         dis_comps[:,2] .= -Inf
    #         dis_comps[:,1] .= Inf
    #     end
    # else
        
    # end

    return distances_function
end

function calc_distances(image, state, buffer)
    distances_function = get_distance_fcn()
    for i in axes(image, 1)
         @views distances_function(image[i,1], image[i,2], state[i,1], state[i,2], buffer[i,:])
    end
    return buffer
end

function buffer_distances!(distances, val)
    for i in axes(buffer,1)
        buffer[i,col] += val
    end
    check_distances!(distances)
    return distances
end

function check_distances!(distances)
    for i in axes(distances, 1)
        if distances[i,3] < distances[i,4]
            distances[i,3] = distances[i,4] = 0
        end
    end
end

#==
    dual_optimize 

This function calculates the optimal (minimum-width) transition intervals for an image and target state. It returns a vector of probabilities, where the first element is the lower-bound probability, and the second element is the upper-bound probability.
==#
function optimal_transition_interval(image::Matrix{Float64}, state::Matrix{Float64}, w_dist::Distribution=Normal(0.0, 0.0), p_buffer::Vector{Float64}=zeros(2), distance_buffer::Matrix{Float64}=zeros(length(image),4), v_dist::Distribution=Normal(0.0, 0.0))

    # todo: update this method to handle static partitions

    # sweep method
    p_low_total = 1
    p_high_total = 1

    distances_function = get_distance_fcn()
    ndims = size(image,1)
    dis_comps = check_buffer(distance_buffer, (ndims, 4))
    for i in axes(image, 1)
        @views distances_function(image[i,1], image[i,2], state[i,1], state[i,2], dis_comps[i,:])
    end

    for i in axes(image, 1)
        @views p_low_total *= optimize_dim_low(dis_comps[i,:], w_dist, v_dist)
        @views p_high_total *= optimize_dim_high(dis_comps[i,:], w_dist, v_dist)
    end

    p_vector = check_buffer(p_buffer, (2,))
    p_vector[1] = check_zero_one_numerical(p_low_total) 
    p_vector[2] = check_zero_one_numerical(p_high_total)
    @assert p_vector[1]  <= p_vector[2]
    return p_vector
end


function optimize_dim_low(distances, w_dist, v_dist; ep_resolution::Float64=0.01)

    ep_max = distances[3] - distances[4]
    ep_sweep = 0.0:ep_resolution:ep_max
    p_best = 0.0

    for ep in ep_sweep
        l_new = distances[4] + ep
        u_new = distances[3] - ep
        if l_new > u_new
            l_new = 0.0
            u_new = 0.0
        end

        p_low = cdf_interval(w_dist, l_new, u_new)*cdf_interval(v_dist, -ep, ep) 
        if p_low > p_best
            p_best = p_low
        end
        if p_low < p_best
            return p_best
        end
    end
    return p_best
end

function optimize_dim_high(distances, w_dist, v_dist; ep_resolution::Float64=0.01)
    ep_max = distances[1] - distances[2]
    ep_sweep = 0.0:ep_resolution:ep_max
    p_best = 1.0

    for ep in ep_sweep
        l_new = distances[2] - ep
        u_new = distances[1] + ep

        v_prob = cdf_interval(v_dist, -ep, ep)
        p_high = cdf_interval(w_dist, l_new, u_new)*v_prob + (1-v_prob) 
        if p_high < p_best
            p_best = p_high
        end
        if p_high > p_best 
            return p_best
        end
    end
    return p_best
end

function simple_transition_interval(image::Matrix{Float64}, state::Matrix{Float64}, w_dist::Distribution, p_buffer::Vector{Float64}=zeros(2), distance_buffer::Matrix{Float64}=zeros(length(image),4), v_dist::Union{Nothing, Distribution}=nothing, ep::Float64=0.0)
    distances_function = get_distance_fcn()
    ndims = size(image,1)
    dis_comps = check_buffer(distance_buffer, (ndims, 4))

    for i in axes(image, 1)
        @views distances_function(image[i,1], image[i,2], state[i,1], state[i,2], dis_comps[i,:])
    end
    ep > 0 ? buffer_distances!(dis_comps, ep) : nothing
    v_prob = isnothing(v_dist) ? 1.0 : cdf_interval(v_dist, -ep, ep)
    v_prob_c = 1 - v_prob
    p_low = 1
    p_high = 1
    for i in axes(image, 1)
        p_low *= cdf_interval(w_dist, dis_comps[i,4], dis_comps[i,3])*v_prob
        p_high *= cdf_interval(w_dist, dis_comps[i,2], dis_comps[i,1])*v_prob + v_prob_c
    end
    p_vector = check_buffer(p_buffer, (2,))
    p_vector[1] = check_zero_one_numerical(p_low) 
    p_vector[2] = check_zero_one_numerical(p_high)
    @assert p_vector[1]  <= p_vector[2]
    return p_vector
end

function calculate_transition_probabilities(states::Vector{Matrix{Float64}}, images::Vector{Matrix{Float64}}, full_state::Matrix{Float64}, process_dist::Distribution, state_dep_sigmas::Union{Nothing, Vector{Float64}}=nothing)
    nstates = length(states)+1
    Plow, Phigh = initialize_transition_matrices(nstates)
    n_transitions = nstates^2
    progress_meter = Progress(n_transitions, desc="Computing transition intervals...", dt=STATUS_BAR_PERIOD)

    p_buffer = zeros(2)
    distance_buffer = zeros(size(states[1],1), 4)

    for (i, image) in enumerate(images)
        if isnothing(state_dep_sigmas)
            state_dep_dist = nothing
        else
            state_dep_dist = UniformError(state_dep_sigmas[i], 1.0)
        end

        for (j, state) in enumerate(states)
            p_vector = optimal_transition_interval(image, state, process_dist, p_buffer, distance_buffer, state_dep_dist)
            Plow[j,i] = p_vector[1]
            Phigh[j,i] = p_vector[2]
            next!(progress_meter)
        end
    
        # to the bad state
        p_buffer = optimal_transition_interval(image, full_state, process_dist, p_buffer, distance_buffer, state_dep_dist)
        Plow[end,i] = 1 - p_buffer[2]
        Phigh[end,i] = 1 - p_buffer[1]
        next!(progress_meter)
    end

    Plow[end,end] = 1.0
    Phigh[end, end] = 1.0
    next!(progress_meter)
    validate_transition_matrices(Plow, Phigh)

    return Plow, Phigh
end
