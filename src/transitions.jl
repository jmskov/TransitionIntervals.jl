function initialize_transition_matrices(nstates)
    return initialize_transition_matrices(nstates, nstates)
end

function initialize_transition_matrices(nstates_src::Int, nstates_dst::Int)
    Plow = spzeros(nstates_dst, nstates_src)
    Phigh = spzeros(nstates_dst, nstates_src)
    Plow[end,end] = 1.0
    Phigh[end,end] = 1.0
    return Plow, Phigh
end

function transition_targets(states::Vector{DiscreteState}, images::Vector{DiscreteState}, discretization::Discretization)
    ndims_in = length(states[1].lower)
    ndims_out = length(images[1].lower)
    if ndims_in > ndims_out
        n_assym = ndims_in - ndims_out
        # use discretization to find # of control regions
        num_control = 1
        for i = 1:n_assym
            del = discretization.compact_space.upper[end-i+1] - discretization.compact_space.lower[end-i+1]
            num_control *= Int(del / discretization.spacing[end-i+1])
        end
        num_states = Int(length(states) / num_control)
        targets = 1:num_states
    else
        targets = 1:length(states)
    end
    return targets
end

# gets the distance function
function get_distance_fcn()
    MULTIPLICATIVE_NOISE_FLAG = false
    if MULTIPLICATIVE_NOISE_FLAG
        distances_function = multiplicative_noise_distances
    else
        distances_function = additive_noise_distances!
    end

    # todo: add static partitions here
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

function calc_distances!(buffer::Matrix{Float64}, image::DiscreteState, state::DiscreteState)
    distances_function = get_distance_fcn()
    for i in axes(image.lower, 1)
         @views distances_function(image.lower[i], image.upper[i], state.lower[i], state.upper[i], buffer[i,:])
    end
    return buffer
end

function buffer_distances!(distances::Matrix{Float64}, val::Float64)
    for i in axes(buffer,1)
        buffer[i,col] += val
    end
    check_distances!(distances)
    return distances
end

function check_distances!(distances::Matrix{Float64})
    for i in axes(distances, 1)
        if distances[i,3] < distances[i,4]
            distances[i,3] = distances[i,4] = 0
        end
    end
end

function verify_interval!(p_vector::Vector{Float64}; eps=1e-10)
    if p_vector[1] > p_vector[2]
        if abs(p_vector[1] - p_vector[2]) < eps
            p_vector[1] = p_vector[2]
        else
            @error "p_low > p_high"
        end
    end
    return p_vector
end

#== deterministic interval ==#
function transition_interval(image::DiscreteState, state::DiscreteState, p_buffer::Vector{Float64}=zeros(2))
    p_low_total = contains(state, image)
    p_high_total = intersects(state, image) 

    p_vector = check_buffer(p_buffer, (2,))
    p_vector[1] = check_zero_one_numerical(p_low_total) 
    p_vector[2] = check_zero_one_numerical(p_high_total)
    verify_interval!(p_vector)

    return p_vector
end

#== stochastic transition interval ==#
function transition_interval(image::DiscreteState, state::DiscreteState, process_dist::Distribution, p_buffer::Vector{Float64}=zeros(2), distance_buffer::Matrix{Float64}=zeros(length(image.lower),4))
    dis_comps = check_buffer(distance_buffer, (length(image.lower), 4))
    calc_distances!(dis_comps, image, state)

    p_low_total = 1
    p_high_total = 1
    for i in axes(image.lower, 1)
        p_low_total *= cdf_interval(process_dist, dis_comps[i,4], dis_comps[i,3])
        p_high_total *= cdf_interval(process_dist, dis_comps[i,2], dis_comps[i,1])
    end

    p_vector = check_buffer(p_buffer, (2,))
    p_vector[1] = check_zero_one_numerical(p_low_total) 
    p_vector[2] = check_zero_one_numerical(p_high_total)
    verify_interval!(p_vector)

    return p_vector
end

#== uniform error transition interval ==#
function transition_interval(image::DiscreteState, state::DiscreteState, uniform_dist::UniformError, p_buffer::Vector{Float64}=zeros(2), distance_buffer::Matrix{Float64}=zeros(length(image.lower),4))
    dis_comps = check_buffer(distance_buffer, (length(image.lower), 4))
    calc_distances!(dis_comps, image, state)

    p_low_total = 1
    p_high_total = 1

    cont_flag = contains(state, image)
    int_flag = intersects(state, image)

    for i in axes(image.lower, 1) 
        if cont_flag 
            p_low_total *= cdf(uniform_dist, min(abs(dis_comps[i,4]), abs(dis_comps[i,3])))
        else
            p_low_total *= 0
        end

        if int_flag 
            p_high_total *= 0
        else
            p_high_total *= cdf(uniform_dist, min(abs(dis_comps[i,2]), abs(dis_comps[i,1]))) 
        end 
    end

    p_high_total = 1 - p_high_total

    p_vector = check_buffer(p_buffer, (2,))
    p_vector[1] = check_zero_one_numerical(p_low_total) 
    p_vector[2] = check_zero_one_numerical(p_high_total)
    verify_interval!(p_vector)

    return p_vector
end

#== stochastic + uniform error transition interval ==#
function transition_interval(image::DiscreteState, state::DiscreteState, process_dist::Distribution, uniform_dist::Distribution, p_buffer::Vector{Float64}=zeros(2), distance_buffer::Matrix{Float64}=zeros(length(image.lower),4))
    dis_comps = check_buffer(distance_buffer, (length(image.lower), 4))
    calc_distances!(dis_comps, image, state)

    p_low_total = 1
    p_high_total = 1
    for i in axes(image.lower, 1)
        @views p_low_total *= optimize_dim_low(dis_comps[i,:], process_dist, uniform_dist)
        @views p_high_total *= optimize_dim_high(dis_comps[i,:], process_dist, uniform_dist)
    end

    p_vector = check_buffer(p_buffer, (2,))
    p_vector[1] = check_zero_one_numerical(p_low_total) 
    p_vector[2] = check_zero_one_numerical(p_high_total)
    verify_interval!(p_vector)

    return p_vector
end

function optimize_dim_low(distances, w_dist, v_dist; ep_resolution::Float64=0.001)

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

        p_low = cdf_interval(w_dist, l_new, u_new)*cdf(v_dist, ep) 
        if p_low > p_best
            p_best = p_low
        end
        if p_low < p_best
            return p_best
        end
    end
    return p_best
end

function new_opt(dist1, dist2, l, u)
    max_iter = 30
    ep_start = 2.0
    ep = ep_start

    min_iter = 15
    iter = 0

    val = totalP(dist1, dist2, l, u, ep)

    # else, start working down!
    del_factor = 2
    del_steps = 2
    for i in 1:max_iter
        ep_low = ep/del_factor
        new_val = totalP(dist1, dist2, l, u, ep_low)
        if new_val <= val
            if new_val < 1.0 && abs(new_val - val) < 1e-2 && iter > min_iter
                return ep_low, new_val
            end
            val = new_val
            ep = ep_low
            # del_steps = 2
            # del_factor = 2
        else
            del_factor = 2^(1/del_steps)
            del_steps += 1
        end
        iter += 1
    end

    return ep, val
end

function totalP(dist1, dist2, l, u, ep)
    p2 = cdf(dist2, ep)
    return cdf_interval(dist1, l-ep, u+ep)*p2 + 1-p2
end

function optimize_dim_high(distances, w_dist, v_dist)
    _, p_best = new_opt(w_dist, v_dist, distances[2], distances[1])
    return p_best
end

function transition_col!(plow_col, phigh_col, states::Vector{DiscreteState}, image::DiscreteState, full_state::DiscreteState, p_buffer=zeros(2); targets=1:length(states))
    for i in targets
        p_buffer = transition_interval(image, states[i], p_buffer)
        plow_col[i] = p_buffer[1]
        phigh_col[i] = p_buffer[2]
    end
    p_buffer = transition_interval(image, full_state, p_buffer)
    plow_col[end] = 1 - p_buffer[2]
    phigh_col[end] = 1 - p_buffer[1]
    return plow_col, phigh_col
end

# todo: the following has a lot of repeated code. consolidate, but smartly
# calculate transition probabilities, deterministic
function calculate_transition_probabilities(states::Vector{DiscreteState}, images::Vector{DiscreteState}, discretization::Discretization)
    targets = transition_targets(states, images, discretization)
    num_src = length(states) + 1
    num_dst = length(targets) + 1
    Plow, Phigh = initialize_transition_matrices(num_src, num_dst)
    progress_meter = Progress(num_dst, desc="Computing transition intervals...", dt=STATUS_BAR_PERIOD, enabled=ENABLE_PROGRESS_BAR)

    p_buffer = zeros(2)
    # todo: parallelize this and test - might need buffers and then sum,,,
    for (i, image) in enumerate(images)
        Plow[:,i], Phigh[:,i] = transition_col!(Plow[:,i], Phigh[:,i], states, image, discretization.compact_space, p_buffer, targets=targets)
        next!(progress_meter)
    end

    validate_transition_matrices(Plow, Phigh)

    return Plow, Phigh
end

function transition_col!(plow_col, phigh_col, states::Vector{DiscreteState}, image::DiscreteState, full_state::DiscreteState, process_dist::Distribution, p_buffer=zeros(2), distance_buffer=zeros(length(image.lower),4); targets=1:length(states))
    for i in targets
        p_buffer = transition_interval(image, states[i], process_dist, p_buffer, distance_buffer)
        plow_col[i] = p_buffer[1]
        phigh_col[i] = p_buffer[2]
    end
    p_buffer = transition_interval(image, full_state, process_dist, p_buffer, distance_buffer)
    plow_col[end] = 1 - p_buffer[2]
    phigh_col[end] = 1 - p_buffer[1]
    return plow_col, phigh_col
end

# calculate transition probabilities, process noise
function calculate_transition_probabilities(states::Vector{DiscreteState}, images::Vector{DiscreteState}, discretization::Discretization, process_dist::Distribution)
    targets = transition_targets(states, images, discretization)
    num_src = length(states) + 1
    num_dst = length(targets) + 1
    Plow, Phigh = initialize_transition_matrices(num_src, num_dst)
    progress_meter = Progress(num_dst, desc="Computing transition intervals...", dt=STATUS_BAR_PERIOD, enabled=ENABLE_PROGRESS_BAR)

    p_buffer = zeros(2)
    distance_buffer = zeros(length(images[1].lower), 4)

    for (i, image) in enumerate(images)
        Plow[:,i], Phigh[:,i] = transition_col!(Plow[:,i], Phigh[:,i], states, image, discretization.compact_space, process_dist, p_buffer, distance_buffer, targets=targets)
        next!(progress_meter)
    end

    validate_transition_matrices(Plow, Phigh)

    return Plow, Phigh
end

function transition_col!(plow_col, phigh_col, states::Vector{DiscreteState}, image::DiscreteState, full_state::DiscreteState, process_dist::Distribution, uniform_dist::Distribution, p_buffer=zeros(2), distance_buffer=zeros(length(image.lower),4); targets=1:length(states))
    for i in targets
        p_buffer = transition_interval(image, states[i], process_dist, uniform_dist, p_buffer, distance_buffer)
        plow_col[i] = p_buffer[1]
        phigh_col[i] = p_buffer[2]
    end
    p_buffer = transition_interval(image, full_state, process_dist, uniform_dist, p_buffer, distance_buffer)
    plow_col[end] = 1 - p_buffer[2]
    phigh_col[end] = 1 - p_buffer[1]
    return plow_col, phigh_col
end

function calculate_transition_probabilities(states::Vector{DiscreteState}, images::Vector{DiscreteState}, discretization::Discretization, process_dist::Distribution, uniform_error_dist::Union{Function, UniformError})
    targets = transition_targets(states, images, discretization)
    num_src = length(states) + 1
    num_dst = length(targets) + 1
    Plow, Phigh = initialize_transition_matrices(num_src, num_dst)
    progress_meter = Progress(num_dst, desc="Computing transition intervals...", dt=STATUS_BAR_PERIOD, enabled=ENABLE_PROGRESS_BAR)

    p_buffer = zeros(2)
    distance_buffer = zeros(length(images[1].lower), 4)

    for (i, image) in enumerate(images)
        
        if uniform_error_dist isa Function
            @views state_dep_dist = uniform_error_dist(states[i].lower, states[i].upper, thread_idx=1)
        else
            state_dep_dist = uniform_error_dist
        end

        Plow[:,i], Phigh[:,i] = transition_col!(Plow[:,i], Phigh[:,i], states, image, discretization.compact_space, process_dist, state_dep_dist, p_buffer, distance_buffer, targets=targets)
        next!(progress_meter)
    end

    next!(progress_meter)
    validate_transition_matrices(Plow, Phigh)

    return Plow, Phigh
end


function calculate_transition_probabilities(states::Vector{DiscreteState}, images::Vector{DiscreteState}, discretization::Discretization, uniform_error_dist::Union{Function, UniformError})
    targets = transition_targets(states, images, discretization)
    num_src = length(states) + 1
    num_dst = length(targets) + 1
    Plow, Phigh = initialize_transition_matrices(num_src, num_dst)
    progress_meter = Progress(num_dst, desc="Computing transition intervals...", dt=STATUS_BAR_PERIOD, enabled=ENABLE_PROGRESS_BAR)

    p_buffer = zeros(2)
    distance_buffer = zeros(length(images[1].lower), 4)

    for (i, image) in enumerate(images)
        if uniform_error_dist isa Function
            @views state_dep_dist = uniform_error_dist(states[i].lower, states[i].upper, thread_idx=1)
        else
            state_dep_dist = uniform_error_dist
        end

        Plow[:,i], Phigh[:,i] = transition_col!(Plow[:,i], Phigh[:,i], states, image, discretization.compact_space, state_dep_dist, p_buffer, distance_buffer, targets=targets)
        next!(progress_meter)
    end

    next!(progress_meter)
    validate_transition_matrices(Plow, Phigh)

    return Plow, Phigh
end