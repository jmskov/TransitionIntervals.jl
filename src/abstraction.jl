# Functions for abstraction

# Full abstraction
function imc_abstraction(full_state, spacing, image_map, noise_distribution)
    grid, grid_spacing = grid_generator(full_state[:,1], full_state[:,2], spacing)
    states = calculate_explicit_states(grid, grid_spacing)
    images = calculate_all_images(states, image_map)
    Plow, Phigh = calculate_transition_probabilities(states, images, full_state, noise_distribution)
    return states, images, Plow, Phigh
end

# States
function grid_generator(L, U, δ)
	generator = nothing
	dim_generators = []
    δ_tight = zeros(size(δ))

    # δ is just the desired discretization - to make it work with the L and U bounds, we can adjust it slightly.
	for i=1:length(L)
        N_tight = Int(floor((U[i] - L[i])/δ[i]))
        δ_tight[i] = (U[i] - L[i])/N_tight
        gen = L[i]:δ_tight[i]:U[i]   # changed to arrays
        push!(dim_generators, gen[1:end-1])
	end

	generator =  Iterators.product(dim_generators...)
	return generator, δ_tight
end

function calculate_explicit_states(grid, grid_spacing)
    # initialize an array of nx2 matrices
    n = length(grid_spacing)
    states = Array{Array{Float64,2},1}(undef, length(grid))
    for (i, grid_point) in enumerate(grid)
        state = zeros(Float64, n, 2)
        for j in 1:n
            state[j,1] = grid_point[j]
            state[j,2] = grid_point[j] + grid_spacing[j]
        end
        states[i] = state
    end
    return states
end

function calculate_all_images(explicit_states, user_defined_map)
    progress_meter = Progress(length(explicit_states), "Computing state images...", dt=STATUS_BAR_PERIOD)
    images = Array{Array{Float64,2},1}(undef, length(explicit_states))

    Threads.@threads for i in eachindex(explicit_states)
        try
            images[i] = user_defined_map(explicit_states[i], thread_idx=Threads.threadid()) 
        catch 
            images[i] = user_defined_map(explicit_states[i]) 
        end 
        next!(progress_meter)
    end
    return images
end

function calculate_state_mean(state)
    state_mean = 0.5*(state[:,2] + state[:,1])
    return state_mean
end

function label_imdp_states(imdp, label_fcn, states)
    # label the states
    for (i, state) in enumerate(states) 
        state_mean = calculate_state_mean(state)
        label = label_fcn(state_mean)
        imdp.labels[i] = label
    end
end

# Transitions
function additive_noise_distances(image, target)
    A, B = target
    C, D = image
    W = D - C
    @assert W > 0

    Δ1 = abs(B-C)
    Δ2 = -abs(A-D)
    # check if target contains image
    if target[1] <= image[1] <= image[2] <= target[2]
        Δ3 = abs(B-D)
        Δ4 = -abs(A-C)
    # check if image is larger than target
    elseif image[1] <= target[1] <= target[2] <= image[2]
        Δ3 = 0  
        Δ4 = 0
    else
        if D > B # image UB is greather than state UB 
            if !(A <= C <= B)
                Δ1 *= -1
            end
            Δ3 = -abs(B-D)
            Δ4 = -abs(A+W-D)
        else
            if !(A <= D <= B)
                Δ2 *= -1
            end
            Δ3 = abs(B-W-C)
            Δ4 = abs(A-C) 
        end
        if Δ4 > Δ3 
            Δ3 = 0
            Δ4 = 0
        end
    end
    
    @assert Δ2 <= Δ1
    @assert Δ4 <= Δ3
    return Δ1, Δ2, Δ3, Δ4
end

function multiplicative_noise_distances(image, target)
    A, B = target
    C, D = image

    # let's see how this ai generated code works...
    # Todo: Δ1 can also be things not Inf possibly...
    if C > 0 && D > 0
        Δ1 = B / C
    elseif C < 0 && D < 0
        Δ1 = A / D
    else
        # if B/C <= 0 && A / D <= 0
        Δ1 = Inf
        # else
            # Δ1 = max(B/C, A/D)
        # end
    end
    Δ2 = ifelse(C > 0 && D > 0, A/D, ifelse(C < 0 && D < 0, B/C, 0))
    # Δ4 = ifelse(C > 0 && D > 0, A/C, ifelse(D < 0 && C < 0, B/D, 0))
    # Δ3 = ifelse(C > 0 && D > 0, B/D, ifelse(C < 0 && D < 0, A/C, min(max(A/C, 0), max(B/D, 0))))

    Δ3 = B/D
    Δ4 = A/C
    
    if Δ4 > Δ3
        Δ3 = 0
        Δ4 = 0
    end

    if Δ2 > Δ1
        Δ1 = Inf
        Δ2 = 0
    end

    # check for NaN...
    if isnan(Δ1)
        Δ1 = Inf
    end
    if isnan(Δ2)
        Δ2 = 0
    end
    if isnan(Δ3)
        Δ3 = 0
    end
    if isnan(Δ4)
        Δ4 = 0
    end
    
    Δ1 = Inf
    # Δ2 = 0
    # Δ3 = 0
    # Δ4 = 0

    @assert Δ2 <= Δ1
    @assert Δ4 <= Δ3
    return Δ1, Δ2, Δ3, Δ4
end

# this is all I need now for transition bounds... for now
function simple_transition_bounds(image, state, dist; p_buffer=nothing, distance_buffer=nothing)
    if MULTIPLICATIVE_NOISE_FLAG
        distances_function = multiplicative_noise_distances
    else
        distances_function = additive_noise_distances
    end

    ndims = size(image,1)

    dis_comps = isnothing(distance_buffer) ? zeros(ndims, 4) : distance_buffer
    if USE_STATIC_PARTITIONS
        dis_comps[:,1] .= STATIC_PARTITION_BOUNDS[1]
        dis_comps[:,2] .= STATIC_PARTITION_BOUNDS[2]
    else
        [dis_comps[i,:] .= distances_function([image[i,1], image[i,2]], [state[i,1], state[i,2]]) for i=1:ndims]
    end

    p_low = prod(cdf(dist, dis_comps[i,3]) - cdf(dist, dis_comps[i,4]) for i=1:ndims)
    if p_low < 0.0 && p_low > -1e-10
        p_low = 0.0
    end

    p_high = prod(cdf(dist, dis_comps[i,1]) - cdf(dist, dis_comps[i,2]) for i=1:ndims)
    if p_high < 0.0 && p_high > -1e-10
        p_high = 0.0
    end

    @assert p_low <= p_high
    @assert p_low >= 0 && p_low <= 1
    @assert p_high >= 0

    if p_high > 1.0
        if p_high <= 1+1e-10
            p_high = 1.0
        else
            @assert p_high <= 1+1e-10
        end
    end

    p_vector = isnothing(p_buffer) ? zeros(2) : p_buffer
    p_vector[1] = p_low
    p_vector[2] = p_high

    return p_vector
end

function initialize_transition_matrices(nstates)
    Plow = spzeros(nstates, nstates)
    Phigh = spzeros(nstates, nstates)
    return Plow, Phigh
end

function state_radius(state)
    return 0.5*sqrt(sum((state[:,2] - state[:,1]).^2))
end

function calculate_transition_probabilities(explicit_states, all_images, compact_state, noise_distribution)
    nstates = length(explicit_states)+1
    Plow, Phigh = initialize_transition_matrices(nstates)

    n_transitions = size(Plow,1)^2
    progress_meter = Progress(n_transitions, "Computing state images...", dt=STATUS_BAR_PERIOD)
    p_vector = zeros(2)
    distance_buffer = zeros(size(explicit_states[1],1), 4)

    for (i, image) in enumerate(all_images)
        for (j, state) in enumerate(explicit_states)
            p_vector = simple_transition_bounds(image, state, noise_distribution, p_buffer=p_vector, distance_buffer=distance_buffer)
            Plow[i,j] = p_vector[1]
            Phigh[i,j] = p_vector[2]
            next!(progress_meter)
        end
    
        # to the bad state
        p_vector = simple_transition_bounds(image, compact_state, noise_distribution, p_buffer=p_vector, distance_buffer=distance_buffer)
        Plow[i,end] = 1 - p_vector[2]
        Phigh[i,end] = 1 - p_vector[1]
        next!(progress_meter)
    end

    Plow[end,end] = 1.0
    Phigh[end, end] = 1.0
    next!(progress_meter)

    for row in eachrow(Plow)
        @assert sum(row) <= 1.0
    end
    for row in eachrow(Phigh)
        @assert sum(Phigh) >= 1.0
    end
    return Plow, Phigh
end

# Results

function classify_results(result_matrix, threshold)
    classification = zeros(size(result_matrix,1))
    for i=1:size(result_matrix,1)
        if result_matrix[i,3] >= threshold
            classification[i] = 1
        elseif result_matrix[i,4] < threshold
            classification[i] = 2
        end
    end
    return classification
end