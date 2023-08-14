# Functions for abstraction

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
    # initialize an array of 2x2 matrices
    states = Array{Array{Float64,2},1}(undef, length(grid))
    for (i, grid_point) in enumerate(grid)
        states[i] = [grid_point[1] grid_point[1] + grid_spacing[1]; grid_point[2] grid_point[2] + grid_spacing[2]]
    end
    return states
end

function calculate_all_images(explicit_states, user_defined_map)
    images = Array{Array{Float64,2},1}(undef, length(explicit_states))
    for (i, state) in enumerate(explicit_states)
        images[i] = user_defined_map(state) 
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
function find_distances(image, target)
    width = image[2] - image[1]
    Δ1 = image[1] - target[2]
    Δ2 = image[2] - target[1]
    Δ3 = image[2] - target[2]
    Δ4 = image[2] - (target[1] + width)
    return -Δ1, -Δ2, -Δ3, -Δ4
end

# this is all I need now for transition bounds... for now
function simple_transition_bounds(image, state, dist)
    ndims = size(image,1)
    dis_comps = zeros(ndims, 4)
    [dis_comps[i,:] .= find_distances([image[i,1], image[i,2]], [state[i,1], state[i,2]]) for i=1:ndims]
    p_low = (cdf(dist, dis_comps[1,3]) - cdf(dist, dis_comps[1,4]))*(cdf(dist, dis_comps[2,3]) - cdf(dist, dis_comps[2,4]))
    p_high = (cdf(dist, dis_comps[1,1]) - cdf(dist, dis_comps[1,2]))*(cdf(dist, dis_comps[2,1]) - cdf(dist, dis_comps[2,2]))
    return p_low, p_high
end

function initialize_transition_matrices(nstates)
    Plow = spzeros(nstates, nstates)
    Phigh = spzeros(nstates, nstates)
    return Plow, Phigh
end

function calculate_transition_probabilities(explicit_states, all_images, compact_state, noise_distribution)
    nstates = length(explicit_states)+1
    Plow, Phigh = initialize_transition_matrices(nstates)

    for (i, image) in enumerate(all_images)
        for (j, state) in enumerate(explicit_states)
            p_low, p_high = simple_transition_bounds(image, state, noise_distribution)
            Plow[i,j] = p_low
            Phigh[i,j] = p_high
        end
    
        # to the bad state
        p_low, p_high = simple_transition_bounds(image, compact_state, noise_distribution)
        Plow[i,end] = 1 - p_high
        Phigh[i,end] = 1 - p_low
    end

    Plow[end,end] = 1.0
    Phigh[end, end] = 1.0
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