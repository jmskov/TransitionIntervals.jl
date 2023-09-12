# Functions that help refinement

# refine abstraction
function refine_abstraction(result_matrix, threshold, states, images, Plow, Phigh, full_state, noise_distribution, image_map)
    # here, perform  the cool refinement stuff
    states_to_refine, _ = find_states_to_refine(result_matrix, threshold, Phigh) 
    new_state_index_dict = refine_states!(states, states_to_refine)
    refine_images!(states, images, states_to_refine, image_map)
    new_Plow, new_Phigh = refine_transitions(states, new_state_index_dict, images, states_to_refine, Plow, Phigh, full_state, noise_distribution)
    return new_Plow, new_Phigh
    # return new_Plow, new_Phigh, new_state_images, new_state_index_dict
end

function find_states_to_refine(result_matrix, threshold)
    classifications = classify_results(result_matrix, threshold)
    states_to_refine = findall(x->x==0, classifications)
    return states_to_refine
end

function find_states_to_refine(result_matrix, threshold, Phigh)
    classifications = classify_results(result_matrix, threshold)
    candidates_to_refine = findall(x->x==0, classifications)

    positive_states = findall(x->x==1, classifications)
    negative_states = findall(x->x==2, classifications)

    states_to_refine = []
    for state in positive_states
        for i in candidates_to_refine
            if Phigh[i,state] > 0.0
                push!(states_to_refine, i)
            end
        end
    end

    for state in negative_states
        for i in candidates_to_refine
            if Phigh[i,state] > 0.0
                push!(states_to_refine, i)
            end
        end
    end

    states_to_keep = setdiff(1:size(Phigh,1)-1, states_to_refine)

    return unique(sort(states_to_refine)), states_to_keep
end

function uniform_refinement(state)
    deltas = (state[:,2] - state[:,1])/2
    refined_grid, refined_deltas = grid_generator(state[:,1], state[:,2], deltas)
    new_states = calculate_explicit_states(refined_grid, refined_deltas) 
    return new_states
end

function refine_states!(explicit_states, states_to_refine; min_size=MIN_REFINE_SIZE)
    new_states = []
    state_map_dict = Dict()
    new_state_idx = length(explicit_states) - length(states_to_refine) + 1
    states_skipped = []
    for state_idx in states_to_refine
        if !all(explicit_states[state_idx][:,2] - explicit_states[state_idx][:,1] .> min_size)
            push!(states_skipped, state_idx)
            continue
        end
        refined_states = uniform_refinement(explicit_states[state_idx])
        push!(new_states, refined_states...)
        state_map_dict[state_idx] = new_state_idx:1:(new_state_idx+length(refined_states)-1)
        new_state_idx += length(refined_states)
    end

    if length(states_skipped) > 0
        setdiff!(states_to_refine, states_skipped)
        @warn "Skipped $(length(states_skipped)) states because they were too small."
    end

    deleteat!(explicit_states, states_to_refine)
    explicit_states = push!(explicit_states, new_states...)
    return state_map_dict 
end

function refine_images!(explicit_states, state_images, states_to_refine, user_defined_map)
    dim_factor = 2^size(state_images[1],1)
    original_state_num = length(explicit_states) - dim_factor*length(states_to_refine)   # this should be the length of the original images...
    @assert length(state_images) == original_state_num + length(states_to_refine)

    new_state_idx = original_state_num + 1
    new_state_images = calculate_all_images(explicit_states[new_state_idx:1:length(explicit_states)], user_defined_map)
    deleteat!(state_images, states_to_refine)
    push!(state_images, new_state_images...)
    @assert length(state_images) == length(explicit_states)

    return state_images
end

function get_refined_index(index, states_to_refine)
    # find the number of states in states_to_refine that is less than index
    @assert index ∉ states_to_refine
    num_less = length(findall(x->x<index, states_to_refine))
    return index -= num_less
end

function refine_transitions(explicit_states, state_index_dict, state_images, states_to_refine, Plow, Phigh, compact_state, noise_distribution)
    n_states_new = length(explicit_states) + 1 # add one for the sink state
    n_states_old = size(Plow,2)
    unrefined_states = setdiff(1:n_states_old-1, states_to_refine)
    num_unrefined_states = length(unrefined_states)
    
    # Plow_new = spzeros(n_states_new, n_states_new)
    # Phigh_new = spzeros(n_states_new, n_states_new)

    # Get the successor states of the unrefined images 
    unrefined_states_to_recomp = Dict()
    num_old_state_transitions = 0
    for unrefined_state_index in unrefined_states 
        target_idxs = []
        # get the successor states of the unrefined state
        succ_states = findall(x->x>0, Phigh[unrefined_state_index, :]) # old indeces

        # If the successor state is in states_to_refine, then we need to recompute the transition
        for succ_state in succ_states
            if succ_state ∈ states_to_refine
                # recall that state_index_dict maps the original state index to the set of new state indeces
                push!(target_idxs, state_index_dict[succ_state])
            end
        end

        if !isempty(target_idxs)
            unrefined_states_to_recomp[unrefined_state_index] = unique(target_idxs)
            num_old_state_transitions += length(unrefined_states_to_recomp[unrefined_state_index])
        end
    end

    # get the possible successor states of the refined images (avoid doing all the comp lol)
    refined_states_to_recomp = Dict()
    num_new_state_transitions = 0
    for old_idx in states_to_refine
        target_idxs = []
        # get the successor states 
        succ_states = findall(x->x>0, Phigh[old_idx, :]) # old indeces

        for succ_state in succ_states

            # if the successor state is a refined state
            if succ_state ∈ states_to_refine
                push!(target_idxs, state_index_dict[succ_state]...)
            # if the successor state is an unrefined state
            else
                push!(target_idxs, get_refined_index(succ_state, states_to_refine))
            end
        end

        # get the new indeces 
        new_indeces = state_index_dict[old_idx]
        for idx in new_indeces
            @assert idx ∉ keys(refined_states_to_recomp)
            refined_states_to_recomp[idx] = unique(target_idxs)
            num_new_state_transitions += length(refined_states_to_recomp[idx]) + 1 # one for the sink state!
        end
    end

    # reverse index dict maps the new indeces to the old (unrefined) index
    reverse_index_dict = Dict()
    for (key, value) in state_index_dict
        reverse_index_dict[value] = key
    end

    # Recompute the transitions between all new refined states with the old unrefined states

    nthreads = Threads.nthreads()
    P_low_buffers = [spzeros(n_states_new, n_states_new) for i=1:nthreads]
    P_high_buffers = [spzeros(n_states_new, n_states_new) for i=1:nthreads]

    p_vectors = [zeros(2) for i=1:nthreads]
    distance_buffers = [zeros(size(explicit_states[1],1), 4) for i=1:nthreads]

    progress_meter = Progress(num_new_state_transitions, "Calculating refined transitions from refined states...", dt=STATUS_BAR_PERIOD)
    Threads.@threads for i in num_unrefined_states+1:n_states_new-1
    # for i in num_unrefined_states+1:n_states_new-1 
        # states to recompute
        target_idxs = refined_states_to_recomp[i]
        @views process_row!(P_low_buffers[Threads.threadid()][i,:], P_high_buffers[Threads.threadid()][i,:], explicit_states, state_images[i], compact_state, target_idxs, noise_distribution, p_vectors[Threads.threadid()], distance_buffers[Threads.threadid()])
    end 

    # Now, recompute the transitions between old unrefined states and new refined states
    progress_meter = Progress(length(keys(unrefined_states_to_recomp)), "Calculating refined transitions from unrefined states...", dt=STATUS_BAR_PERIOD)

    warn_count = 0

    dict_keys = collect(keys(unrefined_states_to_recomp))
    Threads.@threads for idx=1:length(dict_keys) 
    # for idx=1:length(dict_keys) 
        # next!(progress_meter)
        unrefined_state_index = dict_keys[idx]
        target_set = unrefined_states_to_recomp[unrefined_state_index]
        new_index = get_refined_index(unrefined_state_index, states_to_refine)

        for b in eachindex(target_set) #target_indeces in target_set
            target_indeces = target_set[b]
            original_target_index = reverse_index_dict[target_indeces]

            if all(state_images[unrefined_state_index] .== 0)
                # cannot reuse the lower-bound when we are splitting states
                if warn_count < 5
                    @warn "Zero-valued image!"
                    warn_count += 1
                end

                @views P_low_buffers[Threads.threadid()][new_index, target_indeces] .= 0  
                @views P_high_buffers[Threads.threadid()][new_index, target_indeces] .= Phigh[unrefined_state_index, original_target_index]
            else
                for target_idx in target_indeces
                    # ! todo: handle the zero image case...
                    simple_transition_bounds(state_images[new_index], explicit_states[target_idx], noise_distribution, p_buffer=p_vectors[Threads.threadid()], distance_buffer=distance_buffers[Threads.threadid()])
                        @views P_low_buffers[Threads.threadid()][new_index, target_idx] = p_vectors[Threads.threadid()][1]
                        @views P_high_buffers[Threads.threadid()][new_index, target_idx] = p_vectors[Threads.threadid()][2]
                end
            end
        end
    end

    Plow_new = sum(P_low_buffers)
    Phigh_new = sum(P_high_buffers)

    # Resue the old transitions between unrefined states
    # Plow_new[1:num_unrefined_states, 1:num_unrefined_states] = Plow[unrefined_states, unrefined_states]
    # Phigh_new[1:num_unrefined_states, 1:num_unrefined_states] = Phigh[unrefined_states, unrefined_states]
    i = 1
    for unrefined_state_index in unrefined_states
        j = 1
        for target_unrefined_idx in unrefined_states 
            Plow_new[i, j] = Plow[unrefined_state_index, target_unrefined_idx]
            Phigh_new[i, j] = Phigh[unrefined_state_index, target_unrefined_idx] 
            j += 1
        end
        Plow_new[i, end] = Plow[unrefined_state_index, end]
        Phigh_new[i, end] = Phigh[unrefined_state_index, end]
        i += 1
    end
    Plow_new[end,end] = 1.0 
    Phigh_new[end,end] = 1.0

    # Verify the transition matrices
    for row in eachrow(Plow_new)
        @assert sum(row) <= 1.0
    end
    for row in eachrow(Phigh_new)
        @assert sum(row) >= 1.0
    end

    return Plow_new, Phigh_new
end

function process_row!(plow_row, phigh_row, states, image, compact_state, targets, noise_distribution, p_vector, distance_buffer)
    for b in eachindex(targets)
        j = targets[b]
        # next!(progress_meter)
        simple_transition_bounds(image, states[j], noise_distribution, p_buffer=p_vector, distance_buffer=distance_buffer)
        plow_row[j] = p_vector[1]
        phigh_row[j] = p_vector[2]
    end
    simple_transition_bounds(image, compact_state, noise_distribution, p_buffer=p_vector, distance_buffer=distance_buffer)
    plow_row[end] = 1 - p_vector[2]
    phigh_row[end] = 1 - p_vector[1] 
end