# Functions that help refinement

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

    return unique(sort!(states_to_refine)), states_to_keep
end

function uniform_refinement(state)
    deltas = (state[:,2] - state[:,1])/2
    refined_grid, refined_deltas = grid_generator(state[:,1], state[:,2], deltas)
    new_states = calculate_explicit_states(refined_grid, refined_deltas) 
    return new_states
end

function refine_states!(explicit_states, states_to_refine)
    new_states = []
    state_map_dict = Dict()
    new_state_idx = length(explicit_states) - length(states_to_refine) + 1
    for state_idx in states_to_refine
        refined_states = uniform_refinement(explicit_states[state_idx])
        push!(new_states, refined_states...)
        state_map_dict[state_idx] = new_state_idx:1:(new_state_idx+length(refined_states)-1)
        new_state_idx += length(refined_states)
    end
    deleteat!(explicit_states, states_to_refine)
    explicit_states = push!(explicit_states, new_states...)
    return state_map_dict 
end

function refine_images!(explicit_states, state_images, states_to_refine, user_defined_map)
    new_state_idx = length(state_images) - length(states_to_refine) + 1
    deleteat!(state_images, states_to_refine)
    for state_idx in new_state_idx:1:length(explicit_states) 
        push!(state_images, user_defined_map(explicit_states[state_idx]))
    end
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
    
    Plow_new = spzeros(n_states_new, n_states_new)
    Phigh_new = spzeros(n_states_new, n_states_new)

    # Get the successor states of the unrefined images 
    unrefined_states_to_recomp = Dict()
    for unrefined_state_index in unrefined_states 
        target_idxs = []
        succ_states = findall(x->x>0, Phigh[unrefined_state_index, :]) # old indeces

        # If the successor state is in states_to_refine, then we need to recompute the transition
        for succ_state in succ_states
            if succ_state ∈ states_to_refine
                push!(target_idxs, state_index_dict[succ_state]...)
            end
        end

        if !isempty(target_idxs)
            # Get the transformed unrefined state index
            unrefined_states_to_recomp[get_refined_index(unrefined_state_index, states_to_refine)] = sort(unique(target_idxs))
        end
    end

    # Resue the old transitions between unrefined states
    Plow_new[1:num_unrefined_states, 1:num_unrefined_states] = Plow[unrefined_states, unrefined_states]
    Phigh_new[1:num_unrefined_states, 1:num_unrefined_states] = Phigh[unrefined_states, unrefined_states]
    for (i, unrefined_state_index) in enumerate(unrefined_states)
        Plow_new[i, end] = Plow[unrefined_state_index, end]
        Phigh_new[i, end] = Phigh[unrefined_state_index, end]
    end
    Plow_new[end,end] = 1.0 
    Phigh_new[end,end] = 1.0

    # Now, recompute the transitions between old unrefined states and new refined states
    for (unrefined_state_index, target_idxs) in unrefined_states_to_recomp
        for target_idx in target_idxs
            @assert target_idx > num_unrefined_states 
            p_low, p_high = simple_transition_bounds(state_images[unrefined_state_index], explicit_states[target_idx], noise_distribution)
            Plow_new[unrefined_state_index, target_idx] = p_low
            Phigh_new[unrefined_state_index, target_idx] = p_high
        end
    end

    # Now, recompute the transitions between all new refined states with the old unrefined states
    for i = num_unrefined_states+1:n_states_new-1
        for j = 1:n_states_new-1
            p_low, p_high = simple_transition_bounds(state_images[i], explicit_states[j], noise_distribution)
            Plow_new[i,j] = p_low
            Phigh_new[i,j] = p_high
        end
        p_low, p_high = simple_transition_bounds(state_images[i], compact_state, noise_distribution)
        Plow_new[i,end] = 1 - p_high
        Phigh_new[i,end] = 1 - p_low
    end 

    # Verify the transition matrices
    for row in eachrow(Plow_new)
        @assert sum(row) <= 1.0
    end
    for (i,row) in enumerate(eachrow(Phigh_new))
        @assert sum(row) >= 1.0
    end

    return Plow_new, Phigh_new
end
