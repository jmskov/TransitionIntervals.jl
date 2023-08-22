# Tools to help merge states

function shares_face(ex1, ex2)
    # in 2D, either the x row is the same or the y row is the same. 

    if (!all(ex1[1,:] == ex2[1,:]) && !all(ex1[2,:] == ex2[2,:]))
        return false
    end
    if (ex1[1,2] == ex2[1,1] || ex2[1,2] == ex1[1,1]) ||
        (ex1[2,2] == ex2[2,1] || ex2[2,2] == ex1[2,1])
        return true
    end
    return false
end

function consistent_merge(merge_list, new_ex)
    for merge_ex in merge_list
        if (!all(merge_ex[1,:] == new_ex[1,:]) && !all(merge_ex[2,:] == new_ex[2,:]))
            return false
        end
    end
    return true
end

function check_for_merges(state_extents, state_idxs_to_merge)
    merged_idxs = []
    local_merged_idxs = []
    list_idx = 1
    
    # temporary measure only works for states with subsequent indeces
    while list_idx < length(state_idxs_to_merge)
        new_merge = []
        new_local_merge = []
        while (true)
            # this does not take into account the original shared face...
            state_list_idx = state_idxs_to_merge[list_idx]
            next_list_idx = state_idxs_to_merge[list_idx+1]
            if list_idx < length(state_idxs_to_merge) && shares_face(state_extents[state_list_idx], state_extents[next_list_idx])
                state_list_idx = state_idxs_to_merge[list_idx]  # need redunancy for inner loop
                next_list_idx = state_idxs_to_merge[list_idx+1]
                if isempty(new_merge)
                    new_merge = [state_list_idx, next_list_idx]
                    new_local_merge = [list_idx, list_idx+1]
                else
                    # here, check for consistency
                    if consistent_merge(state_extents[new_merge], state_extents[next_list_idx])
                        push!(new_merge, next_list_idx)
                        push!(new_local_merge, list_idx+1)
                    else
                        if !isempty(new_merge)
                            push!(merged_idxs, new_merge)
                            push!(local_merged_idxs, new_local_merge)
                        end
                        break
                    end
                end
                list_idx += 1        

                if list_idx == length(state_idxs_to_merge)
                    # new_merge = []
                    break
                end
            else 
                if !isempty(new_merge)
                    push!(merged_idxs, new_merge)
                    push!(local_merged_idxs, new_local_merge)
                end
                break
            end
        end
        list_idx += 1
    end

    return merged_idxs, local_merged_idxs
end

function merge_extents(extents)
    ndims = size(extents[1], 1)
    ex = ones(ndims,2)
    ex[:,1] *= Inf
    ex[:,2] *= -Inf

    for extent in extents
        for dim = 1:ndims
            if extent[dim,1] < ex[dim,1]
                ex[dim,1] = extent[dim,1] 
            end
            if extent[dim,2] > ex[dim,2] 
                ex[dim,2] = extent[dim,2] 
            end
        end
    end

    return ex
end

function iterative_merge(extents, images, noise_distribution, classifications, classification_idx, Plow, Phigh, result_matrix; num_steps=10)

    step = 0

    all_states_to_merge_idxs = findall(classifications .== classification_idx)
    sort!(all_states_to_merge_idxs)

    Plow_new, Phigh_new = Plow, Phigh
    new_result_matrix = result_matrix
    while step < num_steps 
        global_merge_idxs, local_merge_idxs = check_for_merges(extents, all_states_to_merge_idxs) 
        sort!(global_merge_idxs)
        sort!(local_merge_idxs) # still need these?

        if isempty(global_merge_idxs)
            break
        end

        new_state_start_index = length(extents) + 1
        for idx_set in global_merge_idxs
            new_state_start_index -= length(idx_set)
        end

        new_merged_extents = []
        all_merged_indeces = []
        merged_idx_dict = Dict()
        i = new_state_start_index
        for idx_set in global_merge_idxs
            # new_extent =  merge_extents(all_states_to_merge[idx_set])
            new_extent =  merge_extents(extents[idx_set])
            push!(new_merged_extents, new_extent)
            # here, I want to translate new idxs to old idxs based on how all_extents has changed...
            all_merged_indeces = [all_merged_indeces..., idx_set...]
            merged_idx_dict[i] = idx_set
            i += 1
        end
        # unique!(all_merged_indeces)
        # sort!(all_merged_indeces)

        deleteat!(extents, all_merged_indeces)
        deleteat!(images, all_merged_indeces) # never need these old images again
        push!(extents, new_merged_extents...)
        push!(images, fill(zeros(2,2), length(new_merged_extents))...) 

        step += 1

        # remove merged state idxs, add new ones
        all_local_idxs = []
        for idx_set in local_merge_idxs
            all_local_idxs = [all_local_idxs..., idx_set...]
            sort!(all_local_idxs)
        end

        deleteat!(all_states_to_merge_idxs, all_local_idxs)
        # here, need to translate the local merge idxs to the new idxs
        for i = 1:length(all_states_to_merge_idxs)
            all_states_to_merge_idxs[i] =  get_refined_index(all_states_to_merge_idxs[i], all_merged_indeces)
        end
        push!(all_states_to_merge_idxs, collect(length(extents)-length(new_merged_extents)+1:length(extents))...)

        n_states_original = size(Plow_new, 2)
        n_states_new = length(extents) + 1 

        remaining_indeces = setdiff(1:n_states_original-1, all_merged_indeces) # skip the sink state

        new_state_start_index = length(remaining_indeces) + 1

        Plow_new_merge = zeros(n_states_new, n_states_new)
        Phigh_new_merge = zeros(n_states_new, n_states_new)
        new_result_matrix_merge = zeros(n_states_new, 4)

        #==
            From Old States  
        ==#
        for unrefined_state_idx in remaining_indeces
            refined_idx = get_refined_index(unrefined_state_idx, all_merged_indeces)

            # To Old States
            for target_idx in remaining_indeces
                refined_target_idx = get_refined_index(target_idx, all_merged_indeces)
                Plow_new_merge[refined_idx, refined_target_idx] = Plow_new[unrefined_state_idx, target_idx]
                Phigh_new_merge[refined_idx, refined_target_idx] = Phigh_new[unrefined_state_idx, target_idx]
            end
            # To New States
            for target_index in new_state_start_index:n_states_new-1
                idx_set = merged_idx_dict[target_index] 
                Plow_new_merge[refined_idx, target_index] = maximum(Plow_new[unrefined_state_idx, idx_set]) # increase here, interesting...
                Phigh_new_merge[refined_idx, target_index] = maximum(Phigh_new[unrefined_state_idx, idx_set])
            end
            # To Sink State
            Plow_new_merge[refined_idx, end] = Plow_new[unrefined_state_idx, end]
            Phigh_new_merge[refined_idx, end] = Phigh_new[unrefined_state_idx, end]

            # Update Result Matrix
            new_result_matrix_merge[refined_idx, 1] = refined_idx
            new_result_matrix_merge[refined_idx, 2:end] = new_result_matrix[unrefined_state_idx, 2:end]
        end     

        # Sink accepting
        Plow_new_merge[end,end] = 1.0 
        Phigh_new_merge[end,end] = 1.0

        #==
            From New States 
        ==#
        for state_idx in new_state_start_index:n_states_new-1
            state_idx_set = merged_idx_dict[state_idx]

            # To Old States
            for target_idx in remaining_indeces
                refined_target_idx = get_refined_index(target_idx, all_merged_indeces)
                Plow_new_merge[state_idx, refined_target_idx] = minimum(Plow_new[state_idx_set, target_idx])
                Phigh_new_merge[state_idx, refined_target_idx] = maximum(Phigh_new[state_idx_set, target_idx])
            end

            # To New States
            for target_idx in new_state_start_index:n_states_new-1
                target_idx_set = merged_idx_dict[target_idx]
                plows = []
                for unrefined_state_idx in state_idx_set
                    push!(plows, maximum(Plow_new[unrefined_state_idx, target_idx_set]))
                end
                Plow_new_merge[state_idx, target_idx] = minimum(plows)
                Phigh_new_merge[state_idx, target_idx] = maximum(Phigh_new[state_idx_set, target_idx_set])
            end

            # To Unsafe States
            Plow_new_merge[state_idx, end] = minimum(Plow_new[state_idx_set, end])
            Phigh_new_merge[state_idx, end] = maximum(Phigh_new[state_idx_set, end])

            # Update Result Matrix
            new_result_matrix_merge[state_idx, 1] = state_idx
            new_result_matrix_merge[state_idx, 2:end] = new_result_matrix[state_idx_set[1], 2:end]
        end

        # Sink accepting
        new_result_matrix_merge[end, 1] = n_states_new
        new_result_matrix_merge[end, 2:end] = new_result_matrix[end, 2:end] 

        @assert length(images) == length(extents)

        Plow_new = Plow_new_merge
        Phigh_new = Phigh_new_merge
        new_result_matrix = new_result_matrix_merge

            # Verify the transition matrices
        for row in eachrow(Plow_new)
            @assert sum(row) <= 1.0
        end
        for (i,row) in enumerate(eachrow(Phigh_new))
            if sum(row) < 1.0
                @info "row $i"
            end
            @assert sum(row) >= 1.0
        end
    end

    return extents, images, Plow_new, Phigh_new, new_result_matrix
end
