# Tool related to clustering and implicit merging

"""
    check_intersection
"""
function check_intersection(state1, state2)
    # check if the two states intersect
    dims = size(state1, 1)
    for dim = 1:dims
        if state1[dim, 1] >= state2[dim, 2] || state1[dim, 2] <= state2[dim, 1]
            return false
        end
    end
    return true
end

"""
    find_intersecting_states
"""
function find_intersecting_states(image, states)
    intersecting_states = []
    for (idx, state) in enumerate(states)
        if check_intersection(image, state) 
            push!(intersecting_states, idx)
        end
    end
    return intersecting_states
end

"""
    build_super_state
"""
function build_super_state(states)
    # find min extents
    dims = size(states[1], 1)
    super = zeros(dims, 2)
    for dim = 1:dims
        minv = Inf
        maxv = -Inf
        for state in states
            row = state[dim, :];
            minv = min(minimum(row), minv)
            maxv = max(maximum(row), maxv)
        end
        super[dim, :] = [minv, maxv]
        # push!(exs, [minv, maxv])
    end
    return super 
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

"""
    calculate_implicit_plow

Calculate the new L.B. probability of satisfaction for a certain state with implicit clustering of successor states 
"""
function calculate_implicit_p(P̌_row, P̂_row, image, states, prior_results::AbstractMatrix, noise_distribution, state_idx; log_flag=false)

    # Create Q̃ and compute the transition interval from q
    # numerical? 
    intersecting_state_idxs = find_intersecting_states(image, states) 
    cluster_state = build_super_state(states[intersecting_state_idxs])
    p_low_new, p_high_new = simple_transition_bounds(image, cluster_state, noise_distribution)

    # Find all the states in Q^* 
    all_succ_states_star = setdiff(findall(x->x>0, P̂_row), intersecting_state_idxs)
    @assert all_succ_states_star ∩ intersecting_state_idxs == []
    all_P̌ = Array(P̌_row[all_succ_states_star])
    all_P̂ = Array(P̂_row[all_succ_states_star])
    all_succ_res = prior_results[all_succ_states_star, 3]
    all_succ_res_upper = prior_results[all_succ_states_star, 4]
    push!(all_P̌, p_low_new)

    # trim from P̂, then add the new one! 
    for i in eachindex(all_P̂)
        if all_P̂[i] > 1 - p_low_new 
            all_P̂[i] = 1 - p_low_new
        end
    end
    push!(all_P̂, p_high_new) 

    # Calculate the new results
    p̌_cluster = minimum(prior_results[intersecting_state_idxs, 3])
    all_succ_res = [all_succ_res;  p̌_cluster]
    idx_perm = sortperm(all_succ_res)

    p_true = true_transition_probabilities(all_P̌, all_P̂, idx_perm)
    
    p̌_new = round(sum(p_true .* all_succ_res), digits=6)

    p̂_cluster = maximum(prior_results[intersecting_state_idxs, 4])
    all_succ_res_upper = [all_succ_res_upper; p̂_cluster]
    idx_perm_upper = sortperm(all_succ_res_upper, rev=true)
    p_true_upper = true_transition_probabilities(all_P̌, all_P̂, idx_perm_upper)
    p̂_new = sum(p_true_upper .* all_succ_res_upper)

    # check for numerical accuracy
    if p̂_new > 1 && p̂_new - 1 < 1e-8
        p̂_new = 1
    end

    @assert p̌_new ≤ 1.0
    @assert sum(all_P̌) ≤ 1.0
    @assert sum(all_P̂) ≥ 1.0
    @assert p̂_new ≤ 1.0
    @assert p̂_new ≥ p̌_new 

    for i in eachindex(p_true)
        @assert all_P̌[i] ≤ p_true[i] ≤ all_P̂[i]
    end

    if log_flag
        @info "p̌_cluster: ", p̌_cluster
        @info "all_succ_res: ", sort(all_succ_res)
        @info "p_true: ", sort(p_true)
        @info "p̌_new: ", p̌_new
        @info "plow", sort(all_P̌) 
        @info "phigh", sort(all_P̂)
    end
    return p̌_new, p̂_new, intersecting_state_idxs
end

function get_filtered_results(result_mat; λ=0.90)
    ver_new = copy(result_mat)
    ver_new = ver_new[sortperm(ver_new[:,3], rev=true), :] # verification results sorted from highest to lowest LB
    filter_idx = findall(x -> x < λ, ver_new[:,4]) ∪ findall(x -> x==1, ver_new[:,3])
    keep_idx = setdiff(1:size(ver_new,1), filter_idx)
    ver_new = ver_new[keep_idx,:]
    return ver_new
end

function cluster_all_states(verification_result_mat, images, states; numdfa=1)
    ver_new = get_filtered_results(verification_result_mat)
    states_to_cluster = []
    Qyes = Int.(findall(x->x>0.9, verification_result_mat[:,3])) # TODO: Generalize this

    for row in eachrow(ver_new)
        idx = Int(ceil(row[1]/numdfa))

        # Get succ_states 
        succ_states = Stochascape.find_intersecting_states(images[idx], states)
        if isempty(succ_states) || idx ∈ succ_states # when the image is outside the set
            continue
        end

        if length(succ_states) > 1
            push!(states_to_cluster, idx)
        end
        # push!(states_to_cluster, idx)
        # for succ_state_idx in succ_states
        #     if succ_state_idx ∈ Qyes #&& length(succ_states) > 1
        #         push!(states_to_cluster, idx)
        #         break
        #     end
        # end
    end
    return states_to_cluster 
end

function modify_P!(Plow, Phigh, mods, accepting_state, succ_states_dict)
    m_keys = sort([keys(mods)...])
    modif = 0
    for k in m_keys 
        v = mods[k]
        plow_new = v[1]
        phigh_new = v[2]
        if plow_new < 0.9 #&& phigh_new > 0.1
            continue
        end
        modif += 1
        # @info plow_new
        # @info phigh_new
        Plow[k,:] .= 0 # this works, as all other states necessarily have a LB of zero;
        Phigh[k,:] .= 0 # this works, as all other states necessarily have a LB of zero;
        # @info "previous plow"
        # @info Plow[k, succ_states_dict[k]] 
        # @info plow_new
        # Plow[k, succ_states_dict[k]] .= 0
        # Phigh[k, succ_states_dict[k]] .= 0
        Plow[k, accepting_state] = plow_new
        Phigh[k, accepting_state] = phigh_new             # Set the UB to the accepting_state as trivial
        Plow[k, end] = 1. - phigh_new
        Phigh[k, end] = 1. - plow_new

        # states that are not succ states:
        succ_states_all = findall(x -> x > 0., Phigh[k, :])
        remainder_idxs = setdiff(succ_states_all, succ_states_dict[k])

        # if sum(Plow[k, :]) >= 1
        #     diff = sum(Plow[k, :]) - 1
        #     # remove diff from remainder_idxs
        #     sort_idxs = sortperm(Plow[k, remainder_idxs], rev=true)
        #     for idx in sort_idxs
        #         if diff > 0
        #             if Plow[k, remainder_idxs[idx]] > diff
        #                 Plow[k, remainder_idxs[idx]] -= diff + 1e-9
        #                 diff = 0
        #             else
        #                 diff -= Plow[k, remainder_idxs[idx]]
        #                 Plow[k, remainder_idxs[idx]] = 0
        #             end
        #         end
        #     end

        # end
        # @info sum(Plow[k, :])
        # @assert sum(Plow[k, :]) ≤ 1.0

    #     for remainder_idx in remainder_idxs
    #         # @info remainder_idx, k
    #         # @info accepting_state
    #         # @info remainder_idx ∈ succ_states_dict[k]
    #         # @info succ_states_dict[k]
    #         if Phigh[k, remainder_idx] > 1. - plow_new  #&& remainder_idx != accepting_state  # if the upper bound is closer to 1.0 than v, i.e. of v=20% and Phigh = 90%, then 1-Phigh = 10% and it needs to go to 1-v = 80%; 
    #             Phigh[k, remainder_idx] = max(1. - plow_new, Plow[k, remainder_idx])
    #             @info Phigh[k, remainder_idx], Plow[k, remainder_idx]   
    #             @assert Phigh[k, remainder_idx] >= Plow[k, remainder_idx]  
    #         end
    #     end
    # end

    # for i in eachindex(Plow[accepting_state, :])
    #     for j in eachindex(Plow[accepting_state, :])
    #         @info "i: ", i, " j: ", j
    #         @info "Plow[i,j]: ", Plow[i,j]
    #         @info "Phigh[i,j]: ", Phigh[i,j]
    #         @assert Plow[i,j] ≤ Phigh[i,j]
    #     end
    end

    @info "modif: ", modif

    for row in eachrow(Plow)
        @assert sum(row) ≤ 1.0
    end
    for row in eachrow(Phigh)
        @assert sum(row) >= 1.0
    end
end

function cluster_step!(result_matrix, states, images, Plow, Phigh, noise_distribution)
    states_to_cluster = cluster_all_states(result_matrix, images, states)

    if 143 ∈ states_to_cluster
        @warn("143 is in states_to_cluster")
    end

    updated_bounds = Dict()
    succ_states_dict = Dict()

    Plow_copy = copy(Plow)
    Phigh_copy = copy(Phigh)

    num_improvements = 1
    while num_improvements > 0
        num_improvements = 0
        for state_idx in states_to_cluster
            log_flag = false
            if state_idx == 155 || state_idx == 54
                log_flag = false 
                @info "state_idx: ", state_idx
            end
            plow_new, phigh_new, succ_states = Stochascape.calculate_implicit_p(Plow[state_idx,:], Phigh[state_idx,:], images[state_idx], states, result_matrix, noise_distribution, state_idx, log_flag=log_flag)

            succ_states_dict[state_idx] = succ_states 

            # > not the error
            if plow_new > result_matrix[state_idx, 3] || phigh_new < result_matrix[state_idx, 4]
                updated_bounds[state_idx] = (plow_new, phigh_new)
                num_improvements += 1
                result_matrix[state_idx, 3] = plow_new
                result_matrix[state_idx, 4] = phigh_new
            end
        end
        @info "num_improvements: ", num_improvements
    end
    accepting_idx = findfirst(x -> x == 1, result_matrix[:,3])
    modify_P!(Plow, Phigh, updated_bounds, accepting_idx, succ_states_dict)
    return Plow_copy, Phigh_copy
end