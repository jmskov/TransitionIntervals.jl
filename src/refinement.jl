# Functions that help refinement
# > Methods that copy
function refine_abstraction(abstraction::Abstraction, image_map::Union{Function, Matrix{Float64}}, full_state::DiscreteState, refinement_idxs::Vector{Int})
    new_abstraction = deepcopy(abstraction)
    return refine_abstraction!(new_abstraction, image_map, full_state, refinement_idxs)
end

function refine_abstraction(abstraction::Abstraction, image_map::Union{Function, Matrix{Float64}}, full_state::DiscreteState, refinement_idxs::Vector{Int}, process_dist::Distribution)
    new_abstraction = deepcopy(abstraction)
    return refine_abstraction!(new_abstraction, image_map, full_state, refinement_idxs, process_dist)
end

function refine_abstraction(abstraction::Abstraction, image_map::Union{Function, Matrix{Float64}}, full_state::DiscreteState, refinement_idxs::Vector{Int}, uniform_error_dist::Union{Function, Distribution})
    new_abstraction = deepcopy(abstraction)
    return refine_abstraction!(new_abstraction, image_map, full_state, refinement_idxs, uniform_error_dist)
end

function refine_abstraction(abstraction::Abstraction, image_map::Union{Function, Matrix{Float64}}, full_state::DiscreteState, refinement_idxs::Vector{Int}, process_dist::Distribution, uniform_error_dist::Union{Function, Distribution})
    new_abstraction = deepcopy(abstraction)
    return refine_abstraction!(new_abstraction, image_map, full_state, refinement_idxs, process_dist, uniform_error_dist)
end

# > In-Place Methods
function refine_abstraction!(abstraction::Abstraction, image_map::Union{Function, Matrix{Float64}}, full_state::DiscreteState, refinement_idxs::Vector{Int})
    new_state_index_dict = refine_states!(abstraction.states, refinement_idxs)
    refine_images!(abstraction.states, abstraction.images, refinement_idxs, image_map)
    Plow_new, Phigh_new = refine_transitions(abstraction.states, abstraction.images, full_state, abstraction.Plow, abstraction.Phigh, refinement_idxs, new_state_index_dict, nothing, nothing)
    return Abstraction(abstraction.states, abstraction.images, Plow_new, Phigh_new) 
end

function refine_abstraction!(abstraction::Abstraction, image_map::Union{Function, Matrix{Float64}}, full_state::DiscreteState, refinement_idxs::Vector{Int}, process_dist::Distribution)
    new_state_index_dict = refine_states!(abstraction.states, refinement_idxs)
    refine_images!(abstraction.states, abstraction.images, refinement_idxs, image_map)

    Plow_new, Phigh_new = refine_transitions(abstraction.states, abstraction.images, full_state, abstraction.Plow, abstraction.Phigh, refinement_idxs, new_state_index_dict, process_dist, nothing)
    return Abstraction(abstraction.states, abstraction.images, Plow_new, Phigh_new) 
end

function refine_abstraction!(abstraction::Abstraction, image_map::Union{Function, Matrix{Float64}}, full_state::DiscreteState, refinement_idxs::Vector{Int}, uniform_error_dist::Union{Function, Distribution})
    new_state_index_dict = refine_states!(abstraction.states, refinement_idxs)
    refine_images!(abstraction.states, abstraction.images, refinement_idxs, image_map)

    Plow_new, Phigh_new = refine_transitions(abstraction.states, abstraction.images, full_state, abstraction.Plow, abstraction.Phigh, refinement_idxs, new_state_index_dict, nothing, uniform_error_dist)
    return Abstraction(abstraction.states, abstraction.images, Plow_new, Phigh_new) 
end 

function refine_abstraction!(abstraction::Abstraction, image_map::Union{Function, Matrix{Float64}}, full_state::DiscreteState, refinement_idxs::Vector{Int}, process_dist::Distribution, uniform_error_dist::Union{Function, Distribution})
    new_state_index_dict = refine_states!(abstraction.states, refinement_idxs)
    refine_images!(abstraction.states, abstraction.images, refinement_idxs, image_map)

    Plow_new, Phigh_new = refine_transitions(abstraction.states, abstraction.images, full_state, abstraction.Plow, abstraction.Phigh, refinement_idxs, new_state_index_dict, process_dist, uniform_error_dist)
    return Abstraction(abstraction.states, abstraction.images, Plow_new, Phigh_new) 
end 

# == components of the refinement process ==

function uniform_refinement(state::DiscreteState)
    deltas = (state.upper - state.lower)/2
    refined_grid = grid_generator(state.lower, state.upper, deltas)
    new_states = explicit_states(refined_grid) 
    return new_states
end

function refine_states!(states::Vector{DiscreteState}, states_to_refine::Vector{Int}; min_size=MIN_REFINE_SIZE)
    # check all the states first for minimum size as this affects new_state_idx
    length_old = length(states)
    states_skipped = []
    for state_idx in states_to_refine
        if !all(states[state_idx].upper - states[state_idx].lower .> min_size)
            push!(states_skipped, state_idx)
        end
    end

    if length(states_skipped) > 0
        setdiff!(states_to_refine, states_skipped)
        @warn "Skipped $(length(states_skipped)) states because they were too small."
    end

    new_states = Vector{DiscreteState}()
    state_map_dict = Dict() # maps the old state index to the new state indeces
    state_idx_new = length(states) - length(states_to_refine) + 1
    for state_idx_old in states_to_refine
        @assert state_idx_old  ∉ keys(state_map_dict)
        refined_states = uniform_refinement(states[state_idx_old])
        push!(new_states, refined_states...)
        state_map_dict[state_idx_old] = state_idx_new:1:(state_idx_new+length(refined_states)-1)
        state_idx_new += length(refined_states)
    end

    # map the old state indeces to their new values 
    for state_idx_old in setdiff(1:length(states), states_to_refine)    # this should add 
        @assert state_idx_old  ∉ keys(state_map_dict)
        state_map_dict[state_idx_old] = [refined_index(state_idx_old, states_to_refine),]
    end 

    deleteat!(states, states_to_refine)
    states = push!(states, new_states...)
    # @assert length_old == length(keys(state_map_dict))
    return state_map_dict 
end

function refine_images!(states::Vector{DiscreteState}, images::Vector{DiscreteState}, states_to_refine::Vector{Int}, image_map::Union{Function, Matrix{Float64}})
    new_state_idx = length(images) - length(states_to_refine) + 1
    new_images = state_images(states[new_state_idx:end], image_map)
    deleteat!(images, states_to_refine)
    push!(images, new_images...)
    @assert length(images) == length(states)
    return images
end

# function refine_uncertainties!(uncertainties::Vector{Float64}, states::Vector{DiscreteState}, states_to_refine::Vector{Int64}, sigma_fcn::Function)

#     new_state_idx =  length(uncertainties) - length(states_to_refine) + 1
#     deleteat!(uncertainties, states_to_refine)

#     new_uncertainties = zeros(length(states)-length(uncertainties))
#     Threads.@threads for i=1:length(new_uncertainties)
#         ex_state_idx = new_state_idx + i - 1
#         @views new_uncertainties[i] = sigma_fcn(states[ex_state_idx].lower, states[ex_state_idx].upper, thread_idx=Threads.threadid())[1]
#     end 
    
#     push!(uncertainties, new_uncertainties...)
#     @assert length(uncertainties) == length(states)

#     return uncertainties
# end

function refined_index(index::Int, states_to_refine::Vector{Int})
    # find the number of states in states_to_refine that is less than index
    @assert index ∉ states_to_refine # todo: @assert should not be used like this. replace with real exception handling
    num_less = length(findall(x->x<index, states_to_refine))
    return index -= num_less
end

function refined_index(indeces::Vector{Int}, states_to_refine::Vector{Int})
    refined_indeces = copy(indeces)
    for i in eachindex(refined_indeces)
        refined_indeces[i] = refined_index(refined_indeces[i], states_to_refine)
    end
    return refined_indeces 
end

function find_target_successors(source_idxs_old::Vector{Int}, target_idxs_old::Vector{Int}, state_idx_dict::Dict, Phigh_old)

    # Get the successor states of the unrefined images 
    pairs_to_target = Dict()
    for source_idx_old in source_idxs_old 
        source_idxs_new_to_recomp = []
        # get the successor states of the unrefined state
        successor_idxs_old = findall(x->x>0, Phigh_old[1:end-1, source_idx_old]) # old indeces 

        # find overlap with target_idxs
        overlap = intersect(successor_idxs_old, target_idxs_old)
        for successor_idx_old in overlap
            push!(source_idxs_new_to_recomp, state_idx_dict[successor_idx_old]) 
        end

        if !isempty(source_idxs_new_to_recomp) 
            # check if source_idx has multiple entries in the dict
            for idx_new in state_idx_dict[source_idx_old] # this should account for all the states...
                @assert idx_new ∉ keys(pairs_to_target)
                pairs_to_target[idx_new] = unique(source_idxs_new_to_recomp)
            end
        end
    end
    return pairs_to_target 
end

function cleanup(indeces::Vector)
    new_indeces = []
    total_length = 0
    for idx in indeces
        if idx isa Vector{Int}
            push!(new_indeces, idx...)
        elseif idx isa StepRange || idx isa UnitRange 
            push!(new_indeces, collect(idx)...)
        end
        total_length += length(idx)
    end    
    @assert total_length == length(new_indeces)
    return new_indeces
end

function transition_pairs_to_recompute(unrefined_states_old::Vector{Int}, states_to_refine_old::Vector{Int}, state_idx_dict::Dict, Phigh)
      # Get the transitions from unrefined states to recompute
      unrefined_transition_pairs = find_target_successors(unrefined_states_old, states_to_refine_old, state_idx_dict, Phigh)
      # Get the transitions from refined states to recompute
      refined_transition_pairs = find_target_successors(states_to_refine_old, collect(1:size(Phigh,2)-1), state_idx_dict, Phigh)          
      @assert isempty(intersect(keys(unrefined_transition_pairs), keys(refined_transition_pairs)))
    return merge(unrefined_transition_pairs, refined_transition_pairs)
end

function reuse_transitions!(Plow_new, Phigh_new, Plow_old, Phigh_old, old_indeces, new_indeces)
    Plow_new[new_indeces, new_indeces] = Plow_old[old_indeces, old_indeces]
    Phigh_new[new_indeces, new_indeces] = Phigh_old[old_indeces, old_indeces]
    Plow_new[end, new_indeces] = Plow_old[end, old_indeces]
    Phigh_new[end, new_indeces] = Phigh_old[end, old_indeces]
    Plow_new[end,end] = 1.0 
    Phigh_new[end,end] = 1.0
    return Plow_new, Phigh_new
end

function transitions_subset(states, images, full_state, all_pairs::Dict, process_noise_dist::Union{Nothing, Distribution}, uniform_error_dist::Union{Nothing, Function, Distribution})
    P_low_buffers, P_high_buffers, p_buffers, _ = initialize_buffers(length(states)+1, length(images[1].lower))
    progress_meter = Progress(length(keys(all_pairs)), desc="Calculating refined transition pairs...", dt=STATUS_BAR_PERIOD, enabled=ENABLE_PROGRESS_BAR)
    all_keys = collect(keys(all_pairs))
    Threads.@threads for i in eachindex(all_keys) 
        idx = all_keys[i]
        target_idxs = cleanup(all_pairs[idx])   # these should be NEW indeces only

        if isnothing(process_noise_dist) && isnothing(uniform_error_dist)
            @views transition_col!(P_low_buffers[Threads.threadid()][:, idx], P_high_buffers[Threads.threadid()][:, idx], states, images[idx], full_state, p_buffers[Threads.threadid()]; targets=target_idxs)
        elseif !isnothing(process_noise_dist) && isnothing(uniform_error_dist)
            @views transition_col!(P_low_buffers[Threads.threadid()][:, idx], P_high_buffers[Threads.threadid()][:, idx], states, images[idx], full_state, process_noise_dist, p_buffers[Threads.threadid()]; targets=target_idxs)
        elseif isnothing(process_noise_dist) && !isnothing(uniform_error_dist)
            if uniform_error_dist isa Function
                state_dep_dist = uniform_error_dist(states[idx].lower, states[idx].upper; thread_idx=Threads.threadid())
            else
                state_dep_dist = uniform_error_dist
            end

            @views transition_col!(P_low_buffers[Threads.threadid()][:, idx], P_high_buffers[Threads.threadid()][:, idx], states, images[idx], full_state, state_dep_dist, p_buffers[Threads.threadid()]; targets=target_idxs)
        else
            if uniform_error_dist isa Function
                state_dep_dist = uniform_error_dist(states[idx].lower, states[idx].upper; thread_idx=Threads.threadid())
            else
                state_dep_dist = uniform_error_dist
            end
            
            @views transition_col!(P_low_buffers[Threads.threadid()][:, idx], P_high_buffers[Threads.threadid()][:, idx], states, images[idx], full_state, process_noise_dist, state_dep_dist, p_buffers[Threads.threadid()]; targets=target_idxs)
        end
        next!(progress_meter)
    end 
    return P_low_buffers, P_high_buffers
end
function refine_transitions(states::Vector{DiscreteState}, images::Vector{DiscreteState}, full_state::DiscreteState, Plow, Phigh, states_to_refine_old::Vector{Int}, state_index_dict::Dict, process_dist::Union{Nothing, Distribution}, uniform_error_dist::Union{Nothing, Function, Distribution})

    # n_states_new = length(states) + 1 # add one for the sink state
    n_states_old = size(Plow,2) - 1
    unrefined_states_old = setdiff(1:n_states_old, states_to_refine_old)
    unrefined_states_new = refined_index(unrefined_states_old, states_to_refine_old)
    all_pairs = transition_pairs_to_recompute(unrefined_states_old, states_to_refine_old, state_index_dict, Phigh)

    P_low_buffers, P_high_buffers = transitions_subset(states, images, full_state, all_pairs, process_dist, uniform_error_dist)
    
    # sum the buffers to get the full matrices
    Plow_new = sum(P_low_buffers)
    Phigh_new = sum(P_high_buffers)

    # re-use the transitions from the unmodified transition pairs 
    reuse_transitions!(Plow_new, Phigh_new, Plow, Phigh, unrefined_states_old, unrefined_states_new)
    validate_transition_matrices(Plow_new, Phigh_new)

    return Plow_new, Phigh_new
end
