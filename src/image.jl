# functions to help compute images

#= 
generate images
=#
function state_image(state::DiscreteState, image_map::Function, thread_idx::Int=1)
    try     # try-catch thread_idx to avoid error when not using threads
        return DiscreteState(image_map(state.lower, state.upper, thread_idx=thread_idx)...)
    catch
        return DiscreteState(image_map(state.lower, state.upper)...)
    end
end

function state_images(states::Vector{DiscreteState}, image_map::Function)
    progress_meter = Progress(length(states), "Computing state images...", dt=STATUS_BAR_PERIOD, enabled=ENABLE_PROGRESS_BAR)
    images = Vector{DiscreteState}(undef, length(states))
    Threads.@threads for i in eachindex(states)
        images[i] = state_image(states[i], image_map, Threads.threadid())
        next!(progress_meter)
    end
    return images
end

function state_images(states::Vector{DiscreteState}, system_matrix::Matrix{Float64})
    progress_meter = Progress(length(states), "Computing state images from linear map...", dt=STATUS_BAR_PERIOD, enabled=ENABLE_PROGRESS_BAR)
    images = Vector{DiscreteState}(undef, length(states))
    Threads.@threads for i in eachindex(states)
        images[i] = linear_system_image(system_matrix, states[i])
        next!(progress_meter)
    end
    # todo: warning for degenerate images
    return images
end

function linear_system_image(system_matrix::Matrix{Float64}, state::DiscreteState)
    c1 = system_matrix*state.lower
    c2 = system_matrix*state.upper
    # now compute northwestern and southeastern corners
    c3 = system_matrix*[state.lower[1], state.upper[2]]
    c4 = system_matrix*[state.upper[1], state.lower[2]]
    lower = zeros(length(c1))
    upper = zeros(length(c1))
    for i in eachindex(c1) 
        lower[i] = min(c1[i], c2[i], c3[i], c4[i])
        upper[i] = max(c1[i], c2[i], c3[i], c4[i])
    end
    return  DiscreteState(lower, upper)
end
