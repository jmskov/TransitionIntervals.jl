# Functions for abstraction
Base.@kwdef struct AbstractionProblem
    compact_space::Matrix{Float64}
    spacing::Vector{Float64}
    image_map::Function # function
    process_noise_distribution::Distribution = Normal(0.0, 0.0)
    state_dependent_uncertainty_map::Union{Nothing, Function} = nothing
end

Base.@kwdef struct Abstraction
    states::Vector{Matrix{Float64}}
    images::Vector{Matrix{Float64}}
    Plow::SparseMatrixCSC{Float64,Int64}
    Phigh::SparseMatrixCSC{Float64,Int64}
    uncertainties::Union{Nothing, Vector{Float64}} = nothing
end

function imc_abstraction(problem::AbstractionProblem)
    return Abstraction(Stochascape.imc_abstraction(problem.compact_space, problem.spacing, problem.image_map, problem.process_noise_distribution, problem.state_dependent_uncertainty_map)...)
end

function update_abstraction!(abstraction::Abstraction, problem::AbstractionProblem, image_map::Function, uncertainty_map::Union{Nothing, Function}=nothing)

    abstraction.images[:] = calculate_all_images(abstraction.states, image_map)
    if !isnothing(uncertainty_map)
        bound_sigmas!(abstraction.uncertainties, abstraction.states, uncertainty_map)
        # todo: check uncertainties here
    end

    Plow, Phigh = calculate_transition_probabilities(abstraction.states, abstraction.images, problem.compact_space, problem.process_noise_distribution, abstraction.uncertainties)
    abstraction.Plow[:] = Plow
    abstraction.Phigh[:] = Phigh
    return abstraction
end

# Full abstraction
function imc_abstraction(full_state, spacing, image_map, noise_distribution, uncertainty_fcn::Union{Nothing, Function}=nothing)
    grid, grid_spacing = grid_generator(full_state[:,1], full_state[:,2], spacing)
    states = calculate_explicit_states(grid, grid_spacing)
    images = calculate_all_images(states, image_map)

    if !isnothing(uncertainty_fcn)
        uncertainties = zeros(length(states))
        bound_sigmas!(uncertainties, states, uncertainty_fcn) 
        # todo: save these?
    else
        uncertainties = nothing 
    end

    Plow, Phigh = calculate_transition_probabilities(states, images, full_state, noise_distribution, uncertainties)
    return states, images, Plow, Phigh, uncertainties
end

function bound_sigmas!(sigmas, states, sigma_fcn)
    Threads.@threads for i in eachindex(states)
        sigma_bnd = sigma_fcn(states[i], thread_idx=Threads.threadid())[1]
        sigmas[i] = sigma_bnd
    end
    return sigmas
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
    buffer = zeros(4)
    return additive_noise_distances!(image, target, buffer)
end

function additive_noise_distances!(image, target, buffer)
    return additive_noise_distances!(image[1], image[2], target[1], target[2], buffer)
end

function additive_noise_distances!(C, D, A, B, buffer)
    W = D - C

    Δ1 = abs(B-C)
    Δ2 = -abs(A-D)
    # check if target contains image
    if A <= C <= D <= B
        Δ3 = abs(B-D)
        Δ4 = -abs(A-C)
    # check if image is larger than target
    elseif C <= A <= B <= D
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
    
    buffer[1] = Δ1
    buffer[2] = Δ2
    buffer[3] = Δ3
    buffer[4] = Δ4
    return buffer
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

function initialize_transition_matrices(nstates)
    Plow = spzeros(nstates, nstates)
    Phigh = spzeros(nstates, nstates)
    return Plow, Phigh
end

function state_radius(state)
    return 0.5*sqrt(sum((state[:,2] - state[:,1]).^2))
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