using TransitionIntervals
using Test
using Distributions

@testset "TransitionIntervals.jl" begin

    # test 1D discretization
    grid = TransitionIntervals.dimension_grid(0.0, 1.0, 0.25) 
    @test grid == 0.0:0.25:0.75
    @test TransitionIntervals.dimension_grid(0.0, 1.0, 0.26) == 0.0:(1/3):(2/3)
    @test TransitionIntervals.dimension_grid(0.0, 1.0, 0.24) == 0.0:0.25:0.75

    # todo: this test fails
    # @test TransitionIntervals.explicit_states(grid) == [
    #     [0.0 0.25],
    #     [0.25 0.5],
    #     [0.5 0.75],
    #     [0.75 1.0]
    # ]

    # test 2D discretization
    discretization = UniformDiscretization(
        DiscreteState([0.0, 0.0], [1.0, 1.0]),
        [0.5, 0.5]
    )
    grid = TransitionIntervals.grid_generator(discretization.compact_space.lower, discretization.compact_space.upper, discretization.spacing)
    @test grid == Iterators.product(0.0:0.5:0.5, 0.0:0.5:0.5)
    @test grid == TransitionIntervals.grid_generator(discretization)

    # todo: this test fails
    # @test TransitionIntervals.explicit_states(grid) == Vector{DiscreteState}([
    #     DiscreteState([0.0, 0.0], [0.5, 0.5]),
    #     DiscreteState([0.5, 0.0], [1.0, 0.5]),
    #     DiscreteState([0.0, 0.5], [0.5, 1.0]),
    #     DiscreteState([0.5, 0.5], [1.0, 1.0]),
    # ])

    # todo: is image testing in my scope? I don't provide functions for them...

    # image greater than, no intersection
    state = [0.0 1.0]
    image = [1.2 1.3]
    res = (-0.2, -1.3, -0.3, -1.2)
    dis = TransitionIntervals.additive_noise_distances(image, state)
    @test all([dis[i]≈res[i] for i=1:4])

    # image greater than, intersection
    image = [0.9 1.1]
    res = (0.1, -1.1, -0.1, -0.9)
    dis = TransitionIntervals.additive_noise_distances(image, state)
    @test all([dis[i]≈res[i] for i=1:4])

    # image contained
    image = [0.5 0.6]
    res = (0.5, -0.6, 0.4, -0.5)
    dis = TransitionIntervals.additive_noise_distances(image, state)
    @test all([dis[i]≈res[i] for i=1:4])

    # image less than, intersection
    image = [-0.1 0.1]
    res = (1.1, -0.1, 0.9, 0.1)
    dis = TransitionIntervals.additive_noise_distances(image, state)
    @test all([dis[i]≈res[i] for i=1:4]) 

    # image less than, no intersection
    image = [-1.1 -0.9]
    res = (2.1, 0.9, 1.9, 1.1)
    dis = TransitionIntervals.additive_noise_distances(image, state)
    @test all([dis[i]≈res[i] for i=1:4]) 

    # image larger than state
    image = [-0.1, 1.1]
    res = (1.1, -1.1, 0, 0)
    dis = TransitionIntervals.additive_noise_distances(image, state)
    @test all([dis[i]≈res[i] for i=1:4]) 

    # state relations
    state = DiscreteState([0.], [1.])
    @test TransitionIntervals.contains(state, DiscreteState([0.5], [0.6])) == true
    @test TransitionIntervals.contains(state, DiscreteState([0.9], [1.1])) == false
    @test TransitionIntervals.intersects(state, DiscreteState([0.9], [1.1])) == true
    @test TransitionIntervals.intersects(state, DiscreteState([1.1], [1.2])) == false

    target_state = DiscreteState([0.0], [1.0])
    # deterministic transition
    @test TransitionIntervals.transition_interval(DiscreteState([0.0], [0.5]), target_state) == [1.0, 1.0]
    @test TransitionIntervals.transition_interval(DiscreteState([0.5], [1.0]), target_state) == [1.0, 1.0]

    # transition with additive stochastic noise 
    dist = Normal(0.0, 0.1)
    res1 = TransitionIntervals.transition_interval(DiscreteState([0.0], [0.5]), target_state, dist)
    @test res1[1] == cdf(dist, 0.5) - cdf(dist, 0.0)
    @test res1[2] == 1 - (1 - cdf(dist, 1.0)) - cdf(dist, -0.5)

    # transition with additive stochatic noise and uniform error
    dist = Normal(0.0, 0.1)
    # uniform dist
    err_dist = TransitionIntervals.GaussianUniformError(0.1, 1.0)
    res2 = TransitionIntervals.transition_interval(DiscreteState([0.0], [0.5]), target_state, dist, err_dist)
    @test res1[1] >= res2[1] && res1[2] <= res2[2] # ! this is not a good test

    # todo: test the sub optimization procedures
    
     # symmetry test
    dist = truncated(Normal(0., sqrt(0.09)), -0.4, 0.4) 
    image1 = DiscreteState([-0.9, -0.5], [-0.775, -0.375])
    image2 = DiscreteState([0.775, 0.375], [0.9, 0.5])
    state1 = DiscreteState([-1.0, -1.0], [0.0, 0.0])
    state2 = DiscreteState([0.0, 0.0], [1.0, 1.0])

    plow1, phigh1 = TransitionIntervals.transition_interval(image1, state1, dist)
    plow2, phigh2 = TransitionIntervals.transition_interval(image2, state2, dist)
    @test plow1 ≈ plow2 && phigh1 ≈ phigh2

    # probability values test
    succ_res1 = [0.0, 0.0, 0.481371, 0.948044, 0.98234, 0.833864, 0.86409, 1.0]
    P_low1 = [0.0, 0.0, 0.07989039614518836, 0.0, 0.0022670874692132326, 0.0, 0.021375050920686894, 0.6480394048490924]
    P_high1 = [0.006009936966706811, 0.014735094734041018, 0.23739449986244304, 0.0050637403571615475, 0.04625166326820901, 0.017971005861694717, 0.16414524700710229, 0.8343690688377126]

    succ_res2 =  [0.0, 0.0, 0.481371, 0.86409, 0.833864, 0.98234, 0.948044, 1.0]
    P_low2 = [0.0, 0.0, 0.07989039614518838, 0.021375050920686874, 0.0, 0.0022670874692132434, 0.0, 0.6480394048490924]
    P_high2 = [0.014735094734040928, 0.006009936966706775, 0.23739449986244304, 0.16414524700710226, 0.01797100586169474, 0.04625166326820904, 0.005063740357161557, 0.8343690688377126]

    idx_perm1 = sortperm(succ_res1)
    p_true1 = TransitionIntervals.true_transition_probabilities(P_low1, P_high1, idx_perm1)
    idx_perm2 = sortperm(succ_res2)
    p_true2 = TransitionIntervals.true_transition_probabilities(P_low2, P_high2, idx_perm2)

    ver_p1 = sum(p_true1 .* succ_res1) 
    ver_p2 = sum(p_true2 .* succ_res2)  

    @test ver_p1 ≈ ver_p2

    # refinement tests
    # res_ex = Vector{DiscreteState}([DiscreteState([0.0], [0.25]), DiscreteState([0.25], [0.5])])
    # @test TransitionIntervals.uniform_refinement(DiscreteState([0.0], [0.5])) == res_ex
    # @info Vector{DiscreteState}([DiscreteState([0.0], [0.25]), DiscreteState([0.25], [0.5])])
    # @info TransitionIntervals.uniform_refinement(DiscreteState([0.0], [0.5]))

    states_to_refine = [3, 7, 8, 11]
    @test TransitionIntervals.refined_index(1, states_to_refine) == 1
    @test TransitionIntervals.refined_index(4, states_to_refine) == 3
    @test TransitionIntervals.refined_index(9, states_to_refine) == 6
    @test TransitionIntervals.refined_index(12, states_to_refine) == 8

    # test the refinement of the states
    states = Vector{DiscreteState}([
        DiscreteState([0.0], [0.25]), 
        DiscreteState([0.25], [0.5]), 
        DiscreteState([0.5], [0.75]), 
        DiscreteState([0.75], [1.0])
    ])
    states_to_refine = [3,]
    res = TransitionIntervals.refine_states!(states, states_to_refine)
    @test res == Dict(3 => 4:5, 1 => [1,], 2 => [2,], 4 => [3,])

    # ! all the below tests should work after the refactoring


    # # test super state creation
    # new_state = Stochascape.build_super_state([state1])
    # @test new_state == state1



    # # test out the multiplicative nosie 
    # # TODO: multiplicative noise tests fail
    # Stochascape.MULTIPLICATIVE_NOISE_FLAG = true

    # # both positive
    # A, B = [-1.0, 1.0] 
    # C, D = [0.5, 1.5]
    # expected_result = [B/C, A/D, B/D, A/C]
    # @test all(Stochascape.multiplicative_noise_distances([C, D], [A, B]) .≈ expected_result)
    # test_dist = Uniform(0.5, 1.5)
    # @info plow, phigh = Stochascape.simple_transition_interval([C D], [A B], test_dist)

    # # C < 0
    # C, D = [-0.5, 0.4]
    # expected_result = [Inf, 0, A/C, 0]
    # @test all(Stochascape.multiplicative_noise_distances([C, D], [A, B]) .≈ expected_result)
    # test_dist = Uniform(1.0, 3.0)
    # @info plow, phigh = Stochascape.simple_transition_interval([C D], [A B], test_dist)

    # # C < 0 and A > 0
    # # both positive
    # A, B = [0.0, 1.0] 
    # C, D = [-0.5, 0.4]
    # expected_result = [Inf, 0, 0, 0]
    # @test all(Stochascape.multiplicative_noise_distances([C, D], [A, B]) .≈ expected_result)
    # test_dist = Uniform(1.0, 3.0)
    # @info plow, phigh = Stochascape.simple_transition_interval([C D], [A B], test_dist)

    # # test static partition bounds
    # res_dyn = Stochascape.simple_transition_interval(image1, state1, dist)

    # # test static partition bounds
    # Stochascape.USE_STATIC_PARTITIONS = true
    # Stochascape.STATIC_PARTITION_BOUNDS = [-0.2, 0.2]
    # res_sta = Stochascape.simple_transition_interval(image1, state1, dist)
    # # todo: fix this test
    # @test res_sta[2] > res_dyn[2]
end