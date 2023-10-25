using Stochascape
using Test
using Distributions

@testset "Stochascape.jl" begin
    # Write your tests here.

    states_to_refine = [3, 7, 8, 11]
    @test Stochascape.get_refined_index(1, states_to_refine) == 1
    @test Stochascape.get_refined_index(4, states_to_refine) == 3
    @test Stochascape.get_refined_index(9, states_to_refine) == 6
    @test Stochascape.get_refined_index(12, states_to_refine) == 8

    # image greater than, no intersection
    state = [0.0 1.0]
    image = [1.2 1.3]
    res = (-0.2, -1.3, -0.3, -1.2)
    dis = Stochascape.additive_noise_distances(image, state)
    @test all([dis[i]≈res[i] for i=1:4])

    # image greater than, intersection
    image = [0.9 1.1]
    res = (0.1, -1.1, -0.1, -0.9)
    dis = Stochascape.additive_noise_distances(image, state)
    @test all([dis[i]≈res[i] for i=1:4])

    # image contained
    image = [0.5 0.6]
    res = (0.5, -0.6, 0.4, -0.5)
    dis = Stochascape.additive_noise_distances(image, state)
    @test all([dis[i]≈res[i] for i=1:4])

    # image less than, intersection
    image = [-0.1 0.1]
    res = (1.1, -0.1, 0.9, 0.1)
    dis = Stochascape.additive_noise_distances(image, state)
    @test all([dis[i]≈res[i] for i=1:4]) 

    # image less than, no intersection
    image = [-1.1 -0.9]
    res = (2.1, 0.9, 1.9, 1.1)
    dis = Stochascape.additive_noise_distances(image, state)
    @test all([dis[i]≈res[i] for i=1:4]) 

    # image larger than state
    image = [-0.1, 1.1]
    res = [1.1, -1.1, 0, 0]
    dis = Stochascape.additive_noise_distances(image, state)
    @test all([dis[i]≈res[i] for i=1:4]) 

    # symmetry test

    dist = Truncated(Normal(0., sqrt(0.09)), -0.4, 0.4) 
    image1 = [-0.9  -0.775; -0.5  -0.375]
    image2 = [0.775  0.9; 0.375  0.5]
    state1 = [ -1.0 0.0; -1.0 0.0]
    state2 = [0.0 1.0; 0.0 1.0]

    plow1, phigh1 = Stochascape.simple_transition_bounds(image1, state1, dist)
    plow2, phigh2 = Stochascape.simple_transition_bounds(image2, state2, dist)

    @test plow1 ≈ plow2 && phigh1 ≈ phigh2

    # test super state creation
    new_state = Stochascape.build_super_state([state1])
    @test new_state == state1

    succ_res1 = [0.0, 0.0, 0.481371, 0.948044, 0.98234, 0.833864, 0.86409, 1.0]
    P_low1 = [0.0, 0.0, 0.07989039614518836, 0.0, 0.0022670874692132326, 0.0, 0.021375050920686894, 0.6480394048490924]
    P_high1 = [0.006009936966706811, 0.014735094734041018, 0.23739449986244304, 0.0050637403571615475, 0.04625166326820901, 0.017971005861694717, 0.16414524700710229, 0.8343690688377126]

    succ_res2 =  [0.0, 0.0, 0.481371, 0.86409, 0.833864, 0.98234, 0.948044, 1.0]
    P_low2 = [0.0, 0.0, 0.07989039614518838, 0.021375050920686874, 0.0, 0.0022670874692132434, 0.0, 0.6480394048490924]
    P_high2 = [0.014735094734040928, 0.006009936966706775, 0.23739449986244304, 0.16414524700710226, 0.01797100586169474, 0.04625166326820904, 0.005063740357161557, 0.8343690688377126]

    idx_perm1 = sortperm(succ_res1)
    p_true1 = Stochascape.true_transition_probabilities(P_low1, P_high1, idx_perm1)
    idx_perm2 = sortperm(succ_res2)
    p_true2 = Stochascape.true_transition_probabilities(P_low2, P_high2, idx_perm2)

    ver_p1 = sum(p_true1 .* succ_res1) 
    ver_p2 = sum(p_true2 .* succ_res2)  

    @test ver_p1 ≈ ver_p2

    # test out the multiplicative nosie 
    Stochascape.MULTIPLICATIVE_NOISE_FLAG = true

    # both positive
    A, B = [-1.0, 1.0] 
    C, D = [0.5, 1.5]
    expected_result = [B/C, A/D, B/D, A/C]
    @test all(Stochascape.multiplicative_noise_distances([C, D], [A, B]) .≈ expected_result)
    test_dist = Uniform(0.5, 1.5)
    @info plow, phigh = Stochascape.simple_transition_bounds([C D], [A B], test_dist)

    # C < 0
    C, D = [-0.5, 0.4]
    expected_result = [Inf, 0, A/C, 0]
    @test all(Stochascape.multiplicative_noise_distances([C, D], [A, B]) .≈ expected_result)
    test_dist = Uniform(1.0, 3.0)
    @info plow, phigh = Stochascape.simple_transition_bounds([C D], [A B], test_dist)

    # C < 0 and A > 0
    # both positive
    A, B = [0.0, 1.0] 
    C, D = [-0.5, 0.4]
    expected_result = [Inf, 0, 0, 0]
    @test all(Stochascape.multiplicative_noise_distances([C, D], [A, B]) .≈ expected_result)
    test_dist = Uniform(1.0, 3.0)
    @info plow, phigh = Stochascape.simple_transition_bounds([C D], [A B], test_dist)

    # test static partition bounds
    res_dyn = Stochascape.simple_transition_bounds(image1, state1, dist)

    # test static partition bounds
    Stochascape.USE_STATIC_PARTITIONS = true
    Stochascape.STATIC_PARTITION_BOUNDS = [-0.2, 0.2]
    res_sta = Stochascape.simple_transition_bounds(image1, state1, dist)
    @test res_sta[2] > res_dyn[2]
end