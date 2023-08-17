using Stochascape
using Test

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
    dis = Stochascape.find_distances(image, state)
    @test all([dis[i]≈res[i] for i=1:4])

    # image greater than, intersection
    image = [0.9 1.1]
    res = (0.1, -1.1, -0.1, -0.9)
    dis = Stochascape.find_distances(image, state)
    @test all([dis[i]≈res[i] for i=1:4])

    # image contained
    image = [0.5 0.6]
    res = (0.5, -0.6, 0.4, -0.5)
    dis = Stochascape.find_distances(image, state)
    @test all([dis[i]≈res[i] for i=1:4])

    # image less than, intersection
    image = [-0.1 0.1]
    res = (1.1, -0.1, 0.9, 0.1)
    dis = Stochascape.find_distances(image, state)
    @test all([dis[i]≈res[i] for i=1:4]) 

    # image less than, no intersection
    image = [-1.1 -0.9]
    res = (2.1, 0.9, 1.9, 1.1)
    dis = Stochascape.find_distances(image, state)
    @test all([dis[i]≈res[i] for i=1:4]) 

    # image larger than state
    image = [-0.1, 1.1]
    res = [1.1, -1.1, 0, 0]
    dis = Stochascape.find_distances(image, state)
    @test all([dis[i]≈res[i] for i=1:4]) 

end
