# standard loading and saving functions

function save_abstraction(abstraction::Abstraction, results_dir::String)
    # 0. User-Specified Parameters
    state_filename = "$results_dir/states-explicit.ser"
    images_filename = "$results_dir/images-explicit.ser"
    matrices_filename = "$results_dir/matrices-explicit.ser"

    # 1. IMC Abstraction
    serialize(state_filename, abstraction.states)
    serialize(images_filename, abstraction.images)
    serialize(matrices_filename, Dict("Plow" => abstraction.Plow, "Phigh" => abstraction.Phigh))
end

function load_abstraction(results_dir::String)
    # 0. User-Specified Parameters
    state_filename = "$results_dir/states-explicit.ser"
    images_filename = "$results_dir/images-explicit.ser"
    matrices_filename = "$results_dir/matrices-explicit.ser"

    # 1. IMC Abstraction
    states = deserialize(state_filename)
    images = deserialize(images_filename)
    res = deserialize(matrices_filename)
    Plow = res["Plow"]
    Phigh = res["Phigh"]

    return Abstraction(states, images, Plow, Phigh)
end