module Stochascape

using Distributions
using SparseArrays
using Serialization 

using ProgressMeter
global STATUS_BAR_PERIOD = 30.0

global MULTIPLICATIVE_NOISE_FLAG = false

using Plots

# Write your package code here.
include("abstraction.jl")
include("refinement.jl")
include("merging.jl")
include("cluster.jl")
include("visualize.jl")

end
