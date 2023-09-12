module Stochascape

using Distributions
using SparseArrays
using Serialization 

using ProgressMeter
global STATUS_BAR_PERIOD = 30.0
global MIN_REFINE_SIZE = 0.001

global MULTIPLICATIVE_NOISE_FLAG = false

using Plots
using Colors

# Write your package code here.
include("abstraction.jl")
include("refinement.jl")
include("merging.jl")
include("cluster.jl")
include("visualize.jl")

end
