module Stochascape

using Distributions
import Distributions.cdf
using SparseArrays
using Serialization 

using ProgressMeter
global STATUS_BAR_PERIOD = 30.0
global MIN_REFINE_SIZE = 0.001

global MULTIPLICATIVE_NOISE_FLAG = false

global USE_STATIC_PARTITIONS = false
global STATIC_PARTITION_BOUNDS = [-1.0, 1.0]

using Plots
using Colors

# Write your package code here.
include("abstraction.jl")
include("transitions.jl")
include("refinement.jl")
include("merging.jl")
include("cluster.jl")
include("output.jl")
include("utilities.jl")
include("visualize.jl")

end
