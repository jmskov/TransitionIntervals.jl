module TransitionIntervals 

using Distributions
import Distributions.cdf
using SparseArrays
using Serialization 

using ProgressMeter
global ENABLE_PROGRESS_BAR = true
global STATUS_BAR_PERIOD = 30.0

global MIN_REFINE_SIZE = 0.001
global MULTIPLICATIVE_NOISE_FLAG = false
global USE_STATIC_PARTITIONS = false
global STATIC_PARTITION_BOUNDS = [-1.0, 1.0]

using Plots
import Plots.plot

include("utilities.jl")
export UniformError

include("discretization.jl")
export discretize
export DiscreteState, UniformDiscretization

include("image.jl")
include("distance.jl")
include("transitions.jl")

include("abstraction.jl")
export transition_intervals

include("refinement.jl")
export refine_abstraction

include("output.jl")

include("visualize.jl")
export plot, plot!

end
