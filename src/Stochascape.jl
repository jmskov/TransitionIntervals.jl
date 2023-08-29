module Stochascape

using Distributions
using SparseArrays
using Serialization 

using ProgressMeter

using Plots

# Write your package code here.
include("abstraction.jl")
include("refinement.jl")
include("merging.jl")
include("cluster.jl")
include("visualize.jl")

end
