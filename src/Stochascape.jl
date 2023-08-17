module Stochascape

using Distributions
using SparseArrays
using Serialization 

using Plots

# Write your package code here.
include("abstraction.jl")
include("refinement.jl")
include("visualize.jl")

end
