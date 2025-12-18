module AugmentedTTN

# Include submodules 
include("Topology.jl")
include("Operators.jl")
include("TTNState.jl")
include("DMRG.jl")

# Bring selected names into the top-level namespace
using .TTNState: BinaryTTN
using .DMRG: dmrg

# Public API
export BinaryTTN, dmrg

end # module AugmentedTTN
