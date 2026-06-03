using VLBIImagePriors
using Distributions
import TransformVariables as TV
using ProbabilityTransports
using Test
using ComradeBase
using Serialization
using LinearAlgebra
using Random
using Enzyme
using Reactant


@testset "VLBIImagePriors.jl" begin
    include("angular.jl")
    include("imagepriors.jl")
    include("mrf.jl")
    include("srf.jl")
    include("simplex.jl")
    include("distributions.jl")
    include("reactant.jl")
end
