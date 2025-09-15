using VLBIImagePriors
using ChainRulesCore
using ChainRulesTestUtils
using Distributions
using FiniteDifferences
import TransformVariables as TV
using HypercubeTransform
using Test
using ComradeBase
using Serialization
using LinearAlgebra
using Enzyme
using Zygote


@testset "VLBIImagePriors.jl" begin
    include("angular.jl")
    include("centereg.jl")
    include("imagepriors.jl")
    include("mrf.jl")
    include("srf.jl")
    include("simplex.jl")
    include("rules.jl")
end
