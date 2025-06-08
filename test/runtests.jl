using VLBIImagePriors
using ChainRulesCore
using ChainRulesTestUtils
using Distributions
using FiniteDifferences
using Zygote
import TransformVariables as TV
using HypercubeTransform
using Test
using ComradeBase
using Serialization
using LinearAlgebra
using Enzyme




@testset "VLBIImagePriors.jl" begin
    include("angular.jl")
    include("centereg.jl")
    include("imagepriors.jl")
    include("mrf.jl")
    include("simplex.jl")
    include("rules.jl")
end
