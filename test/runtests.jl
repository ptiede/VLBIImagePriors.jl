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
using ComponentArrays
using Enzyme
Enzyme.API.runtimeActivity!(true)




@testset "VLBIImagePriors.jl" begin
    include("angular.jl")
    include("centering.jl")
    include("imagepriors.jl")
    include("mrf.jl")
    include("named_comp_dist.jl")
    include("simplex.jl")
end
