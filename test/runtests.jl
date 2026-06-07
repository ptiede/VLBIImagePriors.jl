using VLBIImagePriors
using VLBIImagePriors: unnormed_logpdf, lognorm
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
using Random
using Enzyme
using Zygote
using Reactant


@testset "VLBIImagePriors.jl" begin
    include("angular.jl")
    include("imagepriors.jl")
    include("mrf.jl")
    include("srf.jl")
    include("simplex.jl")
    include("rules.jl")
    include("distributions.jl")
    include("reactant.jl")
end
