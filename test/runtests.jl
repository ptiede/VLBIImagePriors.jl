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

# AD test helpers. ChainRules/Zygote/FiniteDifferences were dropped (PT relies on
# Enzyme/Reactant through the primal), so gradient cross-checks use Enzyme reverse
# mode against a local central-difference reference.
enzyme_grad(f, x) =
    Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(f), x)[1]
function fdm_grad(f, x::AbstractArray; h = 1.0e-5)
    xc = collect(float.(x))
    g = similar(xc)
    for i in eachindex(xc)
        xi = xc[i]
        xc[i] = xi + h
        fp = f(xc)
        xc[i] = xi - h
        fm = f(xc)
        xc[i] = xi
        g[i] = (fp - fm) / (2h)
    end
    return reshape(g, size(x))
end


@testset "VLBIImagePriors.jl" begin
    include("angular.jl")
    include("imagepriors.jl")
    include("mrf.jl")
    include("srf.jl")
    include("simplex.jl")
    include("distributions.jl")
    include("reactant.jl")
end
