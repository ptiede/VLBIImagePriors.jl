# VLBIBeta — Beta distribution over [0, 1], as a (multivariate) product of
# independent Beta(α[i], β[i]) marginals. `α`/`β` may be scalars (a single
# Beta) or equal-length vectors (one Beta per element).
#
# pdf(x; α, β) = x^(α-1) (1-x)^(β-1) / B(α, β),  B(α, β) = Γ(α)Γ(β)/Γ(α+β).
#
# Like the rest of this directory, no constructor validation is performed:
# Reactant cannot throw, and shape mismatches surface downstream in the loops
# below. See constructors.jl for the rationale.

export VLBIBeta

struct VLBIBeta{T, L} <: Dists.ContinuousMultivariateDistribution
    α::T
    β::T
    lognorm::L
end

Base.length(d::VLBIBeta) = length(d.α)
Base.eltype(d::VLBIBeta) = eltype(d.α)
Dists.insupport(::VLBIBeta, x) = all(xi -> zero(xi) <= xi <= one(xi), x)

function VLBIBeta(α::AbstractVector, β::AbstractVector)
    return VLBIBeta(α, β, _lognorm_beta(α, β))
end

function VLBIBeta(α::Number, β::Number)
    return VLBIBeta(α, β, _lognorm_beta(α, β))
end

# Additive log-normalisation: log(1 / B(α, β)) = loggamma(α+β) - loggamma(α)
# - loggamma(β), so that logpdf = unnormed_logpdf + lognorm (see
# distributions.jl for the family-wide convention). Reductions use
# broadcasting + `sum` rather than a `@trace` loop, matching the Std*
# distributions — Reactant lowers `sum` cleanly but chokes on a `@trace for`
# scalar reduction (it forces a `while_loop`).
_lognorm_beta(α::Number, β::Number) = loggamma(α + β) - loggamma(α) - loggamma(β)
_lognorm_beta(α::AbstractVector, β::AbstractVector) = sum(_lognorm_beta.(α, β))

lognorm(d::VLBIBeta) = d.lognorm

function unnormed_logpdf(d::VLBIBeta, x::AbstractVector{<:Number})
    return sum(_logpdf_beta.(d.α, d.β, x))
end
function unnormed_logpdf(d::VLBIBeta{<:Number}, x::Number)
    return _logpdf_beta(d.α, d.β, x)
end

function Dists.logpdf(d::VLBIBeta, x::AbstractVector{<:Number})
    return unnormed_logpdf(d, x) + d.lognorm
end
function Dists._logpdf(d::VLBIBeta, x::AbstractVector{<:Real})
    return unnormed_logpdf(d, x) + d.lognorm
end
Dists.logpdf(d::VLBIBeta{<:Number}, x::Number) = unnormed_logpdf(d, x) + d.lognorm

function _logpdf_beta(α::Number, β::Number, x::Number)
    y = clamp(x, zero(x), one(x))
    r = xlogy(α - 1, y) + xlog1py(β - 1, -y)
    return ifelse((x > 0) & (x < 1), r, oftype(y, -Inf))
end


# ----- sampling -----------------------------------------------------------
# Beta(α, β) via the gamma ratio: with X ~ Gamma(α, 1) and Y ~ Gamma(β, 1),
# X / (X + Y) ~ Beta(α, β). `_rand_gamma` (ProbabilityTransports) is the shared
# Reactant-friendly Marsaglia–Tsang sampler, so this whole path traces.
@inline function _rand_beta(rng::AbstractRNG, α::Number, β::Number)
    x = _rand_gamma(rng, α)
    y = _rand_gamma(rng, β)
    return x / (x + y)
end

# Fill with a `@trace for` loop, not a broadcast: under Reactant `rand` is
# unsupported inside a `.`-broadcast (the `elem_apply` lowering), so
# `_rand_beta.(Ref(rng), α, β)` fails, while the scalar `_rand_beta` works fine
# in the loop. `rgetindex`/`rsetindex!` (ComradeBase) are the Reactant-safe
# element accessors that keep the loop from tripping the scalar-indexing guard,
# and the loop threads the RNG across iterations (independent per-pixel draws).
function Dists._rand!(
        rng::AbstractRNG, d::VLBIBeta{<:AbstractVector}, x::AbstractVector{<:Number}
    )
    α = d.α
    β = d.β
    @trace for i in eachindex(x)
        rsetindex!(x, _rand_beta(rng, rgetindex(α, i), rgetindex(β, i)), i)
    end
    return x
end
function Dists._rand!(
        rng::AbstractRNG, d::VLBIBeta{<:Number}, x::AbstractVector{<:Number}
    )
    α = d.α
    β = d.β
    @trace for i in eachindex(x)
        rsetindex!(x, _rand_beta(rng, α, β), i)
    end
    return x
end
