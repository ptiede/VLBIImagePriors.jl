# Reactant-friendly drop-in replacements for the scalar Distributions.jl
# distributions Paul actually uses (`Normal`/`Exponential`/`Uniform`/
# `InverseGamma`/`TDist`), built as affine transforms of a base shape. See
# `/home/ptiede/.claude/plans/i-would-like-to-rippling-ocean.md` for the
# rationale and CLAUDE.md for the high-level architecture.
#
# Layout in this directory:
#   distributions.jl       — exports, shared helpers, include order
#   std_normal.jl          — `StdNormal`     (also used by srf.jl / GMRF)
#   std_exponential.jl     — `StdExponential`
#   std_uniform.jl         — `StdUniform`
#   std_inverse_gamma.jl   — `StdInverseGamma`
#   std_tdist.jl           — `StdTDist`
#   affine.jl              — generic `AffineDistribution` wrapper
#   constructors.jl        — user-facing `VLBI*` constructors

export VLBIGaussian, VLBIExponential, VLBIUniform, VLBIInverseGamma, VLBITDist
export StdNormal, StdExponential, StdUniform, StdInverseGamma, StdTDist
export AffineDistribution, VLBITruncated
export unnormed_logpdf, lognorm


# ----- public unnormed_logpdf / lognorm interface -------------------------
# The two pieces of `logpdf(d, x) = unnormed_logpdf(d, x) + lognorm(d)`.
# Caching `lognorm` is the whole point of the split — it's data-independent
# and can be expensive (`loggamma` over an array of shape parameters,
# `sum(log, scale)` over a per-pixel scale, etc.). The `MarkovRandomField`
# subsystem uses the same names with the same semantics — see
# `src/markovrf/markovrf.jl`.

"""
    unnormed_logpdf(d, x)

Returns the part of `logpdf(d, x)` that depends on `x`. The full log-density
is `unnormed_logpdf(d, x) + lognorm(d)`.

"""
function unnormed_logpdf end

"""
    lognorm(d)

Returns the data-independent log-normalisation constant of `d`.
"""
function lognorm end


# ----- shared sampling helpers -------------------------------------------
# Marsaglia & Tsang (2000) Gamma(α, 1) sampler. The `@trace while` loop
# (from `ReactantCore`) lowers to MLIR `while_loop` under Reactant tracing
# and is a plain Julia `while` loop otherwise — early-exit on first
# accepted sample either way.

# ----- HC.ascube broadcasting helpers ------------------------------------
# Used by the per-base `HC._step_transform` / `HC._step_inverse!` overrides
# (and the AffineDistribution wrapper). The whole point is to force every
# Std base + AffineDistribution through `ArrayHC` so the round-trip is one
# broadcasting code path regardless of dimension. ScalarHC's `_step_inverse!`
# only accepts a scalar, but `transform(::ScalarHC, [u])` returns a Vector
# (from `Distributions.quantile`'s broadcast), which breaks the round-trip.
#
# `_flat_or_scalar` lets us mix `Number` and `AbstractArray` `loc`/`scale`
# (or per-element `α`/`ν`) without per-combination dispatch — scalars
# broadcast as singletons, arrays get `vec`'d so they zip with the flat
# data vector that ArrayHC operates on.
@inline _flat_or_scalar(x::Number) = x
@inline _flat_or_scalar(x::AbstractArray) = vec(x)

# Default for scalar-parameter bases (StdNormal, StdExponential, StdUniform,
# and the scalar-α/ν specialisations of StdInverseGamma/StdTDist):
# broadcast the base's element-wise kernel via `Ref(b)`.
# Per-element-parameter bases override these in their own files.
@inline _ascube_z(b, p) = _std_quantile.(Ref(b), p)
@inline _ascube_p(b, z) = _std_cdf.(Ref(b), z)


function _rand_gamma(rng::AbstractRNG, α::Number)
    # For α < 1, sample Gamma(α+1, 1) and multiply by U^(1/α).
    boost = α < one(α)
    α_eff = ifelse(boost, α + one(α), α)
    d = α_eff - one(α_eff) / 3
    c = one(α_eff) / sqrt(9 * d)

    sample = d
    done = false
    @trace while !done
        x = randn(rng)
        v = (one(α_eff) + c * x)^3
        u = rand(rng)
        v_safe = ifelse(v > zero(v), v, one(v))
        accept = (v > zero(v)) & (log(u) < x * x / 2 + d - d * v_safe + d * log(v_safe))
        sample = ifelse(accept, d * v, sample)
        done = accept
    end

    boost_u = rand(rng)
    return ifelse(boost, sample * boost_u^(one(α) / α), sample)
end


include("std_normal.jl")
include("std_exponential.jl")
include("std_uniform.jl")
include("std_inverse_gamma.jl")
include("std_tdist.jl")
include("affine.jl")
include("constructors.jl")
include("truncated.jl")
