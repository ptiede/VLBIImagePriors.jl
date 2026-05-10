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
export AffineDistribution
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

function _rand_gamma(rng::AbstractRNG, α::Real)
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
