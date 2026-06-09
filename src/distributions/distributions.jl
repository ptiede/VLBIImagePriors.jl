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
export AffineDistribution, VLBITruncated, VLBIBeta


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
    # Promote the shape to a traced number while compiling under Reactant so the
    # loop-carried `dv` below is traced from the start — otherwise a `@trace
    # while` nested inside a sampler's `@trace for` fails with "arguments should
    # be traced but were not". See ProbabilityTransports' `_rand_gamma`.
    af = float(α)
    a = within_compile() ? promote_to_traced(af) : af
    T = typeof(a)

    # shape < 1 via the boost identity `Gamma(a) = Gamma(a+1) · U^(1/a)`.
    cond = a < one(T)
    boost = ifelse(cond, rand(rng, T)^inv(a), one(T))
    a = ifelse(cond, a + one(T), a)

    d = a - one(T) / 3
    c = inv(sqrt(9 * d))

    dv = zero(T)
    i = 0
    @trace while (dv == zero(dv)) & (i < 32)
        x = randn(rng, T)
        v = (one(T) + c * x)^3
        u = rand(rng, T)
        vpos = v > zero(T)
        vsafe = ifelse(vpos, v, one(T))                # keep `log(vsafe)` finite when v <= 0
        cand = vpos & (log(u) < x * x / 2 + d - d * vsafe + d * log(vsafe))
        dv = ifelse(cand, d * vsafe, dv)               # keep the first accepted draw
        i = i + 1
    end
    return boost * dv
end


include("std_normal.jl")
include("std_exponential.jl")
include("std_uniform.jl")
include("std_inverse_gamma.jl")
include("std_tdist.jl")
include("affine.jl")
include("constructors.jl")
include("truncated.jl")
include("beta.jl")
