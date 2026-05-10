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


include("std_normal.jl")
include("std_exponential.jl")
include("std_uniform.jl")
include("std_inverse_gamma.jl")
include("std_tdist.jl")
include("affine.jl")
include("constructors.jl")
