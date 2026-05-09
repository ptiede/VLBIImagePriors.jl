# Reactant-friendly drop-in replacements for the four scalar Distributions.jl
# distributions Paul actually uses (`Normal`/`Exponential`/`Uniform`/
# `InverseGamma`), built as affine transforms of a base shape. See plan
# at `/home/ptiede/.claude/plans/i-would-like-to-rippling-ocean.md` for the
# rationale; CLAUDE.md has the high-level architecture.
#
# Layout in this directory:
#   distributions.jl  — exports, shared helpers, and the include order
#   bases.jl          — `StdExponential`, `StdUniform`, `StdInverseGamma`,
#                       per-base log-pdf / cdf / quantile kernels, and the
#                       `Distributions` interface for those bases
#   affine.jl         — generic `AffineDistribution` wrapper and its
#                       `Distributions` interface
#   constructors.jl   — user-facing `VLBI{Gaussian,Exponential,Uniform,
#                       InverseGamma}` constructors

export VLBIGaussian, VLBIExponential, VLBIUniform, VLBIInverseGamma
export StdNormal, StdExponential, StdUniform, StdInverseGamma
export AffineDistribution
export unnormed_logpdf, lognorm


# ----- shared helpers ------------------------------------------------------

# Per-element parameter access. `_at(p, i)` returns `p` if `p` is a scalar
# Number (broadcast) or `p[i]` if `p` is an array. Resolved at compile time so
# the per-element loops in the array kernels stay monomorphic.
@inline _at(p::Number, _) = p
@inline _at(p::AbstractArray, i) = @inbounds p[i]

# eltype of a parameter that may be a Number or an AbstractArray.
@inline _peltype(p::Number) = typeof(p)
@inline _peltype(p::AbstractArray) = eltype(p)

# Element type used to instantiate the Std base when a user supplies one of
# the constructor functions. Reactant tracers (`<:Number` but not `<:Real`)
# fall back to `Float64` so the base distribution carries a concrete type
# for sampling / `eltype` queries.
@inline _baseT(p::Number) = typeof(p) <: Real ? typeof(p) : Float64
@inline _baseT(p::AbstractArray) = eltype(p) <: Real ? eltype(p) : Float64
@inline _promoteT(p, q) = promote_type(_baseT(p), _baseT(q))


include("bases.jl")
include("affine.jl")
include("constructors.jl")
