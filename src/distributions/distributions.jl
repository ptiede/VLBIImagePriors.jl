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


# ----- shared helpers ------------------------------------------------------

# Per-element parameter access. `_at(p, i)` returns `p` if `p` is a scalar
# Number (broadcast) or `p[i]` if `p` is an array. Resolved at compile time so
# the per-element loops stay monomorphic.
@inline _at(p::Number, _) = p
@inline _at(p::AbstractArray, i) = @inbounds p[i]

# `eltype` of a parameter that may be a `Number` or an `AbstractArray`.
@inline _peltype(p::Number) = typeof(p)
@inline _peltype(p::AbstractArray) = eltype(p)

# Element type used to instantiate the Std base when a user supplies one of
# the constructor functions. Reactant tracers (`<:Number` but not `<:Real`)
# fall back to `Float64` so the base distribution carries a concrete type for
# sampling and `eltype` queries.
@inline _baseT(p::Number) = typeof(p) <: Real ? typeof(p) : Float64
@inline _baseT(p::AbstractArray) = eltype(p) <: Real ? eltype(p) : Float64
@inline _promoteT(p, q) = promote_type(_baseT(p), _baseT(q))

# Sum of `log` of the affine-wrapper scale parameter. Lives here (not in any
# specific base file) because `affine.jl` is the consumer.
@inline _scale_logsum(σ::Number, n::Int) = n * log(σ)
@inline function _scale_logsum(σ::AbstractArray, _)
    tmp = log.(σ)
    return sum(tmp)
end


# ----- argument-validation helpers ---------------------------------------
# Skip the check when any input is non-`Real` (e.g. a Reactant tracer): we
# cannot branch on the value at trace time, and the caller is responsible
# for feeding in valid traced parameters anyway.

@inline _is_real_param(::Real) = true
@inline _is_real_param(::Number) = false
@inline _is_real_param(p::AbstractArray) = eltype(p) <: Real

@inline function _check_pos(name::String, val_name::String, val::Real)
    val > 0 || throw(ArgumentError("$name: $val_name must be positive (got $val)"))
    return nothing
end
@inline function _check_pos(name::String, val_name::String, val::AbstractArray{<:Real})
    all(>(0), val) || throw(ArgumentError("$name: every entry of $val_name must be positive"))
    return nothing
end
@inline _check_pos(::String, ::String, ::Any) = nothing  # traced — skip

@inline function _check_lt(name::String, a::Real, b::Real)
    a < b || throw(ArgumentError("$name: lower bound must be strictly less than upper bound"))
    return nothing
end
@inline function _check_lt(
        name::String, a::AbstractArray{<:Real}, b::AbstractArray{<:Real}
    )
    all(((ai, bi),) -> ai < bi, zip(a, b)) || throw(
        ArgumentError("$name: every lower bound must be strictly less than the corresponding upper bound")
    )
    return nothing
end
@inline function _check_lt(name::String, a::Real, b::AbstractArray{<:Real})
    all(>(a), b) || throw(
        ArgumentError("$name: every upper bound must exceed the scalar lower bound")
    )
    return nothing
end
@inline function _check_lt(name::String, a::AbstractArray{<:Real}, b::Real)
    all(<(b), a) || throw(
        ArgumentError("$name: every lower bound must be less than the scalar upper bound")
    )
    return nothing
end
@inline _check_lt(::String, ::Any, ::Any) = nothing  # traced — skip


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

## Support handling

The scalar (`N = 0`) implementations mask out-of-support inputs to `-Inf`
via a branchless `ifelse`. The vectorised array (`N >= 1`) implementations
do **not** mask: they assume the input is in support and would otherwise
return finite garbage (or `NaN` for `log` of a non-positive value). This
asymmetry is intentional — branchless masking on every element of a traced
array adds arithmetic that we'd rather not pay. Use `Dists.insupport(d, x)`
to validate inputs before calling on the array path.
"""
function unnormed_logpdf end

"""
    lognorm(d)

Returns the data-independent log-normalisation constant of `d`. Useful for
caching when the normalisation is expensive — e.g. a `StdInverseGamma` whose
shape parameter is a large array, or an `AffineDistribution` with a per-pixel
`scale` whose `sum(log, scale)` term doesn't depend on the input. For
`AffineDistribution` the `lognorm` also includes the affine-transform
Jacobian (`-sum(log, scale)`), since that too is data-independent.
"""
function lognorm end


include("std_normal.jl")
include("std_exponential.jl")
include("std_uniform.jl")
include("std_inverse_gamma.jl")
include("std_tdist.jl")
include("affine.jl")
include("constructors.jl")
