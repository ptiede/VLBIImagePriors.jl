# Generic affine wrapper that turns any Std base distribution into a
# location-scale distribution: if `z ~ base`, then
# `loc + scale .* z ~ AffineDistribution(loc, scale, base)`.
#
# `loc` and `scale` may each be a `Number` (broadcast across the support of
# `base`) or an `AbstractArray` of the same dimensionality as `base`. No
# parameter type bound is `<:Real`, so traced numbers from Reactant are
# accepted. The implementation reuses the per-element kernels and reductions
# defined in `bases.jl` so the only thing this file adds is the affine
# transform on top.


"""
    AffineDistribution(loc, scale, base)

Represents the distribution of `loc + scale .* z` where `z ~ base`. `loc` and
`scale` may each be a `Number` (broadcast across the support of `base`) or an
`AbstractArray` of the same dimensionality as `base`. Reactant-friendly: no
parameter type bound is `<:Real`, so traced numbers are accepted.
"""
struct AffineDistribution{Tloc, Tscale, D <: Dists.Distribution, N} <:
    Dists.ContinuousDistribution{Dists.ArrayLikeVariate{N}}
    loc::Tloc
    scale::Tscale
    base::D
end
function AffineDistribution(
        loc, scale, base::Dists.Distribution{Dists.ArrayLikeVariate{N}}
    ) where {N}
    return AffineDistribution{typeof(loc), typeof(scale), typeof(base), N}(loc, scale, base)
end

Base.size(d::AffineDistribution) = size(d.base)
Base.length(d::AffineDistribution) = length(d.base)
function Base.eltype(d::AffineDistribution)
    return promote_type(_peltype(d.loc), _peltype(d.scale), eltype(d.base))
end


# ----- unnormed_logpdf / lognorm split ------------------------------------
# `log p_y(y) = log p_z((y - loc) / scale) - log|scale|`, which we split into:
#   `unnormed_logpdf(d, y) = unnormed_logpdf(d.base, (y - loc) / scale)`
#   `lognorm(d)            = lognorm(d.base) - sum_log(scale)`
# Caching `lognorm(d)` is the whole point — when the only thing changing
# between `logpdf` calls is `y`, the user can compute it once. The scale
# Jacobian goes into `lognorm` because it's data-independent.

# scalar (N = 0)
function unnormed_logpdf(
        d::AffineDistribution{Tloc, Tscale, B, 0}, x::Number
    ) where {Tloc, Tscale, B}
    z = (x - d.loc) / d.scale
    return unnormed_logpdf(d.base, z)
end

# array (N >= 1) — vectorised so it traces under Reactant
function unnormed_logpdf(
        d::AffineDistribution{Tloc, Tscale, B, N}, x::AbstractArray{<:Number, N}
    ) where {Tloc, Tscale, B, N}
    z = (x .- d.loc) ./ d.scale
    return unnormed_logpdf(d.base, z)
end

@inline function lognorm(d::AffineDistribution)
    return lognorm(d.base) - _scale_logsum(d.scale, length(d))
end


# ----- logpdf -------------------------------------------------------------
# Composes the `unnormed_logpdf` + `lognorm` split above.

@inline _affine_array_logpdf(d::AffineDistribution, x) = unnormed_logpdf(d, x) + lognorm(d)

# Scalar
function Dists.logpdf(
        d::AffineDistribution{Tloc, Tscale, B, 0}, x::Number
    ) where {Tloc, Tscale, B}
    return unnormed_logpdf(d, x) + lognorm(d)
end

# Array — three-method pattern matches `bases.jl` to break ambiguity with
# Distributions' fallback `logpdf(::Distribution{ArrayLikeVariate{N}}, ::AbstractArray{<:Real, N})`.
function Dists._logpdf(
        d::AffineDistribution{Tloc, Tscale, B, N}, x::AbstractArray{<:Number, N}
    ) where {Tloc, Tscale, B, N}
    return _affine_array_logpdf(d, x)
end
function Dists.logpdf(
        d::AffineDistribution{Tloc, Tscale, B, N}, x::AbstractArray{<:Real, N}
    ) where {Tloc, Tscale, B, N}
    @argcheck size(x) == size(d) "input/distribution size mismatch"
    return _affine_array_logpdf(d, x)
end
function Dists.logpdf(
        d::AffineDistribution{Tloc, Tscale, B, N}, x::AbstractArray{<:Number, N}
    ) where {Tloc, Tscale, B, N}
    @argcheck size(x) == size(d) "input/distribution size mismatch"
    return _affine_array_logpdf(d, x)
end


# ----- sampling -----------------------------------------------------------

function Random.rand(
        rng::AbstractRNG, d::AffineDistribution{<:Number, <:Number, B, 0}
    ) where {B}
    return d.loc + d.scale * rand(rng, d.base)
end

@inline function _affine_rand!(rng, d::AffineDistribution, x)
    z = similar(x, eltype(d.base))
    Dists._rand!(rng, d.base, z)
    @inbounds for i in eachindex(x)
        x[i] = _at(d.loc, i) + _at(d.scale, i) * z[i]
    end
    return x
end
function Dists._rand!(
        rng::AbstractRNG, d::AffineDistribution{Tloc, Tscale, B, N},
        x::AbstractArray{<:Real, N}
    ) where {Tloc, Tscale, B, N}
    return _affine_rand!(rng, d, x)
end
function Dists._rand!(
        rng::AbstractRNG, d::AffineDistribution{Tloc, Tscale, B, N},
        x::AbstractArray{<:Number, N}
    ) where {Tloc, Tscale, B, N}
    return _affine_rand!(rng, d, x)
end


# ----- support ------------------------------------------------------------

function Dists.insupport(
        d::AffineDistribution{Tloc, Tscale, B, 0}, x::Number
    ) where {Tloc, Tscale, B}
    return Dists.insupport(d.base, (x - d.loc) / d.scale)
end
function Dists.insupport(d::AffineDistribution, x::AbstractArray)
    size(x) == size(d) || return false
    @inbounds for i in eachindex(x)
        zi = (x[i] - _at(d.loc, i)) / _at(d.scale, i)
        Dists.insupport(d.base, zi) || return false
    end
    return true
end


# ----- moments ------------------------------------------------------------

function Dists.mean(d::AffineDistribution{<:Number, <:Number, B, 0}) where {B}
    return d.loc + d.scale * Dists.mean(d.base)
end
function Dists.var(d::AffineDistribution{<:Number, <:Number, B, 0}) where {B}
    return d.scale^2 * Dists.var(d.base)
end
Dists.std(d::AffineDistribution{<:Number, <:Number, B, 0}) where {B} = sqrt(Dists.var(d))


# ----- cdf / quantile -----------------------------------------------------
# `y = loc + scale * z`, so `cdf_y(y) = cdf_z((y - loc) / scale)` and
# `quantile_y(p) = loc + scale * quantile_z(p)`. The base kernels are
# branchless arithmetic, so this whole stack traces under Reactant for
# non-`Real` parameters.

function Dists.cdf(
        d::AffineDistribution{Tloc, Tscale, B, 0}, x::Number
    ) where {Tloc, Tscale, B}
    z = (x - d.loc) / d.scale
    return _std_cdf(d.base, z)
end
function Dists.quantile(
        d::AffineDistribution{Tloc, Tscale, B, 0}, p::Number
    ) where {Tloc, Tscale, B}
    return d.loc + d.scale * _std_quantile(d.base, p)
end


# ----- asflat -------------------------------------------------------------
# Dispatched on the base type. For non-`Real` params (Reactant traced) we
# fall back to unconstrained transforms and let `logpdf` enforce support.

HC.asflat(::AffineDistribution{<:Number, <:Number, <:StdNormal, 0}) = TV.asℝ
HC.asflat(::AffineDistribution{<:Number, <:Number, <:StdExponential, 0}) = TV.asℝ₊
HC.asflat(::AffineDistribution{<:Number, <:Number, <:StdInverseGamma, 0}) = TV.asℝ₊
function HC.asflat(d::AffineDistribution{<:Real, <:Real, <:StdUniform, 0})
    return TV.as(Real, Float64(d.loc), Float64(d.loc) + Float64(d.scale))
end
HC.asflat(::AffineDistribution{<:Number, <:Number, <:StdUniform, 0}) = TV.asℝ

function HC.asflat(d::AffineDistribution{<:Any, <:Any, <:StdNormal, N}) where {N}
    return TV.as(Array, TV.asℝ, size(d)...)
end
function HC.asflat(d::AffineDistribution{<:Any, <:Any, <:StdExponential, N}) where {N}
    return TV.as(Array, TV.asℝ₊, size(d)...)
end
function HC.asflat(d::AffineDistribution{<:Any, <:Any, <:StdInverseGamma, N}) where {N}
    return TV.as(Array, TV.asℝ₊, size(d)...)
end
function HC.asflat(d::AffineDistribution{<:Real, <:Real, <:StdUniform, N}) where {N}
    return TV.as(
        Array, TV.as(Real, Float64(d.loc), Float64(d.loc) + Float64(d.scale)), size(d)...
    )
end
function HC.asflat(d::AffineDistribution{<:Any, <:Any, <:StdUniform, N}) where {N}
    return TV.as(Array, TV.asℝ, size(d)...)
end
