# Generic affine wrapper that turns any Std base distribution into a
# location-scale distribution: if `z ~ base`, then
# `loc + scale .* z ~ AffineDistribution(loc, scale, base)`.
#
# `scale` is one of:
#   * `Number` — broadcast across the support of `base`
#   * `AbstractArray` matching `size(base)` — element-wise (diagonal) scale
#   * `AbstractMatrix` (1D base only) — full linear operator: `y = loc + A z`
#     and `unnormed_logpdf(d, x) = unnormed_logpdf(d.base, A \ (x - loc))`,
#     `lognorm` picks up `-logabsdet(A)`. Anything that supports `*`, `\`,
#     and `LinearAlgebra.logabsdet` works (e.g. a `LinearMap`).
#
# No parameter type bound is `<:Real`, so traced numbers from Reactant are
# accepted.


"""
    AffineDistribution(loc, scale, base)

Represents the distribution of `loc + scale .* z` where `z ~ base`. `loc` and
`scale` may each be a `Number` (broadcast across the support of `base`) or an
`AbstractArray` of the same dimensionality as `base`. Reactant-friendly: no
parameter type bound is `<:Real`, so traced numbers are accepted.
"""
struct AffineDistribution{D <: Dists.Distribution, N, Tloc, Tscale} <:
    Dists.ContinuousDistribution{Dists.ArrayLikeVariate{N}}
    base::D
    loc::Tloc
    scale::Tscale
end
function AffineDistribution(
        loc, scale, base::Dists.Distribution{Dists.ArrayLikeVariate{N}}
    ) where {N}
    return AffineDistribution{typeof(base), N, typeof(loc), typeof(scale)}(base, loc, scale)
end


function Base.show(io::IO, d::AffineDistribution)
    print(io, "AffineDistribution(")
    print(io, "base=", nameof(typeof(d.base)))
    print(io, ", loc::", _summarize_param(d.loc))
    print(io, ", scale::", _summarize_param(d.scale))
    sz = size(d)
    isempty(sz) || print(io, ", size=", sz)
    return print(io, ")")
end
@inline _summarize_param(p::Number) = string(p)
@inline _summarize_param(p::AbstractArray) = string(eltype(p), size(p))

Base.size(d::AffineDistribution) = size(d.base)
Base.length(d::AffineDistribution) = length(d.base)
function Base.eltype(d::AffineDistribution)
    return promote_type(eltype(d.loc), eltype(d.scale), eltype(d.base))
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
        d::AffineDistribution{B, 0, Tloc, Tscale}, x::Number
    ) where {B, Tloc, Tscale}
    z = (x - d.loc) / d.scale
    return unnormed_logpdf(d.base, z)
end

# array (N >= 1) — vectorised so it traces under Reactant
function unnormed_logpdf(
        d::AffineDistribution{B, N, Tloc, Tscale}, x::AbstractArray{<:Number, N}
    ) where {B, N, Tloc, Tscale}
    z = (x .- d.loc) ./ d.scale
    return unnormed_logpdf(d.base, z)
end

@inline function lognorm(d::AffineDistribution{<:Any, <:Any, <:Any, <:Number})
    return lognorm(d.base) - length(d) * log(d.scale)
end
@inline function lognorm(d::AffineDistribution{<:Any, <:Any, <:Any, <:AbstractArray})
    return lognorm(d.base) - sum(log, d.scale)
end

# Matrix / linear-operator scale (1D base): `y = loc + A z`. The base
# `unnormed_logpdf` sees `z = A \ (y - loc)` and the affine Jacobian is
# `logabsdet(A)`, replacing the per-element `sum(log, scale)` term.
function unnormed_logpdf(
        d::AffineDistribution{B, 1, Tloc, <:AbstractMatrix}, x::AbstractVector{<:Number}
    ) where {B, Tloc}
    z = d.scale \ (x .- d.loc)
    return unnormed_logpdf(d.base, z)
end
@inline function lognorm(d::AffineDistribution{<:Any, 1, <:Any, <:AbstractMatrix})
    return lognorm(d.base) - first(logabsdet(d.scale))
end


# ----- logpdf -------------------------------------------------------------
# Composes the `unnormed_logpdf` + `lognorm` split above.

@inline _affine_array_logpdf(d::AffineDistribution, x) = unnormed_logpdf(d, x) + lognorm(d)

# Scalar
function Dists.logpdf(
        d::AffineDistribution{B, 0, Tloc, Tscale}, x::Number
    ) where {B, Tloc, Tscale}
    return unnormed_logpdf(d, x) + lognorm(d)
end

# Three-method pattern, identical to `bases.jl`. The `<:Real` override is
# required to break ambiguity with Distributions' top-level
# `logpdf(::Distribution{ArrayLikeVariate{N}}, ::AbstractArray{<:Real, M})`
# at `Distributions/.../common.jl:261` — Julia errors on `Matrix{Float64}`
# inputs without it (verified empirically).
function Dists._logpdf(
        d::AffineDistribution{B, N, Tloc, Tscale}, x::AbstractArray{<:Number, N}
    ) where {B, N, Tloc, Tscale}
    return _affine_array_logpdf(d, x)
end
function Dists.logpdf(
        d::AffineDistribution{B, N, Tloc, Tscale}, x::AbstractArray{<:Real, N}
    ) where {B, N, Tloc, Tscale}
    return _affine_array_logpdf(d, x)
end
function Dists.logpdf(
        d::AffineDistribution{B, N, Tloc, Tscale}, x::AbstractArray{<:Number, N}
    ) where {B, N, Tloc, Tscale}
    return _affine_array_logpdf(d, x)
end


# ----- sampling -----------------------------------------------------------

function Random.rand(
        rng::AbstractRNG, d::AffineDistribution{B, 0, <:Number, <:Number}
    ) where {B}
    return d.loc + d.scale * rand(rng, d.base)
end

@inline function _affine_rand!(rng, d::AffineDistribution, x)
    z = similar(x, eltype(d.base))
    Dists._rand!(rng, d.base, z)
    @. x = d.loc + d.scale * z
    return x
end
# Matrix-scale sampling: `y = loc + A z` (matrix-vector product).
@inline function _affine_rand!(
        rng, d::AffineDistribution{<:Any, 1, <:Any, <:AbstractMatrix}, x::AbstractVector
    )
    z = similar(x, eltype(d.base))
    Dists._rand!(rng, d.base, z)
    mul!(x, d.scale, z)
    x .+= d.loc
    return x
end
# Sampling is CPU-only — under Reactant tracing, `rand!` has no clean
# semantics so we don't override `<:Number` here. Gradients flow through
# `logpdf`, not sampling.
function Dists._rand!(
        rng::AbstractRNG, d::AffineDistribution{B, N, Tloc, Tscale},
        x::AbstractArray{<:Real, N}
    ) where {B, N, Tloc, Tscale}
    return _affine_rand!(rng, d, x)
end


# ----- support ------------------------------------------------------------

function Dists.insupport(
        d::AffineDistribution{B, 0, Tloc, Tscale}, x::Number
    ) where {B, Tloc, Tscale}
    return Dists.insupport(d.base, (x - d.loc) / d.scale)
end
# Break ambiguity with `Distributions.insupport(::ContinuousUnivariateDistribution, ::Real)`
# — `AffineDistribution{...,0}` is a univariate continuous distribution.
function Dists.insupport(
        d::AffineDistribution{B, 0, Tloc, Tscale}, x::Real
    ) where {B, Tloc, Tscale}
    return Dists.insupport(d.base, (x - d.loc) / d.scale)
end
function Dists.insupport(d::AffineDistribution, x::AbstractArray)
    size(x) == size(d) || return false
    z = (x .- d.loc) ./ d.scale
    return all(zi -> Dists.insupport(d.base, zi), z)
end
function Dists.insupport(
        d::AffineDistribution{<:Any, 1, <:Any, <:AbstractMatrix}, x::AbstractVector
    )
    length(x) == length(d) || return false
    z = d.scale \ (x .- d.loc)
    return all(zi -> Dists.insupport(d.base, zi), z)
end


# ----- moments ------------------------------------------------------------

# Moments. Broadcasting forms work for both scalar (N = 0) and array
# (N >= 1) — the scalar case collapses via scalar arithmetic, the array
# case lifts to the right shape via the per-element base mean/var.
Dists.mean(d::AffineDistribution) = d.loc .+ d.scale .* Dists.mean(d.base)
Dists.var(d::AffineDistribution) = d.scale .^ 2 .* Dists.var(d.base)
Dists.std(d::AffineDistribution) = sqrt.(Dists.var(d))

# Matrix-scale moments: `y = loc + A z`, so `mean(y) = loc + A mean(z)` and
# `cov(y) = A cov(z) A'`. `var(y) = diag(cov(y))`.
function Dists.mean(d::AffineDistribution{<:Any, 1, <:Any, <:AbstractMatrix})
    return d.loc .+ d.scale * Dists.mean(d.base)
end
function Dists.cov(d::AffineDistribution{<:Any, 1, <:Any, <:AbstractMatrix})
    return d.scale * Dists.cov(d.base) * d.scale'
end
function Dists.var(d::AffineDistribution{<:Any, 1, <:Any, <:AbstractMatrix})
    return diag(Dists.cov(d))
end


# ----- params -------------------------------------------------------------
# The user-visible parameter tuple in the order each user-facing constructor
# accepts, so introspecting tools (e.g. for serialisation or pretty-printing
# a hierarchical model) recover the original argument list.

Dists.params(d::AffineDistribution{<:StdNormal}) = (d.loc, d.scale)
Dists.params(d::AffineDistribution{<:StdExponential}) = (d.scale,)
Dists.params(d::AffineDistribution{<:StdUniform}) = (d.loc, d.loc .+ d.scale)
Dists.params(d::AffineDistribution{<:StdInverseGamma}) = (d.base.α, d.scale)
Dists.params(d::AffineDistribution{<:StdTDist}) = (d.base.ν, d.loc, d.scale)


# ----- cdf / quantile -----------------------------------------------------
# `y = loc + scale * z`, so `cdf_y(y) = cdf_z((y - loc) / scale)` and
# `quantile_y(p) = loc + scale * quantile_z(p)`. The base kernels are
# branchless arithmetic, so this whole stack traces under Reactant for
# non-`Real` parameters.

function Dists.cdf(
        d::AffineDistribution{B, 0, Tloc, Tscale}, x::Number
    ) where {B, Tloc, Tscale}
    z = (x - d.loc) / d.scale
    return _std_cdf(d.base, z)
end
function Dists.quantile(
        d::AffineDistribution{B, 0, Tloc, Tscale}, p::Number
    ) where {B, Tloc, Tscale}
    return d.loc + d.scale * _std_quantile(d.base, p)
end


# ----- asflat -------------------------------------------------------------
# Dispatched on the base type. For non-`Real` params (Reactant traced) we
# fall back to unconstrained transforms and let `logpdf` enforce support.
HC.asflat(::AffineDistribution{<:StdNormal, 0, <:Number, <:Number}) = TV.asℝ
HC.asflat(::AffineDistribution{<:StdTDist, 0, <:Number, <:Number}) = TV.asℝ
HC.asflat(::AffineDistribution{<:StdExponential, 0, <:Number, <:Number}) = TV.asℝ₊
HC.asflat(::AffineDistribution{<:StdInverseGamma, 0, <:Number, <:Number}) = TV.asℝ₊
function HC.asflat(d::AffineDistribution{<:StdUniform, 0, <:Real, <:Real})
    return TV.as(Real, d.loc, d.loc + d.scale)
end
HC.asflat(::AffineDistribution{<:StdUniform, 0, <:Number, <:Number}) = TV.asℝ

function HC.asflat(d::AffineDistribution{<:StdNormal, N}) where {N}
    return TV.as(Array, TV.asℝ, size(d)...)
end
function HC.asflat(d::AffineDistribution{<:StdTDist, N}) where {N}
    return TV.as(Array, TV.asℝ, size(d)...)
end
function HC.asflat(d::AffineDistribution{<:StdExponential, N}) where {N}
    return TV.as(Array, TV.asℝ₊, size(d)...)
end
function HC.asflat(d::AffineDistribution{<:StdInverseGamma, N}) where {N}
    return TV.as(Array, TV.asℝ₊, size(d)...)
end
function HC.asflat(d::AffineDistribution{<:StdUniform, N, <:Real, <:Real}) where {N}
    return TV.as(
        Array, TV.as(Real, d.loc, d.loc + d.scale), size(d)...
    )
end
function HC.asflat(d::AffineDistribution{<:StdUniform, N}) where {N}
    return TV.as(Array, TV.asℝ, size(d)...)
end


# ----- ascube -------------------------------------------------------------
# Force ArrayHC for every shape (0-d through N-d), mirroring the per-base
# ascube overrides. Without this, `AffineDistribution{B, 0, ...}` would fall
# through to HC's `ascube(::UnivariateDistribution) = ScalarHC(d)` whose
# `_step_inverse!` only accepts a scalar — but `transform(::ScalarHC, [u])`
# returns a Vector via `Distributions.quantile`'s broadcast, breaking the
# round-trip.
#
# The matrix-scale variant (`<:AbstractMatrix` scale, 1D base) is excluded:
# it's a linear-operator transform with no element-wise quantile, so it
# falls through to HC's default and a clear error.

HC.ascube(d::AffineDistribution{<:Any, <:Any, <:Any, <:Number}) = HC.ArrayHC(d)
HC.ascube(d::AffineDistribution{<:Any, <:Any, <:Any, <:AbstractArray}) = HC.ArrayHC(d)

function HC._step_transform(
        h::HC.ArrayHC{<:AffineDistribution}, p::AbstractVector, index
    )
    d = h.dist
    n = HC.dimension(h)
    pslice = view(p, index:(index + n - 1))
    z = _ascube_z(d.base, pslice)
    out = _flat_or_scalar(d.loc) .+ _flat_or_scalar(d.scale) .* z
    return out, index + n
end

function HC._step_inverse!(
        x::AbstractVector, index, h::HC.ArrayHC{<:AffineDistribution}, y
    )
    d = h.dist
    n = HC.dimension(h)
    z = (vec(y) .- _flat_or_scalar(d.loc)) ./ _flat_or_scalar(d.scale)
    @views x[index:(index + n - 1)] .= _ascube_p(d.base, z)
    return index + n
end


# ----- product_distribution lifting --------------------------------------
# Mirrors the `DiagonalVonMises` pattern: an `AbstractVector` of scalar
# `AffineDistribution`s with the same Std base folds into one 1D
# `AffineDistribution` with concatenated per-element parameters, preserving
# the affine structure (and the cached `lognorm` split).

function Dists.product_distribution(
        dists::AbstractVector{<:AffineDistribution{<:StdNormal, 0}}
    )
    locs = [d.loc for d in dists]
    scales = [d.scale for d in dists]
    T = promote_type(eltype(locs), eltype(scales))
    return AffineDistribution(locs, scales, StdNormal{T, 1}((length(dists),)))
end

function Dists.product_distribution(
        dists::AbstractVector{<:AffineDistribution{<:StdExponential, 0}}
    )
    scales = [d.scale for d in dists]
    T = promote_type(eltype(scales))
    return AffineDistribution(
        zero(eltype(scales)), scales, StdExponential{T, 1}((length(dists),))
    )
end

function Dists.product_distribution(
        dists::AbstractVector{<:AffineDistribution{<:StdUniform, 0}}
    )
    locs = [d.loc for d in dists]
    scales = [d.scale for d in dists]
    T = promote_type(eltype(locs), eltype(scales))
    return AffineDistribution(locs, scales, StdUniform{T, 1}((length(dists),)))
end

function Dists.product_distribution(
        dists::AbstractVector{<:AffineDistribution{<:StdInverseGamma, 0}}
    )
    scales = [d.scale for d in dists]
    αs = [d.base.α for d in dists]
    T = promote_type(eltype(αs), eltype(scales))
    base = StdInverseGamma{T, typeof(αs), 1}(αs, (length(dists),))
    return AffineDistribution(zero(eltype(scales)), scales, base)
end

function Dists.product_distribution(
        dists::AbstractVector{<:AffineDistribution{<:StdTDist, 0}}
    )
    locs = [d.loc for d in dists]
    scales = [d.scale for d in dists]
    νs = [d.base.ν for d in dists]
    T = promote_type(eltype(νs), eltype(locs), eltype(scales))
    base = StdTDist{T, typeof(νs), 1}(νs, (length(dists),))
    return AffineDistribution(locs, scales, base)
end
