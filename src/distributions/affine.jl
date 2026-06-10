# `AffineDistribution(loc, scale, base)` — `loc + scale .* z` for `z ~ base` — is
# now a thin constructor over ProbabilityTransports' `PushforwardDistribution`,
# reusing the shared `ScaleShift`/`AffineTransform` change-of-variables machinery
# (no hand-rolled inverse/log-det). `scale`:
#   * `Number` / `AbstractArray` (matching `size(base)`) → element-wise `ScaleShift`
#   * `AbstractMatrix` (1-D base) → full linear operator `loc + A z` via `AffineTransform`
# Reactant-friendly: no `<:Real` bounds, so traced numbers are accepted.

# Element-wise by default; a `Matrix` scale is a linear operator ONLY for a 1-D
# base (a vector). A `Matrix` scale over a 2-D image base is per-pixel (same shape
# as the base), so it stays a `ScaleShift`.
_affine_map(loc, scale, base) = ScaleShift(loc, scale)
_affine_map(loc, scale::AbstractMatrix, base::Dists.Distribution{Dists.ArrayLikeVariate{1}}) =
    AffineTransform(loc, scale)

"""
    AffineDistribution(loc, scale, base)

The distribution of `loc + scale .* z` (or `loc + A z` for a 1-D base with a matrix
`scale = A`) where `z ~ base`. A `PushforwardDistribution` over the affine map.
"""
AffineDistribution(loc, scale, base) = PushforwardDistribution(_affine_map(loc, scale, base), base)


# ----- asflat (TVFlat transport): centered parametrization ----------------
# PT's generic pushforward flat node wraps `ScaleShift` around the base's flat
# transform, i.e. `transform(t, y) = loc .+ scale .* (base-flat)(y)` — a
# *non-centered* map that applies loc/scale in the transport (and so allocates an
# extra intermediate array). The legacy HypercubeTransform behavior was *centered*:
# the unconstrained coordinates ARE the parameter, loc/scale enter only through
# `logpdf`, and the flat transport is just the base support's TV transform. We
# restore that here for the element-wise (`ScaleShift`) families — identical
# allocations and sampling geometry to the old API. A matrix-scale `AffineTransform`
# (a genuine linear operator, e.g. MvNormal whitening) keeps PT's pushforward node.

# loc/scale-independent support blocks (the affine map lives in `logpdf`).
_flat_block(::StdNormal) = TV.asℝ
_flat_block(::StdTDist) = TV.asℝ
_flat_block(::StdExponential) = TV.asℝ₊
_flat_block(::StdInverseGamma) = TV.asℝ₊

const _CenteredBase = Union{StdNormal, StdTDist, StdExponential, StdInverseGamma}

transport_node(d::PushforwardDistribution{<:ScaleShift, <:_CenteredBase, 0}, ::TVFlat) =
    _flat_block(d.base)
transport_node(d::PushforwardDistribution{<:ScaleShift, <:_CenteredBase, N}, ::TVFlat) where {N} =
    TV.as(Array, _flat_block(d.base), size(d)...)

# StdUniform: the support is the bounded interval [loc, loc+scale]; the flat
# transform maps ℝ onto it. Real bounds only — traced/array bounds fall back to ℝ
# and let `logpdf` enforce support (matching the old `asflat` dispatch).
_uniform_flat(lo::Real, hi::Real) = TV.as(Real, lo, hi)
_uniform_flat(::Any, ::Any) = TV.asℝ

transport_node(d::PushforwardDistribution{<:ScaleShift, <:StdUniform, 0}, ::TVFlat) =
    _uniform_flat(d.f.μ, d.f.μ + d.f.s)
transport_node(d::PushforwardDistribution{<:ScaleShift, <:StdUniform, N}, ::TVFlat) where {N} =
    TV.as(Array, _uniform_flat(d.f.μ, d.f.μ + d.f.s), size(d)...)


# ----- params -------------------------------------------------------------
# The user-visible parameter tuple in the order each user-facing VLBI* constructor
# accepts (loc/scale read off the `ScaleShift` map; shape params off the base).

Dists.params(d::PushforwardDistribution{<:ScaleShift, <:StdNormal}) = (d.f.μ, d.f.s)
Dists.params(d::PushforwardDistribution{<:ScaleShift, <:StdExponential}) = (d.f.s,)
Dists.params(d::PushforwardDistribution{<:ScaleShift, <:StdUniform}) = (d.f.μ, d.f.μ .+ d.f.s)
Dists.params(d::PushforwardDistribution{<:ScaleShift, <:StdInverseGamma}) = (d.base.α, d.f.s)
Dists.params(d::PushforwardDistribution{<:ScaleShift, <:StdTDist}) = (d.base.ν, d.f.μ, d.f.s)


# ----- product_distribution lifting ---------------------------------------
# A vector of scalar affine pushforwards over the same Std base folds into one 1-D
# pushforward with concatenated per-element parameters (preserving the cached
# `lognorm` split), mirroring the `DiagonalVonMises` pattern.

const _AffinePF{B} = PushforwardDistribution{<:ScaleShift, B, 0}

function Dists.product_distribution(dists::AbstractVector{<:_AffinePF{<:StdNormal}})
    locs = [d.f.μ for d in dists]
    scales = [d.f.s for d in dists]
    T = promote_type(eltype(locs), eltype(scales))
    return AffineDistribution(locs, scales, StdNormal{T, 1}((length(dists),)))
end

function Dists.product_distribution(dists::AbstractVector{<:_AffinePF{<:StdExponential}})
    scales = [d.f.s for d in dists]
    T = promote_type(eltype(scales))
    return AffineDistribution(zero(eltype(scales)), scales, StdExponential{T, 1}((length(dists),)))
end

function Dists.product_distribution(dists::AbstractVector{<:_AffinePF{<:StdUniform}})
    locs = [d.f.μ for d in dists]
    scales = [d.f.s for d in dists]
    T = promote_type(eltype(locs), eltype(scales))
    return AffineDistribution(locs, scales, StdUniform{T, 1}((length(dists),)))
end

function Dists.product_distribution(dists::AbstractVector{<:_AffinePF{<:StdInverseGamma}})
    scales = [d.f.s for d in dists]
    αs = [d.base.α for d in dists]
    T = promote_type(eltype(αs), eltype(scales))
    base = StdInverseGamma{T, typeof(αs), 1}(αs, (length(dists),))
    return AffineDistribution(zero(eltype(scales)), scales, base)
end

function Dists.product_distribution(dists::AbstractVector{<:_AffinePF{<:StdTDist}})
    locs = [d.f.μ for d in dists]
    scales = [d.f.s for d in dists]
    νs = [d.base.ν for d in dists]
    T = promote_type(eltype(νs), eltype(locs), eltype(scales))
    base = StdTDist{T, typeof(νs), 1}(νs, (length(dists),))
    return AffineDistribution(locs, scales, base)
end
