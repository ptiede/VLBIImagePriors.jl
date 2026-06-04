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
