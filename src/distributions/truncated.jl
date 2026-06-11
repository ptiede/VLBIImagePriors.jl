# Reactant-friendly truncated wrapper.
#
# Mirrors `Distributions.Truncated` but with the branching ternaries replaced
# by `ifelse` masks and no exception thrown on invalid bounds — caller is
# responsible for `lower <= upper`. The base distribution must provide
# Reactant-traceable `logpdf`, `cdf`, and `quantile` (the `VLBI*` family
# does).


"""
    VLBITruncated(d, lower, upper)
    VLBITruncated(d; lower=nothing, upper=nothing)

Distribution of `X | lower <= X <= upper` where `X ~ d`. Either bound may
be `nothing` for one-sided truncation (left- or right-truncation). The
normalisation constant `log P(lower <= X <= upper)` is cached at construction
so `lognorm(d)` is data-independent. Mirrors the keyword API of
`Distributions.truncated`.

Reactant-friendly: every method is branchless; the bound checks dispatch
on `Nothing` vs `Real` so the support mask compiles away when a bound is
absent. Sampling uses the inverse-CDF method (no rejection loop) so it
traces under `@jit` if `quantile(d.untruncated, ⋅)` does.
"""
struct VLBITruncated{D <: Dists.UnivariateDistribution, Tl, Tu, T} <:
    Dists.ContinuousUnivariateDistribution
    untruncated::D
    lower::Tl
    upper::Tu
    logtp::T   # log P(lower <= X <= upper)
    lcdf::T    # cdf(d.untruncated, lower) — zero when `lower === nothing`
end

# Both bounds finite. Bounds are `<:Number` (not `<:Real`) so Reactant
# tracers go through the same constructor.
function VLBITruncated(d::Dists.UnivariateDistribution, lower::Number, upper::Number)
    lcdf = Dists.cdf(d, lower)
    ucdf = Dists.cdf(d, upper)
    loglcdf = log(lcdf)
    logucdf = log(ucdf)
    # Numerically stable `log(ucdf - lcdf)` via log1p.
    logtp = logucdf + log1p(-exp(loglcdf - logucdf))
    T = promote_type(typeof(logtp), typeof(lcdf))
    return VLBITruncated{typeof(d), typeof(lower), typeof(upper), T}(
        d, lower, upper, T(logtp), T(lcdf)
    )
end
# Right-truncated only: `X <= upper`. `lcdf = 0`, `tp = ucdf`.
function VLBITruncated(d::Dists.UnivariateDistribution, ::Nothing, upper::Number)
    ucdf = Dists.cdf(d, upper)
    logtp = log(ucdf)
    T = promote_type(typeof(logtp), typeof(ucdf))
    return VLBITruncated{typeof(d), Nothing, typeof(upper), T}(
        d, nothing, upper, T(logtp), zero(T)
    )
end
# Left-truncated only: `X >= lower`. `ucdf = 1`, `tp = 1 - lcdf`.
function VLBITruncated(d::Dists.UnivariateDistribution, lower::Number, ::Nothing)
    lcdf = Dists.cdf(d, lower)
    logtp = log1p(-lcdf)
    T = promote_type(typeof(logtp), typeof(lcdf))
    return VLBITruncated{typeof(d), typeof(lower), Nothing, T}(
        d, lower, nothing, T(logtp), T(lcdf)
    )
end

function VLBITruncated(d::Dists.UnivariateDistribution; lower = nothing, upper = nothing)
    return VLBITruncated(d, lower, upper)
end


Base.minimum(d::VLBITruncated{<:Any, <:Number}) = max(d.lower, Dists.minimum(d.untruncated))
Base.minimum(d::VLBITruncated{<:Any, Nothing}) = Dists.minimum(d.untruncated)
Base.maximum(d::VLBITruncated{<:Any, <:Any, <:Number}) = min(d.upper, Dists.maximum(d.untruncated))
Base.maximum(d::VLBITruncated{<:Any, <:Any, Nothing}) = Dists.maximum(d.untruncated)
Dists.params(d::VLBITruncated) = (Dists.params(d.untruncated)..., d.lower, d.upper)


# ----- bound checks (dispatched on Nothing vs Real, monomorphic) ----------

@inline _ge_lower(::Nothing, _) = true
@inline _ge_lower(lower, x) = x >= lower
@inline _le_upper(::Nothing, _) = true
@inline _le_upper(upper, x) = x <= upper


# ----- unnormed_logpdf / lognorm split ------------------------------------
# `unnormed_logpdf` carries the data-dependent piece (the masked base
# log-density); `lognorm` is the constant `-log P(lower <= X <= upper)`.

function unnormed_logpdf(d::VLBITruncated, x::Number)
    in_supp = _ge_lower(d.lower, x) & _le_upper(d.upper, x)
    base_lpdf = Dists.logpdf(d.untruncated, x)
    return ifelse(in_supp, base_lpdf, oftype(base_lpdf, -Inf))
end

@inline lognorm(d::VLBITruncated) = -d.logtp

Dists.logpdf(d::VLBITruncated, x::Number) = unnormed_logpdf(d, x) + lognorm(d)
Dists.logpdf(d::VLBITruncated, x::Real) = unnormed_logpdf(d, x) + lognorm(d)


# ----- cdf / quantile -----------------------------------------------------
# `cdf_trunc(x) = clamp((cdf_base(x) - lcdf) / tp, 0, 1)` — `clamp` handles
# the out-of-support cases (negative below `lower`, > 1 above `upper`). For
# one-sided truncation `lcdf = 0` (right-only) or `tp = 1 - lcdf` is set
# correctly at construction, so the same formula works.

function Dists.cdf(d::VLBITruncated, x::Number)
    raw = (Dists.cdf(d.untruncated, x) - d.lcdf) * exp(-d.logtp)
    return clamp(raw, zero(raw), one(raw))
end

function Dists.quantile(d::VLBITruncated, p::Number)
    return Dists.quantile(d.untruncated, d.lcdf + p * exp(d.logtp))
end


# ----- sampling -----------------------------------------------------------
# Inverse-CDF method: U ~ Uniform(0, 1), X = quantile_trunc(U). No
# rejection loop — traces straight through.

Random.rand(rng::AbstractRNG, d::VLBITruncated) = Dists.quantile(d, rand(rng))


# ----- support ------------------------------------------------------------

function Dists.insupport(d::VLBITruncated, x::Number)
    return _ge_lower(d.lower, x) & _le_upper(d.upper, x)
end
function Dists.insupport(d::VLBITruncated, x::Real)
    return _ge_lower(d.lower, x) & _le_upper(d.upper, x)
end


# ----- transforms ---------------------------------------------------------

# The flat transform must cover the SUPPORT — the truncation bounds intersected with the
# base support (`minimum`/`maximum` above) — never the explicit bounds alone. Building it
# from the bounds mapped `VLBITruncated(VLBIExponential(0.1); upper = 1)` onto (-∞, 1),
# exposing a reachable logpdf = -Inf region in flat space (in practice: a negative
# background-flux optimum and a frozen NUTS chain).
function HC.asflat(d::VLBITruncated)
    lo = Base.minimum(d)
    hi = Base.maximum(d)
    lb = isfinite(lo) ? lo : -TV.∞
    ub = isfinite(hi) ? hi : TV.∞
    return TV.as(Real, lb, ub)
end

HC.inverse_eltype(b::VLBITruncated, y::Type) = HC.inverse_eltype(b.untruncated, y)
# `HC.ascube` not overridden: VLBITruncated is always a UnivariateDistribution
# so HC's default routes it to `ScalarHC`. The scalar `Dists.cdf` /
# `Dists.quantile` defined above feed HC's stock `_step_inverse!` /
# `_step_transform` for ScalarHC directly.
