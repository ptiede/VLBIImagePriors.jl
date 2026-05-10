# StdTDist — Student's t with degrees of freedom `ν`, mean 0, scale 1.
# pdf(z; ν) = Γ((ν+1)/2) / (sqrt(ν π) Γ(ν/2)) · (1 + z²/ν)^(-(ν+1)/2)
# `ν` may be a scalar or a per-element array of the same shape as the
# distribution.

struct StdTDist{T, Tν, N} <: Dists.ContinuousDistribution{Dists.ArrayLikeVariate{N}}
    ν::Tν
    dims::Dims{N}
end
StdTDist(ν::Number) = StdTDist{typeof(ν), typeof(ν), 0}(ν, ())
StdTDist(ν::Number, dims::Dims{N}) where {N} = StdTDist{typeof(ν), typeof(ν), N}(ν, dims)
StdTDist(ν::Number, dims::Int...) = StdTDist(ν, dims)
function StdTDist(ν::AbstractArray{<:Number, N}) where {N}
    return StdTDist{eltype(ν), typeof(ν), N}(ν, size(ν))
end

Base.size(d::StdTDist) = d.dims
Base.length(d::StdTDist) = prod(d.dims)
Base.eltype(::StdTDist{T}) where {T} = T


# ----- log-pdf split ------------------------------------------------------
# `lognorm` for an array `ν` involves `loggamma((ν+1)/2)` per element — that's
# the expensive piece a caller will want to cache.

@inline function _t_lognorm_per_elem(ν)
    return loggamma((ν + 1) / 2) - loggamma(ν / 2) - oftype(ν, log(π)) / 2 - log(ν) / 2
end

@inline function _unnormed_kernel(d::StdTDist, z)
    ν = d.ν
    return -((ν + one(ν)) / 2) * log1p(z * z / ν)
end
@inline function _unnormed_kernel_sum(d::StdTDist, z)
    ν = d.ν
    sq = abs2.(z)
    log_terms = log1p.(sq ./ ν)
    return -sum(((ν .+ 1) ./ 2) .* log_terms)
end

unnormed_logpdf(d::StdTDist{T, <:Number, 0}, x::Number) where {T} = _unnormed_kernel(d, x)
function unnormed_logpdf(
        d::StdTDist{T, Tν, N}, x::AbstractArray{<:Number, N}
    ) where {T, Tν, N}
    return _unnormed_kernel_sum(d, x)
end

@inline lognorm(d::StdTDist{T, <:Number}) where {T} = length(d) * _t_lognorm_per_elem(d.ν)
@inline lognorm(d::StdTDist{T, <:AbstractArray}) where {T} = sum(_t_lognorm_per_elem, d.ν)


# ----- Distributions interface --------------------------------------------

function Dists.logpdf(d::StdTDist{T, <:Number, 0}, x::Number) where {T}
    return unnormed_logpdf(d, x) + lognorm(d)
end
function Dists._logpdf(
        d::StdTDist{T, Tν, N}, x::AbstractArray{<:Number, N}
    ) where {T, Tν, N}
    return unnormed_logpdf(d, x) + lognorm(d)
end
function Dists.logpdf(
        d::StdTDist{T, Tν, N}, x::AbstractArray{<:Real, N}
    ) where {T, Tν, N}
    return unnormed_logpdf(d, x) + lognorm(d)
end
function Dists.logpdf(
        d::StdTDist{T, Tν, N}, x::AbstractArray{<:Number, N}
    ) where {T, Tν, N}
    return unnormed_logpdf(d, x) + lognorm(d)
end


# ----- sampling -----------------------------------------------------------

# `T = Z / sqrt(W/ν)` with `Z ~ N(0, 1)` and `W ~ χ²(ν) = 2·Gamma(ν/2, 1)`.
@inline function _rand_tdist(rng::AbstractRNG, ν::Number)
    z = randn(rng)
    g = _rand_gamma(rng, ν / 2)
    return z / sqrt(2 * g / ν)
end

function Random.rand(rng::AbstractRNG, d::StdTDist{T, <:Number, 0}) where {T}
    return _rand_tdist(rng, d.ν)
end
function Dists._rand!(
        rng::AbstractRNG, d::StdTDist{T, <:Number, N}, x::AbstractArray{<:Real, N}
    ) where {T, N}
    ν = d.ν
    @trace for i in eachindex(x)
        x[i] = _rand_tdist(rng, ν)
    end
    return x
end
function Dists._rand!(
        rng::AbstractRNG, d::StdTDist{T, <:AbstractArray, N}, x::AbstractArray{<:Real, N}
    ) where {T, N}
    @trace for i in eachindex(x)
        x[i] = _rand_tdist(rng, d.ν[i])
    end
    return x
end


# ----- support / moments --------------------------------------------------

Dists.insupport(::StdTDist, ::Number) = true
# `<:Real` overload breaks ambiguity with Distributions' generic
# `insupport(::ContinuousUnivariateDistribution, ::Number)`.
Dists.insupport(::StdTDist, ::Real) = true
Dists.insupport(d::StdTDist, x::AbstractArray) = size(d) == size(x)

function Dists.mean(d::StdTDist{T, <:Real, 0}) where {T}
    return d.ν > 1 ? zero(T) : T(NaN)
end
function Dists.var(d::StdTDist{T, <:Real, 0}) where {T}
    return d.ν > 2 ? T(d.ν / (d.ν - 2)) : T(Inf)
end
@inline _t_elemmean(ν::Number, T) = ν > 1 ? zero(T) : T(NaN)
@inline _t_elemvar(ν::Number, T) = ν > 2 ? T(ν / (ν - 2)) : T(Inf)
function Dists.mean(d::StdTDist{T, <:Real, N}) where {T, N}
    return fill(_t_elemmean(d.ν, T), size(d))
end
function Dists.var(d::StdTDist{T, <:Real, N}) where {T, N}
    return fill(_t_elemvar(d.ν, T), size(d))
end
function Dists.mean(d::StdTDist{T, <:AbstractArray, N}) where {T, N}
    return _t_elemmean.(d.ν, T)
end
function Dists.var(d::StdTDist{T, <:AbstractArray, N}) where {T, N}
    return _t_elemvar.(d.ν, T)
end


# ----- cdf / quantile -----------------------------------------------------
# Built on the regularised incomplete beta function. With `a = ν/2`, `b = 1/2`,
# and `arg = ν / (ν + z²)`:
#   cdf(z) = 1 - I(arg; a, b) / 2   for z >= 0
#   cdf(z) =     I(arg; a, b) / 2   for z <  0
# Quantile inverts the same identity.
# Element-wise kernels take `ν` directly so they broadcast against either a
# scalar or a per-element parameter array — used by the ArrayHC ascube path
# below for the per-element-ν specialisation.

@inline function _t_elem_cdf(ν, x)
    a = ν / 2
    b = oftype(ν, 0.5)
    arg = ν / (ν + x * x)
    P_arg = first(SpecialFunctions.beta_inc(a, b, arg))
    return ifelse(x >= zero(x), one(x) - P_arg / 2, P_arg / 2)
end
@inline function _t_elem_quantile(ν, p)
    a = ν / 2
    b = oftype(ν, 0.5)
    p_in = ifelse(p < oftype(p, 0.5), 2 * p, 2 * (one(p) - p))
    q_in = one(p) - p_in
    arg = first(SpecialFunctions.beta_inc_inv(a, b, p_in, q_in))
    z_abs = sqrt(ν * (one(ν) / arg - one(ν)))
    return ifelse(p < oftype(p, 0.5), -z_abs, z_abs)
end

@inline _std_cdf(d::StdTDist, x) = _t_elem_cdf(d.ν, x)
@inline _std_quantile(d::StdTDist, p) = _t_elem_quantile(d.ν, p)

function Dists.cdf(d::StdTDist{T, <:Number, 0}, x::Number) where {T}
    return _std_cdf(d, x)
end
function Dists.quantile(d::StdTDist{T, <:Number, 0}, p::Number) where {T}
    return _std_quantile(d, p)
end


# ----- transforms ---------------------------------------------------------

HC.asflat(::StdTDist{T, <:Number, 0}) where {T} = TV.asℝ
HC.asflat(d::StdTDist{T, <:Any, N}) where {T, N} = TV.as(Array, TV.asℝ, size(d)...)

HC.ascube(d::StdTDist) = HC.ArrayHC(d)
HC.inverse_eltype(::StdTDist{T}, y::Type) where {T} = promote_type(T, eltype(y))


function HC._step_transform(h::HC.ArrayHC{<:StdTDist}, p::AbstractVector, index)
    out = _ascube_z(h.dist, p)
    return out, index + HC.dimension(h)
end
function HC._step_inverse!(
        x::AbstractVector, index, h::HC.ArrayHC{<:StdTDist}, y::AbstractVector
    )
    x .= _ascube_p(h.dist, y)
    return index + HC.dimension(h)
end

# Per-element-ν override; see std_inverse_gamma.jl for the rationale.
@inline _ascube_z(b::StdTDist{T, <:AbstractArray}, p) where {T} =
    _t_elem_quantile.(vec(b.ν), p)
@inline _ascube_p(b::StdTDist{T, <:AbstractArray}, z) where {T} =
    _t_elem_cdf.(vec(b.ν), z)
