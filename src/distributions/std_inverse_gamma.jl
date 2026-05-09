# StdInverseGamma — inverse-gamma with shape `α` and scale 1.
# pdf(z; α) = z^(-α-1) exp(-1/z) / Γ(α) for z > 0.
# `α` may be a scalar (broadcast across the support) or an array of the same
# shape as the distribution; the per-element loops use `_at` to dispatch.

struct StdInverseGamma{T, Tα, N} <: Dists.ContinuousDistribution{Dists.ArrayLikeVariate{N}}
    α::Tα
    dims::Dims{N}
end
StdInverseGamma(α::Number) = StdInverseGamma{typeof(α), typeof(α), 0}(α, ())
function StdInverseGamma(α::Number, dims::Dims{N}) where {N}
    return StdInverseGamma{typeof(α), typeof(α), N}(α, dims)
end
StdInverseGamma(α::Number, dims::Int...) = StdInverseGamma(α, dims)
function StdInverseGamma(α::AbstractArray{<:Number, N}) where {N}
    return StdInverseGamma{eltype(α), typeof(α), N}(α, size(α))
end

Base.size(d::StdInverseGamma) = d.dims
Base.length(d::StdInverseGamma) = prod(d.dims)
Base.eltype(::StdInverseGamma{T}) where {T} = T


# ----- log-pdf split ------------------------------------------------------
# `loggamma(α)` is the expensive piece for an array `α`; folding it into
# `lognorm` lets a caller cache it across many `logpdf` evaluations.

@inline _ig_loggamma_sum(α::Number, n::Int) = n * loggamma(α)
@inline function _ig_loggamma_sum(α::AbstractArray, _)
    tmp = loggamma.(α)
    return sum(tmp)
end

@inline function _unnormed_kernel(d::StdInverseGamma, z, i)
    αi = _at(d.α, i)
    zsafe = ifelse(z > zero(z), z, oftype(z, 1))
    val = -(αi + one(αi)) * log(zsafe) - inv(zsafe)
    return ifelse(z > zero(z), val, oftype(z, -Inf))
end
@inline function _unnormed_kernel_sum(d::StdInverseGamma, z)
    α = d.α
    log_z = log.(z)
    inv_z = inv.(z)
    return -sum((α .+ 1) .* log_z) - sum(inv_z)
end

function unnormed_logpdf(d::StdInverseGamma{T, <:Number, 0}, x::Number) where {T}
    return _unnormed_kernel(d, x, 1)
end
function unnormed_logpdf(
        d::StdInverseGamma{T, Tα, N}, x::AbstractArray{<:Number, N}
    ) where {T, Tα, N}
    return _unnormed_kernel_sum(d, x)
end

@inline lognorm(d::StdInverseGamma) = -_ig_loggamma_sum(d.α, length(d))


# ----- Distributions interface --------------------------------------------

function Dists.logpdf(d::StdInverseGamma{T, <:Number, 0}, x::Number) where {T}
    return unnormed_logpdf(d, x) + lognorm(d)
end
function Dists._logpdf(
        d::StdInverseGamma{T, Tα, N}, x::AbstractArray{<:Number, N}
    ) where {T, Tα, N}
    return unnormed_logpdf(d, x) + lognorm(d)
end
function Dists.logpdf(
        d::StdInverseGamma{T, Tα, N}, x::AbstractArray{<:Real, N}
    ) where {T, Tα, N}
    @argcheck size(x) == size(d) "input/distribution size mismatch"
    return unnormed_logpdf(d, x) + lognorm(d)
end
function Dists.logpdf(
        d::StdInverseGamma{T, Tα, N}, x::AbstractArray{<:Number, N}
    ) where {T, Tα, N}
    @argcheck size(x) == size(d) "input/distribution size mismatch"
    return unnormed_logpdf(d, x) + lognorm(d)
end


# ----- sampling -----------------------------------------------------------

function Random.rand(rng::AbstractRNG, d::StdInverseGamma{T, <:Number, 0}) where {T}
    return T(rand(rng, Dists.InverseGamma(Float64(d.α), 1.0)))
end
function Dists._rand!(
        rng::AbstractRNG, d::StdInverseGamma{T, <:Number, N}, x::AbstractArray{<:Real, N}
    ) where {T, N}
    rd = Dists.InverseGamma(Float64(d.α), 1.0)
    @inbounds for i in eachindex(x)
        x[i] = rand(rng, rd)
    end
    return x
end
function Dists._rand!(
        rng::AbstractRNG, d::StdInverseGamma{T, <:AbstractArray, N},
        x::AbstractArray{<:Real, N}
    ) where {T, N}
    @inbounds for i in eachindex(x)
        x[i] = rand(rng, Dists.InverseGamma(Float64(d.α[i]), 1.0))
    end
    return x
end


# ----- support / moments --------------------------------------------------

Dists.insupport(::StdInverseGamma, x::Number) = x > 0
# `<:Real` overload breaks ambiguity with Distributions' generic
# `insupport(::ContinuousUnivariateDistribution, ::Real)`.
Dists.insupport(::StdInverseGamma, x::Real) = x > 0
function Dists.insupport(d::StdInverseGamma, x::AbstractArray)
    return size(d) == size(x) && all(>(0), x)
end

function Dists.mean(d::StdInverseGamma{T, <:Real, 0}) where {T}
    return d.α > 1 ? T(1 / (d.α - 1)) : T(Inf)
end
function Dists.var(d::StdInverseGamma{T, <:Real, 0}) where {T}
    return d.α > 2 ? T(1 / ((d.α - 1)^2 * (d.α - 2))) : T(Inf)
end
@inline _ig_elemmean(α::Real, T) = α > 1 ? T(1 / (α - 1)) : T(Inf)
@inline _ig_elemvar(α::Real, T) = α > 2 ? T(1 / ((α - 1)^2 * (α - 2))) : T(Inf)
function Dists.mean(d::StdInverseGamma{T, <:Real, N}) where {T, N}
    return fill(_ig_elemmean(d.α, T), size(d))
end
function Dists.var(d::StdInverseGamma{T, <:Real, N}) where {T, N}
    return fill(_ig_elemvar(d.α, T), size(d))
end
function Dists.mean(d::StdInverseGamma{T, <:AbstractArray, N}) where {T, N}
    return _ig_elemmean.(d.α, T)
end
function Dists.var(d::StdInverseGamma{T, <:AbstractArray, N}) where {T, N}
    return _ig_elemvar.(d.α, T)
end


# ----- cdf / quantile -----------------------------------------------------
# StdInverseGamma(α): cdf(x) = Q(α, 1/x), where Q is the regularised upper
# incomplete gamma. SpecialFunctions provides both directions.

@inline function _std_cdf(d::StdInverseGamma, x)
    α = _at(d.α, 1)
    return last(SpecialFunctions.gamma_inc(α, inv(x), 0))
end
@inline function _std_quantile(d::StdInverseGamma, p)
    α = _at(d.α, 1)
    return inv(SpecialFunctions.gamma_inc_inv(α, one(p) - p, p))
end

function Dists.cdf(d::StdInverseGamma{T, <:Number, 0}, x::Number) where {T}
    return _std_cdf(d, x)
end
function Dists.quantile(d::StdInverseGamma{T, <:Number, 0}, p::Number) where {T}
    return _std_quantile(d, p)
end


# ----- transforms ---------------------------------------------------------

HC.asflat(::StdInverseGamma{T, <:Number, 0}) where {T} = TV.asℝ₊
HC.asflat(d::StdInverseGamma{T, <:Any, N}) where {T, N} = TV.as(Array, TV.asℝ₊, size(d)...)
