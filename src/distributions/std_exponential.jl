# StdExponential — rate-1 exponential of arbitrary shape.
# pdf(z) = exp(-z) for z >= 0, 0 otherwise. Already normalised (`lognorm = 0`).

struct StdExponential{T, N} <: Dists.ContinuousDistribution{Dists.ArrayLikeVariate{N}}
    dims::Dims{N}
end
StdExponential(dims::Dims{N}) where {N} = StdExponential{Float64, N}(dims)
StdExponential(dims::Int...) = StdExponential(dims)
StdExponential() = StdExponential{Float64, 0}(())

Base.size(d::StdExponential) = d.dims
Base.length(d::StdExponential) = prod(d.dims)
Base.eltype(::StdExponential{T}) where {T} = T


# ----- log-pdf split ------------------------------------------------------

@inline function _unnormed_kernel(::StdExponential, z, _)
    return ifelse(z >= zero(z), -z, oftype(z, -Inf))
end
@inline _unnormed_kernel_sum(::StdExponential, z) = -sum(z)

function unnormed_logpdf(d::StdExponential{T, 0}, x::Number) where {T}
    return _unnormed_kernel(d, x, 1)
end
function unnormed_logpdf(
        d::StdExponential{T, N}, x::AbstractArray{<:Number, N}
    ) where {T, N}
    return _unnormed_kernel_sum(d, x)
end

@inline lognorm(d::StdExponential) = zero(eltype(d))


# ----- Distributions interface --------------------------------------------

function Dists.logpdf(d::StdExponential{T, 0}, x::Number) where {T}
    return unnormed_logpdf(d, x) + lognorm(d)
end
function Dists._logpdf(d::StdExponential{T, N}, x::AbstractArray{<:Number, N}) where {T, N}
    return unnormed_logpdf(d, x) + lognorm(d)
end
function Dists.logpdf(d::StdExponential{T, N}, x::AbstractArray{<:Real, N}) where {T, N}
    return unnormed_logpdf(d, x) + lognorm(d)
end
function Dists.logpdf(d::StdExponential{T, N}, x::AbstractArray{<:Number, N}) where {T, N}
    return unnormed_logpdf(d, x) + lognorm(d)
end


# ----- sampling -----------------------------------------------------------

Random.rand(rng::AbstractRNG, ::StdExponential{T, 0}) where {T} = T(randexp(rng))
function Dists._rand!(
        rng::AbstractRNG, ::StdExponential{T, N}, x::AbstractArray{<:Real, N}
    ) where {T, N}
    return randexp!(rng, x)
end


# ----- support / moments --------------------------------------------------

Dists.insupport(::StdExponential, x::Number) = x >= 0
# `<:Real` overload breaks ambiguity with Distributions' generic
# `insupport(::ContinuousUnivariateDistribution, ::Real)`.
Dists.insupport(::StdExponential, x::Real) = x >= 0
function Dists.insupport(d::StdExponential, x::AbstractArray)
    return size(d) == size(x) && all(>=(0), x)
end

Dists.mean(::StdExponential{T, 0}) where {T} = one(T)
Dists.var(::StdExponential{T, 0}) where {T} = one(T)
Dists.mean(d::StdExponential) = fill(one(eltype(d)), size(d))
Dists.var(d::StdExponential) = fill(one(eltype(d)), size(d))


# ----- cdf / quantile -----------------------------------------------------

@inline _std_cdf(::StdExponential, x) = -expm1(-x)
@inline _std_quantile(::StdExponential, p) = -log1p(-p)

Dists.cdf(d::StdExponential{T, 0}, x::Number) where {T} = _std_cdf(d, x)
Dists.quantile(d::StdExponential{T, 0}, p::Number) where {T} = _std_quantile(d, p)


# ----- transforms ---------------------------------------------------------

HC.asflat(::StdExponential{T, 0}) where {T} = TV.asℝ₊
HC.asflat(d::StdExponential{T, N}) where {T, N} = TV.as(Array, TV.asℝ₊, size(d)...)

# Force ArrayHC for all dimensions so the round-trip uses the broadcasting
# kernel — see the comment in distributions.jl for why ScalarHC doesn't work.
HC.ascube(d::StdExponential) = HC.ArrayHC(d)
function HC._step_transform(h::HC.ArrayHC{<:StdExponential}, p::AbstractVector, index)
    out = _ascube_z(h.dist, p)
    return out, index + HC.dimension(h)
end
function HC._step_inverse!(
        x::AbstractVector, index, h::HC.ArrayHC{<:StdExponential}, y::AbstractVector
    )
    x .= _ascube_p(h.dist, y)
    return index + HC.dimension(h)
end
