# StdUniform — uniform on `[0, 1]^N`. Already normalised (`lognorm = 0`).

struct StdUniform{T, N} <: Dists.ContinuousDistribution{Dists.ArrayLikeVariate{N}}
    dims::Dims{N}
end
StdUniform(dims::Dims{N}) where {N} = StdUniform{Float64, N}(dims)
StdUniform(dims::Int...) = StdUniform(dims)
StdUniform() = StdUniform{Float64, 0}(())

Base.size(d::StdUniform) = d.dims
Base.length(d::StdUniform) = prod(d.dims)
Base.eltype(::StdUniform{T}) where {T} = T


# ----- log-pdf split ------------------------------------------------------

@inline function _unnormed_kernel(::StdUniform, z, _)
    return ifelse((z >= zero(z)) & (z <= one(z)), zero(z), oftype(z, -Inf))
end
@inline _unnormed_kernel_sum(::StdUniform, z) = zero(eltype(z))

unnormed_logpdf(d::StdUniform{T, 0}, x::Number) where {T} = _unnormed_kernel(d, x, 1)
function unnormed_logpdf(d::StdUniform{T, N}, x::AbstractArray{<:Number, N}) where {T, N}
    return _unnormed_kernel_sum(d, x)
end

@inline lognorm(d::StdUniform) = zero(eltype(d))


# ----- Distributions interface --------------------------------------------

Dists.logpdf(d::StdUniform{T, 0}, x::Number) where {T} = unnormed_logpdf(d, x) + lognorm(d)
function Dists._logpdf(d::StdUniform{T, N}, x::AbstractArray{<:Number, N}) where {T, N}
    return unnormed_logpdf(d, x) + lognorm(d)
end
function Dists.logpdf(d::StdUniform{T, N}, x::AbstractArray{<:Real, N}) where {T, N}
    return unnormed_logpdf(d, x) + lognorm(d)
end
function Dists.logpdf(d::StdUniform{T, N}, x::AbstractArray{<:Number, N}) where {T, N}
    return unnormed_logpdf(d, x) + lognorm(d)
end


# ----- sampling -----------------------------------------------------------

Random.rand(rng::AbstractRNG, ::StdUniform{T, 0}) where {T} = rand(rng, T)
function Dists._rand!(
        rng::AbstractRNG, ::StdUniform{T, N}, x::AbstractArray{<:Real, N}
    ) where {T, N}
    return rand!(rng, x)
end


# ----- support / moments --------------------------------------------------

Dists.insupport(::StdUniform, x::Number) = (0 <= x <= 1)
# `<:Real` overload breaks ambiguity with Distributions' generic
# `insupport(::ContinuousUnivariateDistribution, ::Real)`.
Dists.insupport(::StdUniform, x::Real) = (0 <= x <= 1)
function Dists.insupport(d::StdUniform, x::AbstractArray)
    return size(d) == size(x) && all(xi -> 0 <= xi <= 1, x)
end

Dists.mean(::StdUniform{T, 0}) where {T} = T(0.5)
Dists.var(::StdUniform{T, 0}) where {T} = T(1) / T(12)
Dists.mean(d::StdUniform) = fill(eltype(d)(0.5), size(d))
Dists.var(d::StdUniform) = fill(eltype(d)(1) / eltype(d)(12), size(d))


# ----- cdf / quantile -----------------------------------------------------

@inline _std_cdf(::StdUniform, x) = clamp(x, zero(x), one(x))
@inline _std_quantile(::StdUniform, p) = p

Dists.cdf(d::StdUniform{T, 0}, x::Number) where {T} = _std_cdf(d, x)
Dists.quantile(d::StdUniform{T, 0}, p::Number) where {T} = _std_quantile(d, p)


# ----- transforms ---------------------------------------------------------

HC.asflat(::StdUniform{T, 0}) where {T} = TV.as(Real, 0.0, 1.0)
HC.asflat(d::StdUniform{T, N}) where {T, N} = TV.as(Array, TV.as(Real, 0.0, 1.0), size(d)...)
HC.inverse_eltype(::StdUniform{T}, y::Type) where {T} = promote_type(T, eltype(y))

HC.ascube(d::StdUniform{T, 0}) where {T} = HC.ScalarHC(d)
HC.ascube(d::StdUniform) = HC.ArrayHC(d)

function HC._step_transform(h::HC.ArrayHC{<:StdUniform}, p::AbstractVector, index)
    out = _ascube_z(h.dist, p)
    return out, index + HC.dimension(h)
end
function HC._step_inverse!(
        x::AbstractVector, index, h::HC.ArrayHC{<:StdUniform}, y::AbstractVector
    )
    x .= _ascube_p(h.dist, y)
    return index + HC.dimension(h)
end
