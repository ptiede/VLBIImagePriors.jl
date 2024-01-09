using SparseArrays
using LinearAlgebra
using ComradeBase
using Serialization

export GaussMarkovRandomField, ConditionalMarkov

"""
    $(TYPEDEF)

A image prior based off of the first-order zero mean unit variance Gaussian Markov random field.
This prior is similar to the combination of *total squared variation* TSV and L₂ norm, and
is given by

    TSV(I) + (ρ²π²)L₂(I) + lognorm(ρ)

where ρ is the correlation length the random field and
`lognorm(ρ)` is the log-normalization of the random field. This normalization is needed to
jointly infer `I` and the hyperparameters ρ.


# Fields
$(FIELDS)

# Example

```julia
julia> ρ = 10.0
julia> d = GaussMarkovRandomField(ρ, (32, 32))
julia> cache = MarkovRandomFieldCache(Float64, (32, 32)) # now instead construct the cache
julia> d2 = GaussMarkovRandomField(ρ, cache)
julia> invcov(d) ≈ invcov(d2)
true
```
"""
struct GaussMarkovRandomField{T<:Number,C} <: MarkovRandomField
    """
    The correlation length of the random field.
    """
    ρ::T
    """
    The Markov Random Field cache used to define the specific Markov random field class used.
    """
    cache::C
end


Base.size(d::GaussMarkovRandomField)  = size(d.cache)
Dists.mean(d::GaussMarkovRandomField{T}) where {T} = FillArrays.Zeros(T, size(d))
Dists.cov(d::GaussMarkovRandomField)  = inv(Array(Dists.invcov(d)))

(c::ConditionalMarkov{<:Dists.Normal})(ρ)     = GaussMarkovRandomField(ρ, c.cache)


HC.asflat(d::GaussMarkovRandomField) = TV.as(Matrix, size(d)...)

"""
    GaussMarkovRandomField(ρ, img::AbstractArray; order::Integer=1)

Constructs a `order`ᵗʰ order  Gaussian Markov random field with
dimensions `size(img)`, correlation `ρ` and unit covariance.

The `order` parameter controls the smoothness of the field with higher orders being smoother.
We recommend sticking with either `order=1,2`. Noting that `order=1` is equivalent to the
usual TSV and L₂ regularization from RML imaging.
"""
function GaussMarkovRandomField(ρ::Number, img::AbstractMatrix; order::Integer=1)
    cache = MarkovRandomFieldCache(eltype(img), size(img); order)
    return GaussMarkovRandomField(ρ, cache)
end

"""
    GaussMarkovRandomField(ρ, dims)

Constructs a `order`ᵗʰ order Gaussian Markov random field with
dimensions `size(img)`, correlation `ρ` and unit covariance.

The `order` parameter controls the smoothness of the field with higher orders being smoother.
We recommend sticking with either `order=1,2`. Noting that `order=1` is equivalent to the
usual TSV and L₂ regularization from RML imaging.

"""
function GaussMarkovRandomField(ρ::Number, dims::Dims{2}; order::Integer=1)
    cache = MarkovRandomFieldCache(typeof(ρ), dims; order)
    return GaussMarkovRandomField(ρ, cache)
end


"""
    GaussMarkovRandomField(ρ, cache::MarkovRandomFieldCache)

Constructs a unit variance Gaussian Markov random field using the
precomputed Markov Random Field graph cache `cache`.
"""
function GaussMarkovRandomField(ρ::Number, cache::MarkovRandomFieldCache)
    GaussMarkovRandomField{typeof(ρ), typeof(cache)}(ρ, cache)
end

function lognorm(d::GaussMarkovRandomField)
    N = length(d)
    return (logdet(d.cache, d.ρ) - Dists.log2π*N)/2
end

function unnormed_logpdf(d::GaussMarkovRandomField, I::AbstractMatrix)
    (;ρ) = d
    return -sq_manoblis(d.cache, I, ρ)/2
end

function Dists._rand!(rng::AbstractRNG, d::GaussMarkovRandomField, x::AbstractMatrix{<:Real})
    Q = Dists.invcov(d)
    cQ = cholesky(Q)
    z = randn(rng, length(x))
    x .= reshape(cQ.PtL'\z, size(d))
end


struct StdNormal{T, N} <: Dists.ContinuousDistribution{Dists.ArrayLikeVariate{N}}
    dims::Dims{N}
end

StdNormal(d::Dims{N}) where {N} = StdNormal{Float64, N}(d)

Base.size(d::StdNormal) = d.dims
Base.length(d::StdNormal) = prod(d.dims)
Base.eltype(::StdNormal{T}) where {T} = T
Dists.insupport(::StdNormal, x::AbstractVector) = true

HC.asflat(d::StdNormal) = TV.as(Array, size(d)...)
Dists.mean(d::StdNormal) = zeros(size(d))
Dists.cov(d::StdNormal)  = Diagonal(prod(size(d)))


function Dists._logpdf(d::StdNormal{T, N}, x::AbstractArray{T, N}) where {T<:Real, N}
    return __logpdf(d, x)
end
Dists._logpdf(d::StdNormal{T, 2}, x::AbstractMatrix{T}) where {T<:Real} = __logpdf(d, x)


__logpdf(d::StdNormal, x) = -sum(abs2, x)/2 - prod(d.dims)*Dists.log2π/2


function Dists._rand!(rng::AbstractRNG, ::StdNormal{T, N}, x::AbstractArray{T, N}) where {T<: Real, N}
    return randn!(rng, x)
end
