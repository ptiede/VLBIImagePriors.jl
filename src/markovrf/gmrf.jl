using SparseArrays
using LinearAlgebra
using ComradeBase
using Serialization

export GaussMarkovRandomField

"""
    $(TYPEDEF)

A image prior based off of the first-order zero mean Gaussian Markov random field.
This prior is similar to the combination of *total squared variation* TSV and L₂ norm, and
is given by

    (π²Σ)⁻¹ TSV(I) + (ρ^2π²Σ)L₂(I) + lognorm(ρ, Σ)

where ρ and Σ are the correlation length and variance of the random field and
`lognorm(ρ,Σ)` is the log-normalization of the random field. This normalization is needed to
jointly infer `I` and the hyperparameters ρ, Σ.


# Fields
$(FIELDS)

# Example

```julia
julia> mimg = zeros(6, 6) # The mean image
julia> ρ, Σ = 2.0, 1.0
julia> d = GaussMarkovRandomField(mimg, ρ, Σ)
julia> cache = MarkovRandomFieldCache(mimg) # now instead construct the cache
julia> d2 = GaussMarkovRandomField(mimg, ρ, Σ, cache)
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
    The variance of the random field
    """
    Σ::T
    """
    The Markov Random Field cache used to define the specific Markov random field class used.
    """
    cache::C
end


Base.size(d::GaussMarkovRandomField)  = size(d.cache)
Dists.mean(d::GaussMarkovRandomField{T}) where {T} = FillArrays.Zeros(T, size(d))
Dists.cov(d::GaussMarkovRandomField)  = inv(Array(Dists.invcov(d)))


HC.asflat(d::GaussMarkovRandomField) = TV.as(Matrix, size(d)...)

"""
    GaussMarkovRandomField(ρ, Σ, img::AbstractArray)

Constructs a first order zero-mean Gaussian Markov random field with
dimensions `size(img)`, correlation `ρ` and diagonal covariance `Σ`.
"""
function GaussMarkovRandomField(ρ::Number, Σ::Number, img::AbstractMatrix)
    cache = MarkovRandomFieldCache(eltype(img), size(img))
    return GaussMarkovRandomField(ρ, Σ, cache)
end

"""
    GaussMarkovRandomField(ρ, Σ, dims)

Constructs a first order zero-mean Gaussian Markov random field with
dimensions `size(img)`, correlation `ρ` and diagonal covariance `Σ`.
"""
function GaussMarkovRandomField(ρ::Number, Σ::Number, dims::Dims{2})
    T = promote_type(typeof(ρ), typeof(Σ))
    cache = MarkovRandomFieldCache(T, dims)
    return GaussMarkovRandomField{T, typeof(cache)}(convert(T, ρ), convert(T, Σ), cache)
end


"""
    GaussMarkovRandomField(ρ, Σ, cache::MarkovRandomFieldCache)

Constructs a first order Gaussian Markov random field using the precomputed cache `cache`.
`ρ` and diagonal covariance `Σ` and the precomputed MarkovRandomFieldCache `cache`.
If `mean` is not included then it is assume the mean is identically zero.
"""
function GaussMarkovRandomField(ρ::Number, Σ::Number, cache::MarkovRandomFieldCache)
    T = promote_type(typeof(ρ), typeof(Σ))
    GaussMarkovRandomField{T, typeof(cache)}(convert(T,ρ), convert(T,Σ), cache)
end

function lognorm(d::GaussMarkovRandomField)
    N = length(d)
    return (logdet(d.cache, d.ρ, d.Σ) - Dists.log2π*N)/2
end

function unnormed_logpdf(d::GaussMarkovRandomField, I::AbstractMatrix)
    (;ρ, Σ) = d
    return -sq_manoblis(d.cache, I, ρ, Σ)/2
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


struct MarkovTransform{TΛ, P}
    Λ::TΛ
    p::P
end

function Serialization.serialize(s::Serialization.AbstractSerializer, cache::MarkovTransform)
    Serialization.writetag(s.io, Serialization.OBJECT_TAG)
    Serialization.serialize(s, typeof(cache))
    Serialization.serialize(s, cache.Λ)
end

function Serialization.deserialize(s::AbstractSerializer, ::Type{<:MarkovTransform})
    Λ = Serialization.deserialize(s)
    p = plan_fft(Λ)
    return MarkovTransform(Λ, p)
end


function (θ::MarkovTransform)(x::AbstractArray, mean, σ, κ, ν=1)
    (;Λ, p) = θ
    T = eltype(x)
    τ = σ*κ^ν*sqrt(ν)
    rast = (@. τ*(κ^2 + Λ)^(-(ν+1)/2)*x)
    return real.(p*rast.*complex(one(T), one(T)))./sqrt(prod(size(Λ))) .+ mean
end

export standardize
function standardize(d::MarkovRandomFieldCache, ::Type{<:Dists.Normal})
    kx = fftfreq(size(d.λQ, 1))
    ky = fftfreq(size(d.λQ, 2))
    k2 = kx.*kx .+ ky'.*ky'
    return MarkovTransform(k2, plan_fft(d.λQ)), StdNormal(size(d))
end
