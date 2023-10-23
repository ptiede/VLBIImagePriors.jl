using SparseArrays
using LinearAlgebra
using ComradeBase
using Serialization

export GaussMarkovRandomField

"""
    $(TYPEDEF)

A image prior based off of the first-order Gaussian Markov random field with mean image `m`.
This prior is similar to the combination of *total squared variation* TSV and L₂ norm, and
is given by

    (λ²π²Σ)⁻¹ TSV(I-M) + (π²Σ)L₂(I-M) + lognorm(λ, Σ)

where λ and Σ are the inverse correlation length and variance of the random field and
`lognorm(λ,Σ)` is the log-normalization of the random field. This normalization is needed to
jointly infer `I` and the hyperparameters λ, Σ.


# Fields
$(FIELDS)

# Example

```julia
julia> mimg = zeros(6, 6) # The mean image
julia> λ, Σ = 2.0, 1.0
julia> d = GaussMarkovRandomField(mimg, λ, Σ)
julia> cache = MarkovRandomFieldCache(mimg) # now instead construct the cache
julia> d2 = GaussMarkovRandomField(mimg, λ, Σ, cache)
julia> invcov(d) ≈ invcov(d2)
true
```
"""
struct GaussMarkovRandomField{T,M<:AbstractMatrix{T},P,C,TDi} <: MarkovRandomField
    """
    The mean image of the Gaussian Markov random field
    """
    m::M
    """
    The inverse correlation length of the random field.
    """
    λ::P
    """
    The variance of the random field
    """
    Σ::P
    """
    The Markov Random Field cache used to define the specific Markov random field class used.
    """
    cache::C
    """
    The dimensions of the image.
    """
    dims::TDi
end

Base.size(d::GaussMarkovRandomField)  = size(d.m)
Dists.mean(d::GaussMarkovRandomField) = d.m
Dists.cov(d::GaussMarkovRandomField)  = inv(Array(Dists.invcov(d)))


HC.asflat(d::GaussMarkovRandomField) = TV.as(Matrix, size(d)...)

"""
    GaussMarkovRandomField(mean::AbstractMatrix, λ, Σ)

Constructs a first order Gaussian Markov random field with mean image
`mean` and inverse correlation `λ` and diagonal covariance `Σ`.
"""
function GaussMarkovRandomField(mean::AbstractMatrix, λ, Σ)
    cache = MarkovRandomFieldCache(eltype(mean), size(mean))
    dims = size(mean)
    return GaussMarkovRandomField(mean, λ, Σ, cache, dims)
end

function GaussMarkovRandomField(λ, Σ, dims)
    T = promote_type(typeof(λ), typeof(Σ))
    return GaussMarkovRandomField(zeros(T, dims), convert(T, λ), convert(T, Σ))
end

"""
    GaussMarkovRandomField(mean::ComradeBase.AbstractModel, grid::ComradeBase.AbstractDims, λ, Σ [,cache]; transform=identity)

Create a `GaussMarkovRandomField` object using a ComradeBase model.

# Arguments
 - `mean`: A ComradeBase model that will define the mean image
 - `grid`: The grid on which the image of the model will be created. This calls `ComradeBase.intensitymap`.
 - `λ`: The inverse correlation length of the GMRF
 - `Σ`: The variance of the GMRF
 - `cache`: Optionally specify the precomputed MarkovRandomFieldCache

# Keyword Arguments
- `transform = identity`: A transform to apply to the image when creating the mean image. See the examples.

# Examples
```julia
julia> m1 = GaussMarkovRandomField(Gaussian(), imagepixels(10.0, 10.0, 128, 128), 5.0, 1.0; transform=alr)
julia> cache = MarkovRandomFieldCache(Gaussian(), imagepixels(10.0, 10.0, 128, 128), 5.0, 1.0; transform=alr)
julia> m2 = GaussMarkovRandomField(Gaussian(), imagepixels(10.0, 10.0, 128, 128), 5.0, 1.0, cache; transform=alr)
julia> m1 == m2
true
```

"""
function GaussMarkovRandomField(mean::ComradeBase.AbstractModel, grid::ComradeBase.AbstractDims, args...; transform=identity)
    img = intensitymap(mean, grid)
    return GaussMarkovRandomField(transform(baseimage(img)), args...)
end

"""
    GaussMarkovRandomField(λ, Σ, cache::MarkovRandomFieldCache)
    GaussMarkovRandomField(mean::AbstractMatrix, λ, Σ, cache::MarkovRandomFieldCache)

Constructs a first order Gaussian Markov random field with mean image
`mean` and inverse correlation `λ` and diagonal covariance `Σ` and the precomputed MarkovRandomFieldCache `cache`.
If `mean` is not included then it is assume the mean is identically zero.
"""
GaussMarkovRandomField(mean::AbstractMatrix, λ::Number, Σ::Number, cache::MarkovRandomFieldCache) = GaussMarkovRandomField(mean, λ, Σ, cache, size(mean))
GaussMarkovRandomField(λ::Number, Σ::Number, cache::MarkovRandomFieldCache) = GaussMarkovRandomField(zeros(promote_type(λ, Σ), size(cache.λQ)))

function lognorm(d::GaussMarkovRandomField)
    N = length(d)
    return (logdet(d.cache, d.λ, d.Σ) - Dists.log2π*N)/2
end

function unnormed_logpdf(d::GaussMarkovRandomField, I::AbstractMatrix)
    (;λ, Σ) = d
    ΔI = d.m - I
    return -sq_manoblis(d.cache, ΔI, λ, Σ)/2
end

function Dists._rand!(rng::AbstractRNG, d::GaussMarkovRandomField, x::AbstractMatrix{<:Real})
    Q = Dists.invcov(d)
    cQ = cholesky(Q)
    z = randn(rng, length(x))
    x .= Dists.mean(d) .+ reshape(cQ.PtL'\z, size(d))
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
    return MarkovTransform(k2, plan_fft(d.λQ)), StdNormal(size(d.λQ))
end
