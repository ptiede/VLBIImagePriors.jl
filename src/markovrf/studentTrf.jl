using SparseArrays
using LinearAlgebra
using ComradeBase

export TDistMarkovRandomField

abstract type MarkovRandomField <: Dists.ContinuousMatrixDistribution end

"""
    $(TYPEDEF)

A image prior based off of the first-order Multivariate T distribution Markov random field.

# Fields
$(FIELDS)

# Examples

```julia
julia> mimg = zeros(6, 6) # The mean image
julia> λ, Σ = 2.0, 1.0
julia> d = TDistMarkovRandomField(mimg, λ, Σ)
julia> cache = MarkovRandomFieldCache(mimg) # now instead construct the cache
julia> d2 = TDistMarkovRandomField(mimg, λ, Σ, cache)
julia> invcov(d) ≈ invcov(d2)
true
```
"""
struct TDistMarkovRandomField{T,M<:AbstractMatrix{T},P,C,TDi} <: MarkovRandomField
    """
    The mean image of the TDistian Markov random field
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
    The student T "degrees of freedom parameter which > 1
    """
    ν::P
    """
    The Markov Random Field cache used to define the specific Markov random field class used.
    """
    cache::C
    """
    The dimensions of the image.
    """
    dims::TDi
end

Base.size(d::TDistMarkovRandomField)  = size(d.m)
Dists.mean(d::TDistMarkovRandomField) = d.m
Dists.cov(d::TDistMarkovRandomField)  = inv(Array(Dists.invcov(d)))


HC.asflat(d::TDistMarkovRandomField) = TV.as(Matrix, size(d)...)

"""
    TDistMarkovRandomField(mean::AbstractMatrix, λ, Σ)

Constructs a first order TDistian Markov random field with mean image
`mean` and correlation `λ` and diagonal covariance `Σ`.
"""
function TDistMarkovRandomField(mean::AbstractMatrix, λ, Σ, ν)
    cache = MarkovRandomFieldCache(eltype(mean), size(mean))
    dims = size(mean)
    return TDistMarkovRandomField(mean, λ, Σ, ν, cache, dims)
end

"""
    TDistMarkovRandomField(mean::ComradeBase.AbstractModel, grid::ComradeBase.AbstractDims, λ, Σ [,cache]; transform=identity)

Create a `TDistMarkovRandomField` object using a ComradeBase model.

# Arguments
 - `mean`: A ComradeBase model that will define the mean image
 - `grid`: The grid on which the image of the model will be created. This calls `ComradeBase.intensitymap`.
 - `λ`: The correlation length of the GMRF
 - `Σ`: The variance of the GMRF
 - `cache`: Optionally specify the precomputed MarkovRandomFieldCache

# Keyword Arguments
- `transform = identity`: A transform to apply to the image when creating the mean image. See the examples.

# Examples
```julia
julia> m1 = TDistMarkovRandomField(TDistian(), imagepixels(10.0, 10.0, 128, 128), 5.0, 1.0; transform=alr)
julia> cache = MarkovRandomFieldCache(TDistian(), imagepixels(10.0, 10.0, 128, 128), 5.0, 1.0; transform=alr)
julia> m2 = TDistMarkovRandomField(TDistian(), imagepixels(10.0, 10.0, 128, 128), 5.0, 1.0, cache; transform=alr)
julia> m1 == m2
true
```

"""
function TDistMarkovRandomField(mean::ComradeBase.AbstractModel, grid::ComradeBase.AbstractDims, args...; transform=identity)
    img = intensitymap(mean, grid)
    return TDistMarkovRandomField(transform(baseimage(img)), args...)
end

"""
    TDistMarkovRandomField(mean::AbstractMatrix, λ, Σ, cache::MarkovRandomFieldCache)

Constructs a first order TDistian Markov random field with mean image
`mean` and correlation `λ` and diagonal covariance `Σ` and the precomputed MarkovRandomFieldCache `cache`.
"""
TDistMarkovRandomField(mean::AbstractMatrix, λ, Σ, ν, cache::MarkovRandomFieldCache) = TDistMarkovRandomField(mean, λ, Σ, ν, cache, size(mean))

function lognorm(d::TDistMarkovRandomField)
    ν = d.ν
    N = length(d)
    det = Base.logdet(d.cache, λ, Σ)
    return log(ν/2 + 1) + log(ν*π) + det/2
end

function unnormed_logpdf(d::TDistMarkovRandomField, I::AbstractMatrix)
    (;λ, Σ, ν) = d
    ΔI = ds.m - I
    sq = sq_manoblis(d.cache, ΔI, λ, Σ)
    return -(ν/2+1)*log1p(inv(ν)*sq)
end

function Dists._rand!(rng::AbstractRNG, d::TDistMarkovRandomField, x::AbstractMatrix{<:Real})
    Q = Dists.invcov(d)
    cQ = cholesky(Q)
    z = randn(rng, length(x))
    x .= Dists.mean(d) .+ sqrt(d.ν/rand(rng, Dists.Chisq(d.ν))).*reshape(cQ\z, size(d))
end
