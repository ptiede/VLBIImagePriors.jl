using SparseArrays
using LinearAlgebra
using ComradeBase

export GaussMarkovRandomField

"""
    $(TYPEDEF)

A image prior based off of the first-order Gaussian Markov random field.
This is similar to the combination of L₂ and TSV regularization and is equal to

    λ TSV(I-M) + Σ⁻¹L₂(I-M) + lognorm(λ, Σ)

where λ and Σ are given below and `M` is the mean image and `lognorm(λ,Σ)` is the
log-normalization of the random field and is needed to jointly infer `I` and the
hyperparameters λ, Σ.

# Fields
$(FIELDS)

# Examples

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
`mean` and correlation `λ` and diagonal covariance `Σ`.
"""
function GaussMarkovRandomField(mean::AbstractMatrix, λ, Σ)
    cache = MarkovRandomFieldCache(eltype(mean), size(mean))
    dims = size(mean)
    return GaussMarkovRandomField(mean, λ, Σ, cache, dims)
end

"""
    GaussMarkovRandomField(mean::ComradeBase.AbstractModel, grid::ComradeBase.AbstractDims, λ, Σ [,cache]; transform=identity)

Create a `GaussMarkovRandomField` object using a ComradeBase model.

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
    GaussMarkovRandomField(mean::AbstractMatrix, λ, Σ, cache::MarkovRandomFieldCache)

Constructs a first order Gaussian Markov random field with mean image
`mean` and correlation `λ` and diagonal covariance `Σ` and the precomputed MarkovRandomFieldCache `cache`.
"""
GaussMarkovRandomField(mean::AbstractMatrix, λ, Σ, cache::MarkovRandomFieldCache) = GaussMarkovRandomField(mean, λ, Σ, cache, size(mean))

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
    x .= Dists.mean(d) .+ reshape(cQ\z, size(d))
end
