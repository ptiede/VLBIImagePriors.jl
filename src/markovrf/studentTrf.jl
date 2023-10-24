using SparseArrays
using LinearAlgebra
using ComradeBase

export TDistMarkovRandomField

"""
    $(TYPEDEF)

A image prior based off of the first-order Multivariate T distribution Markov random field.

# Fields
$(FIELDS)

# Examples

```julia
julia> mimg = zeros(6, 6) # The mean image
julia> ρ, Σ = 2.0, 1.0
julia> d = TDistMarkovRandomField(mimg, ρ, Σ)
julia> cache = MarkovRandomFieldCache(mimg) # now instead construct the cache
julia> d2 = TDistMarkovRandomField(mimg, ρ, Σ, cache)
julia> invcov(d) ≈ invcov(d2)
true
```
"""
struct TDistMarkovRandomField{T<:Number,C} <: MarkovRandomField
    """
    The correlation length of the random field.
    """
    ρ::T
    """
    The variance of the random field
    """
    Σ::T
    """
    The student T "degrees of freedom parameter which ≥ 1 for a proper prior
    """
    ν::T
    """
    The Markov Random Field cache used to define the specific Markov random field class used.
    """
    cache::C
end

Base.size(d::TDistMarkovRandomField)  = size(d.cache)
Dists.mean(d::TDistMarkovRandomField{T}) where {T} = FillArrays.Zeros(T, size(d))
Dists.cov(d::TDistMarkovRandomField)  = inv(Array(Dists.invcov(d)))


HC.asflat(d::TDistMarkovRandomField) = TV.as(Matrix, size(d)...)

"""
    TDistMarkovRandomField(ρ, Σ, img::AbstractArray)

Constructs a first order TDist Markov random field with mean image
`mean` and correlation `ρ` and diagonal covariance `Σ`.
"""
function TDistMarkovRandomField(ρ::Number, Σ::Number, ν::Number, img::AbstractMatrix)
    cache = MarkovRandomFieldCache(eltype(img), size(img))
    return TDistMarkovRandomField(ρ, Σ, ν, cache)
end

"""
    TDistMarkovRandomField(ρ, Σ, cache::MarkovRandomFieldCache)

Constructs a first order TDist Markov random field with zero mean ,correlation `ρ`,
diagonal covariance `Σ`, and the precomputed MarkovRandomFieldCache `cache`.
"""
function TDistMarkovRandomField(ρ::Number, Σ::Number, ν::Number, cache::MarkovRandomFieldCache)
    T = promote_type(typeof(ρ), typeof(Σ), typeof(ν))
    return TDistMarkovRandomField{T, typeof(cache)}(convert(T,ρ), convert(T,Σ), convert(T,ν), cache)
end

function lognorm(d::TDistMarkovRandomField)
    ν = d.ν
    N = length(d)
    det = logdet(d.cache, d.ρ, d.Σ)
    return loggamma((ν+N)/2) - loggamma(ν/2) - N/2*log(ν*π) + det/2
end

function unnormed_logpdf(d::TDistMarkovRandomField, I::AbstractMatrix)
    (;ρ, Σ, ν) = d
    sq = sq_manoblis(d.cache, I, ρ, Σ)
    return -((ν+length(I))/2)*log1p(inv(ν)*sq)
end

function Dists._rand!(rng::AbstractRNG, d::TDistMarkovRandomField, x::AbstractMatrix{<:Real})
    Q = Dists.invcov(d)
    cQ = cholesky(Q)
    z = randn(rng, length(x))
    x .= sqrt(d.ν/rand(rng, Dists.Chisq(d.ν))).*reshape(cQ.PtL'\z, size(d))
end
