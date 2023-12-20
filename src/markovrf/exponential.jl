export ExpMarkovRandomField

"""
    $(TYPEDEF)

A image prior based off of the first-order zero mean unit variance Exponential Markov random field.
This prior is similar to the combination of *total variation* TSV and L₁ norm and is given by

    √TV(I)² + inv(ρ²)L₁(I)²) + lognorm(ρ)

where ρ is the correlation length the random field and
`lognorm(ρ)` is the log-normalization of the random field. This normalization is needed to
jointly infer `I` and the hyperparameters ρ.


# Fields
$(FIELDS)

# Example

```julia
julia> ρ = 10.0
julia> d = ExpMarkovRandomField(ρ, (32, 32))
julia> cache = MarkovRandomFieldCache(Float64, (32, 32)) # now instead construct the cache
julia> d2 = ExpMarkovRandomField(ρ, cache)
julia> invcov(d) ≈ invcov(d2)
true
```
"""
struct ExpMarkovRandomField{T<:Number,C} <: MarkovRandomField
    """
    The correlation length of the random field.
    """
    ρ::T
    """
    The Markov Random Field cache used to define the specific Markov random field class used.
    """
    cache::C
end

(c::ConditionalMarkov{<:Dists.Exponential})(ρ) = ExpMarkovRandomField(ρ, c.cache)


Base.size(d::ExpMarkovRandomField)  = size(d.cache)
Dists.mean(d::ExpMarkovRandomField{T}) where {T} = FillArrays.Zeros(T, size(d))
Dists.cov(d::ExpMarkovRandomField)  = inv(Array(Dists.invcov(d)))

HC.asflat(d::ExpMarkovRandomField) = TV.as(Matrix, size(d)...)

"""
    ExpMarkovRandomField(ρ, img::AbstractArray)

Constructs a first order zero-mean Exponential Markov random field with
dimensions `size(img)`, correlation `ρ` and unit covariance.
"""
function ExpMarkovRandomField(ρ::Number, img::AbstractMatrix)
    cache = MarkovRandomFieldCache(eltype(img), size(img))
    return ExpMarkovRandomField(ρ, cache)
end

"""
    ExpMarkovRandomField(ρ, dims)

Constructs a first order zero-mean unit variance Exponential Markov random field with
dimensions `dims`, correlation `ρ`.
"""
function ExpMarkovRandomField(ρ::Number, dims::Dims{2})
    cache = MarkovRandomFieldCache(typeof(ρ), dims)
    return ExpMarkovRandomField(ρ, cache)
end


"""
    ExpMarkovRandomField(ρ, cache::MarkovRandomFieldCache)

Constructs a first order zero-mean and unit variance Exponential Markov random field using the
precomputed cache `cache`.
"""
function ExpMarkovRandomField(ρ::Number, cache::MarkovRandomFieldCache)
    ExpMarkovRandomField{typeof(ρ), typeof(cache)}(ρ, cache)
end

function lognorm(d::ExpMarkovRandomField)
    N = length(d)
    return (logdet(d.cache, d.ρ) - Dists.log2π*N)/2
end

function unnormed_logpdf(d::ExpMarkovRandomField, I::AbstractMatrix)
    (;ρ) = d
    return -2*sqrt(sq_manoblis(d.cache, I, ρ)*length(d))
end

function Dists._rand!(rng::AbstractRNG, d::ExpMarkovRandomField, x::AbstractMatrix{<:Real})
    Q = Dists.invcov(d)
    cQ = cholesky(Q)
    z = randn(rng, length(x))
    R = rand(rng, Dists.Chisq(2*length(d)))
    x .= R.*reshape(cQ.PtL'\z, size(d))./norm(z)/sqrt(4*length(d))
end
