export ExpMarkovRandomField, EMRF

"""
    $(TYPEDEF)

A image prior based off of the zero mean unit variance Exponential Markov random field.
The order of the Markov random field is specified

# Fields
$(FIELDS)

# Example

```julia
julia> ρ = 10.0
julia> d = ExpMarkovRandomField(ρ, (32, 32))
julia> cache = MarkovRandomFieldGraph(Float64, (32, 32)) # now instead construct the cache
julia> d2 = ExpMarkovRandomField(ρ, cache)
julia> scalematrix(d) ≈ scalematrix(d2)
true
```
"""
struct ExpMarkovRandomField{T <: Number, C} <: MarkovRandomField
    """
    The correlation length of the random field.
    """
    ρ::T
    """
    The Markov Random Field graph cache used to define the specific Markov random field class used.
    """
    graph::C
end

"""
    Alias for `ExpMarkovRandomField`
"""
const EMRF = ExpMarkovRandomField


(c::ConditionalMarkov{<:EMRF})(ρ) = ExpMarkovRandomField(ρ, c.cache)


Dists.mean(d::ExpMarkovRandomField{T}) where {T} = FillArrays.Zeros(T, size(d))


"""
    ExpMarkovRandomField(ρ, img::AbstractArray; order::Integer=1)

Constructs a `order`ᵗʰ order  Exponential Markov random field with
dimensions `size(img)`, correlation `ρ` and unit covariance.

The `order` parameter controls the smoothness of the field with higher orders being smoother.
We recommend sticking with either `order=1,2`. For more information about the
impact of the order see [`MarkovRandomFieldGraph`](@ref).
"""
function ExpMarkovRandomField(ρ::Number, img::AbstractMatrix; order = 1)
    cache = MarkovRandomFieldGraph(eltype(img), size(img); order)
    return ExpMarkovRandomField(ρ, cache)
end

"""
    ExpMarkovRandomField(ρ, dims; order=1)

Constructs a first order zero-mean unit variance Exponential Markov random field with
dimensions `dims`, correlation `ρ`.

The `order` parameter controls the smoothness of the field with higher orders being smoother.
We recommend sticking with either `order=1,2`. For more information about the
impact of the order see [`MarkovRandomFieldGraph`](@ref).
"""
function ExpMarkovRandomField(ρ::Number, dims::Dims{2}; order = 1)
    cache = MarkovRandomFieldGraph(typeof(ρ), dims; order)
    return ExpMarkovRandomField(ρ, cache)
end


"""
    ExpMarkovRandomField(ρ, cache::MarkovRandomFieldGraph)

Constructs a first order zero-mean and unit variance Exponential Markov random field using the
precomputed cache `cache`.
"""
function ExpMarkovRandomField(ρ::Number, cache::MarkovRandomFieldGraph)
    return ExpMarkovRandomField{typeof(ρ), typeof(cache)}(ρ, cache)
end

function lognorm(d::ExpMarkovRandomField)
    N = length(d)
    return (logdet(d) - Dists.log2π * N) / 2
end

function unnormed_logpdf(d::ExpMarkovRandomField, I::AbstractMatrix)
    ρ = corrparam(d)
    return -sqrt(sq_manoblis(graph(d), I, ρ) * length(d))
end

function Dists._rand!(rng::AbstractRNG, d::ExpMarkovRandomField, x::AbstractMatrix{<:Real})
    Q = scalematrix(d)
    cQ = cholesky(Q)
    z = randn(rng, length(x))
    R = rand(rng, Dists.Chisq(2 * length(d)))
    return x .= R .* reshape(cQ.PtL' \ z, size(d)) ./ norm(z) / sqrt(length(d))
end
