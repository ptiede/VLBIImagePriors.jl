using SparseArrays
using LinearAlgebra
using ComradeBase
using Serialization

export GaussMarkovRandomField, GMRF

#=
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
julia> cache = MarkovRandomFieldGraph(Float64, (32, 32)) # now instead construct the cache
julia> d2 = GaussMarkovRandomField(ρ, cache)
julia> invcov(d) ≈ invcov(d2)
true
```
=#
struct GaussMarkovRandomField{T<:Number,C<:MarkovRandomFieldGraph} <: MarkovRandomField
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
    Alias for `GaussMarkovRandomField`
"""
const GMRF = GaussMarkovRandomField

Dists.mean(d::GaussMarkovRandomField{T}) where {T} = FillArrays.Zeros(T, size(d))
Dists.cov(d::GaussMarkovRandomField)  = inv(Array(Dists.invcov(d)))
Dists.invcov(d::GaussMarkovRandomField) = scalematrix(d)

(c::ConditionalMarkov{<:GMRF})(ρ)  = GaussMarkovRandomField(ρ, c.cache)


"""
    GaussMarkovRandomField(ρ, img::AbstractArray; order::Integer=1)

Constructs a `order`ᵗʰ order  Gaussian Markov random field with
dimensions `size(img)`, correlation `ρ` and unit covariance.

The `order` parameter controls the smoothness of the field with higher orders being smoother.
We recommend sticking with either `order=1,2`. Noting that `order=1` is equivalent to the
usual TSV and L₂ regularization from RML imaging. For more information about the
impact of the order see [`MarkovRandomFieldGraph`](@ref).
"""
function GaussMarkovRandomField(ρ::Number, img::AbstractMatrix; order::Integer=1)
    cache = MarkovRandomFieldGraph(eltype(img), size(img); order)
    return GaussMarkovRandomField(ρ, cache)
end

"""
    GaussMarkovRandomField(ρ, dims; order=1)
    GaussMarkovRandomField(ρ, g::AbstractRectiGrid; order=1)

Constructs a `order`ᵗʰ order Gaussian Markov random field with
dimensions `size(img)`, correlation `ρ` and unit covariance.

The `order` parameter controls the smoothness of the field with higher orders being smoother.
We recommend sticking with either `order=1,2`. Noting that `order=1` is equivalent to the
usual TSV and L₂ regularization from RML imaging. For more information about the
impact of the order see [`MarkovRandomFieldGraph`](@ref).

"""
function GaussMarkovRandomField(ρ::Number, dims::Dims{2}; order::Integer=1)
    cache = MarkovRandomFieldGraph(typeof(ρ), dims; order)
    return GaussMarkovRandomField(ρ, cache)
end

function GaussMarkovRandomField(ρ::Number, g::ComradeBase.AbstractRectiGrid{<:Tuple{ComradeBase.X, ComradeBase.Y}}; order::Integer=1)
    return GaussMarkovRandomField(ρ, size(g); order)
end



@inline function lognorm(d::GaussMarkovRandomField)
    N = length(d)

    return (logdet(d) - Dists.log2π*N)/2
end

function unnormed_logpdf(d::GaussMarkovRandomField, I::AbstractMatrix)
    ρ = corrparam(d)
    return -sq_manoblis(graph(d), I, ρ)/2
end

function Dists._rand!(rng::AbstractRNG, d::GaussMarkovRandomField, x::AbstractMatrix{<:Real})
    Q = scalematrix(d)
    cQ = cholesky(Q)
    z = randn(rng, length(x))
    x .= reshape(cQ.PtL'\z, size(d))
end
