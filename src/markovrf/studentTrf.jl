using SparseArrays
using LinearAlgebra
using ComradeBase

export TDistMarkovRandomField, TMRF, CauchyMarkovRandomField

"""
    $(TYPEDEF)

A image prior based off of the first-order Multivariate T distribution Markov random field.

# Fields
$(FIELDS)

# Examples

```julia
julia> ρ, ν = 16.0, 1.0
julia> d = TDistMarkovRandomField(ρ, ν, (32, 32))
julia> cache = MarkovRandomFieldGraph(Float64, (32, 32)) # now instead construct the cache
julia> d2 = TDistMarkovRandomField(ρ, ν, cache)
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
    The student T "degrees of freedom parameter which ≥ 1 for a proper prior
    """
    ν::T
    """
    The Markov Random Field graph cache used to define the specific Markov random field class used.
    """
    graph::C
end

"""
    Alias for `TDistMarkovRandomField`
"""
const TMRF = TDistMarkovRandomField


(c::ConditionalMarkov{<:TMRF})(ρ, ν=1) = TDistMarkovRandomField(ρ, ν, c.cache)

Dists.mean(d::TDistMarkovRandomField{T}) where {T} = d.ν > 1 ? FillArrays.Zeros(T, size(d)) : FillArrays.Fill(convert(T, Inf), size(d))

function Dists.cov(d::TDistMarkovRandomField)
    d.ν > 2 && return d.ν*inv(d.ν - 2)*inv(Array(scalematrix(d)))
    return FillArrays.Fill(convert(typeof(ρ), Inf), size(scalematrix(d)))
end

function Dists.invcov(d::TDistMarkovRandomField)
    d.ν > 2 && return (d.ν - 2)/d.ν*(scalematrix(d))
    return FillArrays.Fill(convert(typeof(ρ), NaN), size(scalematrix(d)))
end



"""
    TDistMarkovRandomField(ρ, ν, img::AbstractArray; order=1)

Constructs a first order TDist Markov random field with zero median
dimensions `size(img)`, correlation `ρ` and degrees of freedom ν.

Note `ν ≥ 1` to be a well-defined probability distribution.

The `order` parameter controls the smoothness of the field with higher orders being smoother.
We recommend sticking with either `order=1,2`. For more information about the
impact of the order see [`MarkovRandomFieldGraph`](@ref).
"""
function TDistMarkovRandomField(ρ::Number, ν::Number, img::AbstractMatrix; order=1)
    cache = MarkovRandomFieldGraph(eltype(img), size(img); order)
    return TDistMarkovRandomField(ρ, ν, cache)
end

CauchyMarkovRandomField(ρ::Number, img::AbstractMatrix; order=1) = TDistMarkovRandomField(ρ, 1, img; order)

"""
    TDistMarkovRandomField(ρ, ν, cache::MarkovRandomFieldGraph)

Constructs a first order TDist Markov random field with zero mean ,correlation `ρ`,
degrees of freedom `ν`, and the precomputed MarkovRandomFieldGraph `cache`.
"""
function TDistMarkovRandomField(ρ::Number, ν::Number, cache::MarkovRandomFieldGraph)
    T = promote_type(typeof(ρ), typeof(ν))
    return TDistMarkovRandomField{T, typeof(cache)}(convert(T,ρ), convert(T,ν), cache)
end

CauchyMarkovRandomField(ρ::Number, cache::MarkovRandomFieldGraph) = TDistMarkovRandomField(ρ, 1, cache)


"""
    TDistMarkovRandomField(ρ, ν, dims)

Constructs a first order TDist Markov random field with zero mean ,correlation `ρ`,
degrees of freedom `ν`, with dimension `dims`.

The `order` parameter controls the smoothness of the field with higher orders being smoother.
We recommend sticking with either `order=1,2`. For more information about the
impact of the order see [`MarkovRandomFieldGraph`](@ref).
"""
function TDistMarkovRandomField(ρ::Number, ν::Number, dims::Dims{2}; order=1)
    T = promote_type(typeof(ρ), typeof(ν))
    cache = MarkovRandomFieldGraph(typeof(ρ), dims; order)
    return TDistMarkovRandomField{T, typeof(cache)}(convert(T,ρ), convert(T,ν), cache)
end

CauchyMarkovRandomField(ρ::Number, dims::Dims{2}; order=1) = TDistMarkovRandomField(ρ, 1, dims; order)





function lognorm(d::TDistMarkovRandomField)
    ν = d.ν
    N = length(d)
    det = logdet(d)
    return loggamma((ν+N)/2) - loggamma(ν/2) - N/2*log(ν*π) + det/2
end

function unnormed_logpdf(d::TDistMarkovRandomField, I::AbstractMatrix)
    (;ρ, ν) = d
    sq = sq_manoblis(d.graph, I, ρ)
    return -((ν+length(I))/2)*log1p(inv(ν)*sq)
end

function Dists._rand!(rng::AbstractRNG, d::TDistMarkovRandomField, x::AbstractMatrix{<:Real})
    Q = scalematrix(d)
    cQ = cholesky(Q)
    z = randn(rng, length(x))
    x .= sqrt(d.ν/rand(rng, Dists.Chisq(d.ν))).*reshape(cQ.PtL'\z, size(d))
end
