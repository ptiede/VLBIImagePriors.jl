export ImageUniform

"""
    ImageUniform(a::Real, b::Real, ny, ny)

A uniform distribution in image pixels where `a/b` are the
lower/upper bound for the interval. This then concatenates ny×nx
uniform distributions together.
"""
struct ImageUniform{T} <: Dists.ContinuousMatrixDistribution
    a::T
    b::T
    nx::Int
    ny::Int
    function ImageUniform(a::Real, b::Real, nx::Integer, ny::Integer)
        aT,bT = promote(a,b)
        T = typeof(aT)
        return new{T}(aT, bT, nx, ny)
    end
end

Base.size(d::ImageUniform) = (d.ny,d.nx)

Dists.mean(::ImageUniform) = FillArrays.Fill((b-a)/2, ny, ny)

HC.asflat(d::ImageUniform) = TV.as(Array, as(Real, d.a, d.b), ny*nx)

function Dists.insupport(d::ImageUniform, x::AbstractMatrix)
    return (size(d) == size(x)) && !any(x-> (d.a >x)||(x> d.b), x)
end

function Dists._logpdf(d::ImageUniform, x::AbstractMatrix{<:Real})
    !Dists.insupport(d, x) && return -Inf
    return -log(d.b-d.a)^(d.nx*d.ny)
end

function ChainRulesCore.rrule(::typeof(Dists._logpdf), d::ImageUniform, x::AbstractMatrix{<:Real})
    return Dists._logpdf(d, x), Δ->(NoTangent(), ZeroTangent())
end
