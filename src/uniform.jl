export ImageUniform, ImageSphericalUniform
using FillArrays

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

function ImageUniform(nx, ny)
    return ImageUniform(0.0, 1.0, nx, ny)
end

Base.size(d::ImageUniform) = (d.nx,d.ny)

Dists.mean(d::ImageUniform) = FillArrays.Fill((d.b-d.a)/2, size(d)...)

HC.asflat(d::ImageUniform) = TV.as(Matrix, TV.as(Real, d.a, d.b), d.ny, d.nx)

function Dists.insupport(d::ImageUniform, x::AbstractMatrix)
    return (size(d) == size(x)) && !any(x-> (d.a >x)||(x> d.b), x)
end

function Dists._logpdf(d::ImageUniform, x::AbstractMatrix{<:Real})
    !Dists.insupport(d, x) && return -Inf
    return -log(d.b-d.a)*(d.nx*d.ny)
end

function ChainRulesCore.rrule(::typeof(Dists._logpdf), d::ImageUniform, x::AbstractMatrix{<:Real})
    return Dists._logpdf(d, x), Δ->(NoTangent(), NoTangent(), ZeroTangent())
end

function Dists._rand!(rng::AbstractRNG, d::ImageUniform, x::AbstractMatrix)
    @assert size(d) == size(x) "Size of input matrix and distribution are not the same"
    d = Dists.Uniform(d.a, d.b)
    rand!(rng, d, x)
end


struct ImageSphericalUniform{T} <: Dists.ContinuousMatrixDistribution
    nx::Int
    ny::Int
end

ImageSphericalUniform(nx::Int, ny::Int) = ImageSphericalUniform{Float64}(nx, ny)

HC.asflat(d::RadioImagePriors.ImageSphericalUniform) = TV.as(Matrix, SphericalUnitVector{2}(), d.nx, d.ny)


Base.size(d::ImageSphericalUniform) = (d.nx, d.ny)

function Dists.logpdf(::ImageSphericalUniform, X::Union{AbstractMatrix{NTuple{3,S}}, NTuple{3, T}}) where {T<:AbstractMatrix, S<:Real}
    return -length(X[1])*log(4π)
end

function Dists.rand!(rng::Random.AbstractRNG, ::ImageSphericalUniform, x::AbstractMatrix)
    t = SphericalUnitVector{3}()
    for p in x
        p = TV.transform(t, randn(rng, 3))
    end
end

function Dists.rand(rng::Random.AbstractRNG, d::ImageSphericalUniform)
    t = TV.as(d)
    r1 = randn(rng, d.nx*d.ny)
    r2 = randn(rng, d.nx*d.ny)
    r3 = randn(rng, d.nx*d.ny)

    for i in eachindex(r1)
        norm = hypot(r1[i], r2[i], r3[i])
        r1[i] /= norm
        r2[i] /= norm
        r3[i] /= norm
    end

    return (r1, r2, r3)
end

function ChainRulesCore.rrule(::typeof(Dists.logpdf), d::ImageSphericalUniform, x::NTuple{3,<:AbstractMatrix{S}}) where {S<:Number}
    lp =  Dists.logpdf(d, x)
    function _spherical_uniform_pullback(Δ)
        # z = last.(x)
        # z .= -z./(1 .- z.^2)
        # Δx = map(x->(zero(T), zero(T), zero(T)), z)
        return (NoTangent(), NoTangent(), ZeroTangent())
    end
    return lp, _spherical_uniform_pullback
end
