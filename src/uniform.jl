export ImageUniform, ImageSphericalUniform, SphericalUnitVector

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

struct SphericalUnitVector{N} <: TV.VectorTransform
    function SphericalUnitVector{N}() where {N}
        TV.@argcheck N ≥ 1 "Dimension should be positive."
        new{N}()
    end
end

TV.dimension(::SphericalUnitVector{N}) where {N} = N

function TV.transform_with(flag::TV.LogJacFlag, ::SphericalUnitVector{N}, y::AbstractVector, index) where {N}
    T = TV.extended_eltype(y)
    index2 = index + N
    # normalized vector
    vy = NTuple{N,T}(@view(y[index:(index2-1)]))
    sly = sum(abs2, vy)

    x = sly > 0 ? vy ./ sqrt(sly) : ntuple(i->(i==1 ? one(T) : zero(T)), N)
    # jacobian term
    ℓi = TV.logjac_zero(flag, T)

    if !(flag isa TV.NoLogJac)
        ℓi -= sly/2
    end

    return x, ℓi, index2
end



struct ImageSphericalUniform{T} <: Dists.ContinuousMatrixDistribution
    nx::Int
    ny::Int
end

Base.size(d::ImageSphericalUniform) = (d.ny, d.nx)

function Dists.logpdf(::ImageSphericalUniform, X::AbstractMatrix{NTuple{3, T}}) where {T}
    return sum(x->log(1-last(x)^2), X)/2 - length(X)*log(4π)
end

function Dists.rand!(rng::Random.AbstractRNG, ::ImageSphericalUniform, x::AbstractMatrix)
    t = SphericalUnitVector{3}()
    for p in x
        p = TV.transform(t, randn(rng, 3))
    end
end

function Dists.rand(rng::Random.AbstractRNG, d::ImageSphericalUniform)
    t = SphericalUnitVector{3}()
    r = rand(rng, 3, d.nx*d.ny)
    dr = TV.transform.(Ref(t), eachcol(r))
    return reshape(dr, d.ny, d.nx)
end

function ChainRulesCore.rrule(::typeof(Dists.logpdf), d::ImageSphericalUniform, x::AbstractMatrix{NTuple{3,T}}) where {T}
    lp =  Dists.logpdf(d, x)
    function _spherical_uniform_pullback(Δ)
        z = last.(x)
        z .= -z./(1 .- z.^2)
        Δx = map(x->(zero(T), zero(T), Δ*x), z)
        return (NoTangent(), NoTangent(), Δx)
    end
    return lp, _spherical_uniform_pullback
end
