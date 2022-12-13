export ImageDirichlet

"""
    ImageDirichlet(α::AbstractMatrix)
    ImageDirichlet(α::Real, ny, nx)

A Dirichlet distribution defined on a matrix. Samples from this produce matrices
whose elements sum to unity. This is a useful image prior when you want to separately
constrain the flux. The  α parameter defines the usual Dirichlet concentration amount.

# Notes

Much of this code was taken from [Distributions.jl](https://github.com/JuliaStats/Distributions.jl)
and it's Dirichlet distribution. However, some changes were made to make it faster. Additionally,
we use define a custom `rrule` to speed up derivatives.
"""
struct ImageDirichlet{T,A<:AbstractMatrix{T}, S} <: Dists.ContinuousMatrixDistribution
    α::A
    α0::T
    lmnB::S
    function ImageDirichlet{T}(α::AbstractMatrix{T}) where {T}
        α0 = sum(α)
        lmnB = sum(loggamma, α) - loggamma(α0)
        new{T, typeof(α), typeof(lmnB)}(α, α0, lmnB)
    end
end

function ImageDirichlet(α::AbstractMatrix{T}) where {T}
    return ImageDirichlet{T}(α)
end

function ImageDirichlet(α::Real, nx::Int, ny::Int)
    return ImageDirichlet(FillArrays.Fill(α, nx, ny))
end

Base.size(d::ImageDirichlet) = size(d.α)

Dists.mean(d::ImageDirichlet) = d.α.*inv(d.α0)

HC.asflat(d::ImageDirichlet) = ImageSimplex(size(d))

function Dists.insupport(d::ImageDirichlet, x::AbstractMatrix)
    return (size(d.α) == size(x)) && !any(x -> x < zero(x), x) && sum(x) ≈ 1
end

function Dists._logpdf(d::ImageDirichlet, x::AbstractMatrix{<:Real})
    if !(Dists.insupport(d, x))
        return xlogy(one(eltype(d.α)), zero(eltype(x))) - d.lmnB
    end

    return dirichlet_lpdf(d.α, d.lmnB, x)
end


function dirichlet_lpdf(α, lmnB, x)
    s = -lmnB
    @simd for i in eachindex(x, α)
        s += xlogy(α[i]-1, x[i])
    end
    return s
end

function ChainRulesCore.rrule(::typeof(dirichlet_lpdf), α, lmnB, x::AbstractMatrix{<:Real})
    f = dirichlet_lpdf(α, lmnB, x)
    px = ProjectTo(x)
    function _dirichlet_lpdf_pullback(Δ)
        Δα = @thunk(Δ.*log.(x))
        ΔlmnB = @thunk(-Δ)
        Δx = @thunk((α .- 1)./x)
        return (NoTangent(),Δα, ΔlmnB, px(Δx))
    end
    return f, _dirichlet_lpdf_pullback
end

function ChainRulesCore.rrule(::typeof(dirichlet_lpdf), α::FillArrays.AbstractFill, lmnB, x::AbstractMatrix{<:Real})
    f = dirichlet_lpdf(α, lmnB, x)
    px = ProjectTo(x)
    function _dirichlet_lpdf_pullback(Δ)
        Δα = Δ.*log.(x)
        ΔlmnB = -Δ
        Δx = Δ*(α[begin] - 1)./x
        return (NoTangent(),Δα, ΔlmnB, px(Δx))
    end
    return f, _dirichlet_lpdf_pullback
end

# Taken from  https://github.com/JuliaStats/Distributions.jl/blob/master/src/multivariate/dirichlet.jl
function Dists._rand!(rng::AbstractRNG, d::ImageDirichlet, x::AbstractMatrix)
    for (i, αi) in zip(eachindex(x), d.alpha)
        @inbounds x[i] = Dists.rand(rng, Dists.Gamma(αi))
    end
    lmul!(inv(sum(x)), x) # this returns x
end

# Taken from https://github.com/JuliaStats/Distributions.jl/blob/master/src/multivariate/dirichlet.jl
function Dists._rand!(rng::AbstractRNG,
    d::ImageDirichlet{T,<:FillArrays.AbstractFill{T}},
    x::AbstractMatrix{<:Real}) where {T<:Real}
    Dists.rand!(rng, Dists.Gamma(FillArrays.getindex_value(d.α)), x)
    lmul!(inv(sum(x)), x) # this returns x
end
