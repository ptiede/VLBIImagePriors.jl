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
struct ImageDirichlet{T, A <: AbstractMatrix{T}, S} <: Dists.ContinuousMatrixDistribution
    α::A
    α0::T
    lmnB::S
    function ImageDirichlet(α::AbstractMatrix{T}) where {T}
        α0 = sum(α)
        lmnB = sum(loggamma, α) - loggamma(α0)
        return new{T, typeof(α), typeof(lmnB)}(α, α0, lmnB)
    end
end

function ImageDirichlet(α::Real, nx::Int, ny::Int)
    return ImageDirichlet(fill(α, nx, ny))
end

Base.size(d::ImageDirichlet) = size(d.α)

Dists.mean(d::ImageDirichlet) = d.α .* inv(d.α0)

HC.asflat(d::ImageDirichlet) = ImageSimplex(size(d))

function Dists.insupport(d::ImageDirichlet, x::AbstractMatrix)
    return (size(d.α) == size(x)) & !any(<(zero(eltype(x))), x) & isapprox(sum(x), one(eltype(x)))
end

using EnzymeCore: EnzymeRules
EnzymeRules.inactive(::typeof(Dists.insupport), args...) = nothing

function Dists.logpdf(d::ImageDirichlet, x::AbstractMatrix)
    l = dirichlet_lpdf(d.α, d.lmnB, x)
    return ifelse(Dists.insupport(d, x), l, oftype(typeof(l), -Inf))
end

function myxlogym1(x::Number, y::Number)
    result = (x-1) * log(y)
    b = iszero(x) & isnan(y)
    r = ifelse(b, zero(result), result)
    return r
end



function dirichlet_lpdf(α, lmnB, x)
    s = mapreduce(splat(myxlogym1), +, zip(α, x))
    return s - lmnB
end

# function ChainRulesCore.rrule(::typeof(dirichlet_lpdf), α, lmnB, x::AbstractMatrix{<:Real})
#     f = dirichlet_lpdf(α, lmnB, x)
#     px = ProjectTo(x)
#     function _dirichlet_lpdf_pullback(Δ)
#         Δα = @thunk(Δ.*log.(x))
#         ΔlmnB = @thunk(-Δ)
#         Δx = @thunk((α .- 1)./x)
#         return (NoTangent(),Δα, ΔlmnB, px(Δx))
#     end
#     return f, _dirichlet_lpdf_pullback
# end

function ChainRulesCore.rrule(::typeof(dirichlet_lpdf), α, lmnB, x::AbstractMatrix{<:Real})
    f = dirichlet_lpdf(α, lmnB, x)
    px = ProjectTo(x)
    function _dirichlet_lpdf_pullback(Δ)
        Δα = @thunk(Δ .* log.(x))
        ΔlmnB = -Δ
        Δx = @thunk(Δ .* (α .- 1) ./ x)
        return (NoTangent(), Δα, ΔlmnB, px(Δx))
    end
    return f, _dirichlet_lpdf_pullback
end

# Taken from  https://github.com/JuliaStats/Distributions.jl/blob/master/src/multivariate/dirichlet.jl
function Dists._rand!(rng::AbstractRNG, d::ImageDirichlet, x::AbstractMatrix)
    for (i, αi) in zip(eachindex(x), d.α)
        @inbounds x[i] = Dists.rand(rng, Dists.Gamma(αi))
    end
    return lmul!(inv(sum(x)), x) # this returns x
end

# Taken from https://github.com/JuliaStats/Distributions.jl/blob/master/src/multivariate/dirichlet.jl
function Dists._rand!(
        rng::AbstractRNG,
        d::ImageDirichlet{T, <:FillArrays.AbstractFill{T}},
        x::AbstractMatrix{<:Real}
    ) where {T <: Real}
    Dists.rand!(rng, Dists.Gamma(FillArrays.getindex_value(d.α)), x)
    return lmul!(inv(sum(x)), x) # this returns x
end
