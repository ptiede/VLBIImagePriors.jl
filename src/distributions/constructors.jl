# User-facing constructor functions. Names are prefixed with `VLBI` to avoid
# clashing with `Distributions.jl` exports of the same family names. Each
# family covers three regimes via dispatch:
#
#   1. pure scalar           — `VLBIGaussian(μ, σ)`            (N = 0)
#   2. shared params + shape — `VLBIGaussian(μ, σ, dims)`      (N > 0, scalar params)
#   3. per-element params    — `VLBIGaussian(μ_grid, σ_grid)`  (N > 0, array params)
#
# plus the obvious mixed scalar/array combinations. Each of these returns
# an `AffineDistribution` over the appropriate Std base. No validation is
# performed — Reactant cannot throw exceptions, and shape mismatches are
# caught downstream by broadcasting.


# --- Gaussian -------------------------------------------------------------

function VLBIGaussian(μ::Number, σ::Number)
    T = promote_type(eltype(μ), eltype(σ))
    return AffineDistribution(μ, σ, StdNormal{T, 0}(()))
end
function VLBIGaussian(μ::Number, σ::Number, dims::Dims{N}) where {N}
    T = promote_type(eltype(μ), eltype(σ))
    return AffineDistribution(μ, σ, StdNormal{T, N}(dims))
end
VLBIGaussian(μ::Number, σ::Number, dims::Int...) = VLBIGaussian(μ, σ, dims)
function VLBIGaussian(
        μ::AbstractArray{<:Number, N}, σ::AbstractArray{<:Number, N}
    ) where {N}
    T = promote_type(eltype(μ), eltype(σ))
    return AffineDistribution(μ, σ, StdNormal{T, N}(size(μ)))
end
function VLBIGaussian(μ::Number, σ::AbstractArray{<:Number, N}) where {N}
    T = promote_type(eltype(μ), eltype(σ))
    return AffineDistribution(μ, σ, StdNormal{T, N}(size(σ)))
end
function VLBIGaussian(μ::AbstractArray{<:Number, N}, σ::Number) where {N}
    T = promote_type(eltype(μ), eltype(σ))
    return AffineDistribution(μ, σ, StdNormal{T, N}(size(μ)))
end


# --- Exponential ----------------------------------------------------------

function VLBIExponential(θ::Number)
    T = eltype(θ)
    return AffineDistribution(zero(θ), θ, StdExponential{T, 0}(()))
end
function VLBIExponential(θ::Number, dims::Dims{N}) where {N}
    T = eltype(θ)
    return AffineDistribution(zero(θ), θ, StdExponential{T, N}(dims))
end
VLBIExponential(θ::Number, dims::Int...) = VLBIExponential(θ, dims)
function VLBIExponential(θ::AbstractArray{<:Number, N}) where {N}
    T = eltype(θ)
    return AffineDistribution(zero(eltype(θ)), θ, StdExponential{T, N}(size(θ)))
end


# --- Uniform --------------------------------------------------------------

function VLBIUniform(a::Number, b::Number)
    T = promote_type(eltype(a), eltype(b))
    return AffineDistribution(a, b - a, StdUniform{T, 0}(()))
end
function VLBIUniform(a::Number, b::Number, dims::Dims{N}) where {N}
    T = promote_type(eltype(a), eltype(b))
    return AffineDistribution(a, b - a, StdUniform{T, N}(dims))
end
VLBIUniform(a::Number, b::Number, dims::Int...) = VLBIUniform(a, b, dims)
function VLBIUniform(
        a::AbstractArray{<:Number, N}, b::AbstractArray{<:Number, N}
    ) where {N}
    T = promote_type(eltype(a), eltype(b))
    return AffineDistribution(a, b .- a, StdUniform{T, N}(size(a)))
end
function VLBIUniform(a::Number, b::AbstractArray{<:Number, N}) where {N}
    T = promote_type(eltype(a), eltype(b))
    return AffineDistribution(a, b .- a, StdUniform{T, N}(size(b)))
end
function VLBIUniform(a::AbstractArray{<:Number, N}, b::Number) where {N}
    T = promote_type(eltype(a), eltype(b))
    return AffineDistribution(a, b .- a, StdUniform{T, N}(size(a)))
end


# --- InverseGamma ---------------------------------------------------------

function VLBIInverseGamma(α::Number, θ::Number)
    return AffineDistribution(zero(θ), θ, StdInverseGamma(α, ()))
end
function VLBIInverseGamma(α::Number, θ::Number, dims::Dims{N}) where {N}
    return AffineDistribution(zero(θ), θ, StdInverseGamma(α, dims))
end
VLBIInverseGamma(α::Number, θ::Number, dims::Int...) = VLBIInverseGamma(α, θ, dims)
function VLBIInverseGamma(
        α::AbstractArray{<:Number, N}, θ::AbstractArray{<:Number, N}
    ) where {N}
    return AffineDistribution(zero(eltype(θ)), θ, StdInverseGamma(α, size(α)))
end
function VLBIInverseGamma(α::Number, θ::AbstractArray{<:Number, N}) where {N}
    return AffineDistribution(zero(eltype(θ)), θ, StdInverseGamma(α, size(θ)))
end
function VLBIInverseGamma(α::AbstractArray{<:Number, N}, θ::Number) where {N}
    return AffineDistribution(zero(θ), θ, StdInverseGamma(α, size(α)))
end


# --- TDist (Student's t) -------------------------------------------------
# `ν` is the degrees-of-freedom (intrinsic shape). `μ` and `σ` shift and
# scale the standard t — typical use is robust regression, where `σ` is the
# scale and `ν` controls tail heaviness.

VLBITDist(ν::Number) = VLBITDist(ν, 0.0, 1.0)
function VLBITDist(ν::Number, μ::Number, σ::Number)
    return AffineDistribution(μ, σ, StdTDist(ν, ()))
end
function VLBITDist(ν::Number, μ::Number, σ::Number, dims::Dims{N}) where {N}
    return AffineDistribution(μ, σ, StdTDist(ν, dims))
end
function VLBITDist(ν::Number, μ::Number, σ::Number, dims::Int...)
    return VLBITDist(ν, μ, σ, dims)
end
function VLBITDist(
        ν::AbstractArray{<:Number, N},
        μ::AbstractArray{<:Number, N},
        σ::AbstractArray{<:Number, N}
    ) where {N}
    return AffineDistribution(μ, σ, StdTDist(ν, size(ν)))
end
function VLBITDist(
        ν::AbstractArray{<:Number, N}, μ::Number, σ::Number
    ) where {N}
    return AffineDistribution(μ, σ, StdTDist(ν, size(ν)))
end
function VLBITDist(
        ν::Number,
        μ::AbstractArray{<:Number, N},
        σ::AbstractArray{<:Number, N}
    ) where {N}
    return AffineDistribution(μ, σ, StdTDist(ν, size(μ)))
end
