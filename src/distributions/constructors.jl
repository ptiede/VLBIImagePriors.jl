# User-facing constructor functions. Names are prefixed with `VLBI` to avoid
# clashing with `Distributions.jl` exports of the same family names. Each
# family covers three regimes via dispatch:
#
#   1. pure scalar           — `VLBIGaussian(μ, σ)`            (N = 0)
#   2. shared params + shape — `VLBIGaussian(μ, σ, dims)`      (N > 0, scalar params)
#   3. per-element params    — `VLBIGaussian(μ_grid, σ_grid)`  (N > 0, array params)
#
# plus the obvious mixed scalar/array combinations. Each of these returns
# an `AffineDistribution` over the appropriate Std base. Argument validity
# is checked at construction time via `_check_pos` / `_check_lt`; the checks
# are silently skipped for non-`Real` (Reactant traced) inputs since we
# can't branch on a traced value.


# --- Gaussian -------------------------------------------------------------

function VLBIGaussian(μ::Number, σ::Number)
    _check_pos("VLBIGaussian", "σ", σ)
    T = _promoteT(μ, σ)
    return AffineDistribution(μ, σ, StdNormal{T, 0}(()))
end
function VLBIGaussian(μ::Number, σ::Number, dims::Dims{N}) where {N}
    _check_pos("VLBIGaussian", "σ", σ)
    T = _promoteT(μ, σ)
    return AffineDistribution(μ, σ, StdNormal{T, N}(dims))
end
VLBIGaussian(μ::Number, σ::Number, dims::Int...) = VLBIGaussian(μ, σ, dims)
function VLBIGaussian(
        μ::AbstractArray{<:Number, N}, σ::AbstractArray{<:Number, N}
    ) where {N}
    @argcheck size(μ) == size(σ) "VLBIGaussian: μ and σ must have the same shape"
    _check_pos("VLBIGaussian", "σ", σ)
    T = _promoteT(μ, σ)
    return AffineDistribution(μ, σ, StdNormal{T, N}(size(μ)))
end
function VLBIGaussian(μ::Number, σ::AbstractArray{<:Number, N}) where {N}
    _check_pos("VLBIGaussian", "σ", σ)
    T = _promoteT(μ, σ)
    return AffineDistribution(μ, σ, StdNormal{T, N}(size(σ)))
end
function VLBIGaussian(μ::AbstractArray{<:Number, N}, σ::Number) where {N}
    _check_pos("VLBIGaussian", "σ", σ)
    T = _promoteT(μ, σ)
    return AffineDistribution(μ, σ, StdNormal{T, N}(size(μ)))
end


# --- Exponential ----------------------------------------------------------

function VLBIExponential(θ::Number)
    _check_pos("VLBIExponential", "θ", θ)
    T = _baseT(θ)
    return AffineDistribution(zero(θ), θ, StdExponential{T, 0}(()))
end
function VLBIExponential(θ::Number, dims::Dims{N}) where {N}
    _check_pos("VLBIExponential", "θ", θ)
    T = _baseT(θ)
    return AffineDistribution(zero(θ), θ, StdExponential{T, N}(dims))
end
VLBIExponential(θ::Number, dims::Int...) = VLBIExponential(θ, dims)
function VLBIExponential(θ::AbstractArray{<:Number, N}) where {N}
    _check_pos("VLBIExponential", "θ", θ)
    T = _baseT(θ)
    return AffineDistribution(zero(eltype(θ)), θ, StdExponential{T, N}(size(θ)))
end


# --- Uniform --------------------------------------------------------------

function VLBIUniform(a::Number, b::Number)
    _check_lt("VLBIUniform", a, b)
    T = _promoteT(a, b)
    return AffineDistribution(a, b - a, StdUniform{T, 0}(()))
end
function VLBIUniform(a::Number, b::Number, dims::Dims{N}) where {N}
    _check_lt("VLBIUniform", a, b)
    T = _promoteT(a, b)
    return AffineDistribution(a, b - a, StdUniform{T, N}(dims))
end
VLBIUniform(a::Number, b::Number, dims::Int...) = VLBIUniform(a, b, dims)
function VLBIUniform(
        a::AbstractArray{<:Number, N}, b::AbstractArray{<:Number, N}
    ) where {N}
    @argcheck size(a) == size(b) "VLBIUniform: a and b must have the same shape"
    _check_lt("VLBIUniform", a, b)
    T = _promoteT(a, b)
    return AffineDistribution(a, b .- a, StdUniform{T, N}(size(a)))
end
function VLBIUniform(a::Number, b::AbstractArray{<:Number, N}) where {N}
    _check_lt("VLBIUniform", a, b)
    T = _promoteT(a, b)
    return AffineDistribution(a, b .- a, StdUniform{T, N}(size(b)))
end
function VLBIUniform(a::AbstractArray{<:Number, N}, b::Number) where {N}
    _check_lt("VLBIUniform", a, b)
    T = _promoteT(a, b)
    return AffineDistribution(a, b .- a, StdUniform{T, N}(size(a)))
end


# --- InverseGamma ---------------------------------------------------------

function VLBIInverseGamma(α::Number, θ::Number)
    _check_pos("VLBIInverseGamma", "α", α)
    _check_pos("VLBIInverseGamma", "θ", θ)
    T = _promoteT(α, θ)
    return AffineDistribution(zero(θ), θ, StdInverseGamma{T, typeof(α), 0}(α, ()))
end
function VLBIInverseGamma(α::Number, θ::Number, dims::Dims{N}) where {N}
    _check_pos("VLBIInverseGamma", "α", α)
    _check_pos("VLBIInverseGamma", "θ", θ)
    T = _promoteT(α, θ)
    return AffineDistribution(zero(θ), θ, StdInverseGamma{T, typeof(α), N}(α, dims))
end
VLBIInverseGamma(α::Number, θ::Number, dims::Int...) = VLBIInverseGamma(α, θ, dims)
function VLBIInverseGamma(
        α::AbstractArray{<:Number, N}, θ::AbstractArray{<:Number, N}
    ) where {N}
    @argcheck size(α) == size(θ) "VLBIInverseGamma: α and θ must have the same shape"
    _check_pos("VLBIInverseGamma", "α", α)
    _check_pos("VLBIInverseGamma", "θ", θ)
    T = _promoteT(α, θ)
    return AffineDistribution(
        zero(eltype(θ)), θ, StdInverseGamma{T, typeof(α), N}(α, size(α))
    )
end
function VLBIInverseGamma(α::Number, θ::AbstractArray{<:Number, N}) where {N}
    _check_pos("VLBIInverseGamma", "α", α)
    _check_pos("VLBIInverseGamma", "θ", θ)
    T = _promoteT(α, θ)
    return AffineDistribution(
        zero(eltype(θ)), θ, StdInverseGamma{T, typeof(α), N}(α, size(θ))
    )
end
function VLBIInverseGamma(α::AbstractArray{<:Number, N}, θ::Number) where {N}
    _check_pos("VLBIInverseGamma", "α", α)
    _check_pos("VLBIInverseGamma", "θ", θ)
    T = _promoteT(α, θ)
    return AffineDistribution(zero(θ), θ, StdInverseGamma{T, typeof(α), N}(α, size(α)))
end


# --- TDist (Student's t) -------------------------------------------------
# `ν` is the degrees-of-freedom (intrinsic shape). `μ` and `σ` shift and
# scale the standard t — typical use is robust regression, where `σ` is the
# scale and `ν` controls tail heaviness.

VLBITDist(ν::Number) = VLBITDist(ν, 0.0, 1.0)
function VLBITDist(ν::Number, μ::Number, σ::Number)
    _check_pos("VLBITDist", "ν", ν)
    _check_pos("VLBITDist", "σ", σ)
    T = promote_type(_baseT(ν), _baseT(μ), _baseT(σ))
    return AffineDistribution(μ, σ, StdTDist{T, typeof(ν), 0}(ν, ()))
end
function VLBITDist(ν::Number, μ::Number, σ::Number, dims::Dims{N}) where {N}
    _check_pos("VLBITDist", "ν", ν)
    _check_pos("VLBITDist", "σ", σ)
    T = promote_type(_baseT(ν), _baseT(μ), _baseT(σ))
    return AffineDistribution(μ, σ, StdTDist{T, typeof(ν), N}(ν, dims))
end
function VLBITDist(ν::Number, μ::Number, σ::Number, dims::Int...)
    return VLBITDist(ν, μ, σ, dims)
end
function VLBITDist(
        ν::AbstractArray{<:Number, N},
        μ::AbstractArray{<:Number, N},
        σ::AbstractArray{<:Number, N}
    ) where {N}
    @argcheck size(ν) == size(μ) == size(σ) "VLBITDist: ν, μ, σ must have the same shape"
    _check_pos("VLBITDist", "ν", ν)
    _check_pos("VLBITDist", "σ", σ)
    T = promote_type(_baseT(ν), _baseT(μ), _baseT(σ))
    return AffineDistribution(μ, σ, StdTDist{T, typeof(ν), N}(ν, size(ν)))
end
function VLBITDist(
        ν::AbstractArray{<:Number, N}, μ::Number, σ::Number
    ) where {N}
    _check_pos("VLBITDist", "ν", ν)
    _check_pos("VLBITDist", "σ", σ)
    T = promote_type(_baseT(ν), _baseT(μ), _baseT(σ))
    return AffineDistribution(μ, σ, StdTDist{T, typeof(ν), N}(ν, size(ν)))
end
function VLBITDist(
        ν::Number,
        μ::AbstractArray{<:Number, N},
        σ::AbstractArray{<:Number, N}
    ) where {N}
    @argcheck size(μ) == size(σ) "VLBITDist: μ and σ must have the same shape"
    _check_pos("VLBITDist", "ν", ν)
    _check_pos("VLBITDist", "σ", σ)
    T = promote_type(_baseT(ν), _baseT(μ), _baseT(σ))
    return AffineDistribution(μ, σ, StdTDist{T, typeof(ν), N}(ν, size(μ)))
end
