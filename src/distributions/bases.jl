# Standard base distributions: `StdExponential`, `StdUniform`,
# `StdInverseGamma`, plus the scalar-(N=0) augmentation for `StdNormal` that
# `src/srf.jl` does not provide. All four serve as the fixed shape inside an
# `AffineDistribution(loc, scale, base)` wrapper.
#
# Internally this file defines:
#   - the three new Std structs and their `Base.size`/`length`/`eltype`
#   - per-element log-pdf kernels (scalar, branchless via `ifelse`)
#   - vectorised array log-pdf kernels (Reactant-safe: no scalar `getindex`)
#   - cdf / quantile helpers written directly in arithmetic
#   - the `Distributions` interface (logpdf, _logpdf, _rand!, insupport,
#     moments, cdf, quantile) and `HC.asflat` for each Std base


# ----- struct definitions --------------------------------------------------

struct StdExponential{T, N} <: Dists.ContinuousDistribution{Dists.ArrayLikeVariate{N}}
    dims::Dims{N}
end
StdExponential(dims::Dims{N}) where {N} = StdExponential{Float64, N}(dims)
StdExponential(dims::Int...) = StdExponential(dims)
StdExponential() = StdExponential{Float64, 0}(())

struct StdUniform{T, N} <: Dists.ContinuousDistribution{Dists.ArrayLikeVariate{N}}
    dims::Dims{N}
end
StdUniform(dims::Dims{N}) where {N} = StdUniform{Float64, N}(dims)
StdUniform(dims::Int...) = StdUniform(dims)
StdUniform() = StdUniform{Float64, 0}(())

struct StdInverseGamma{T, Tα, N} <: Dists.ContinuousDistribution{Dists.ArrayLikeVariate{N}}
    α::Tα
    dims::Dims{N}
end
StdInverseGamma(α::Number) = StdInverseGamma{typeof(α), typeof(α), 0}(α, ())
function StdInverseGamma(α::Number, dims::Dims{N}) where {N}
    return StdInverseGamma{typeof(α), typeof(α), N}(α, dims)
end
StdInverseGamma(α::Number, dims::Int...) = StdInverseGamma(α, dims)
function StdInverseGamma(α::AbstractArray{<:Number, N}) where {N}
    return StdInverseGamma{eltype(α), typeof(α), N}(α, size(α))
end

Base.size(d::StdExponential) = d.dims
Base.size(d::StdUniform) = d.dims
Base.size(d::StdInverseGamma) = d.dims

Base.length(d::StdExponential) = prod(d.dims)
Base.length(d::StdUniform) = prod(d.dims)
Base.length(d::StdInverseGamma) = prod(d.dims)

Base.eltype(::StdExponential{T}) where {T} = T
Base.eltype(::StdUniform{T}) where {T} = T
Base.eltype(::StdInverseGamma{T}) where {T} = T


# ----- log-pdf split: `unnormed_logpdf` + `lognorm` -----------------------
# We expose the same two-piece interface as `MarkovRandomField` (see
# `src/markovrf/markovrf.jl`), so that callers can cache the data-independent
# normalisation (which can be expensive — `loggamma` over an array of shape
# parameters, `sum(log, scale)` over a per-element scale, etc.) and reuse it
# across many `logpdf` / `rand` evaluations.
#
# Branchless `ifelse` keeps the scalar kernels traceable under Reactant; the
# trailing `i` argument is the per-element index, used only by
# `StdInverseGamma` whose shape parameter `α` may itself be a per-element
# array.

# --- scalar unnormalised kernels ---
@inline _unnormed_kernel(::StdNormal, z, _) = -z * z / 2

@inline function _unnormed_kernel(::StdExponential, z, _)
    return ifelse(z >= zero(z), -z, oftype(z, -Inf))
end

@inline function _unnormed_kernel(::StdUniform, z, _)
    return ifelse((z >= zero(z)) & (z <= one(z)), zero(z), oftype(z, -Inf))
end

@inline function _unnormed_kernel(d::StdInverseGamma, z, i)
    αi = _at(d.α, i)
    zsafe = ifelse(z > zero(z), z, oftype(z, 1))
    val = -(αi + one(αi)) * log(zsafe) - inv(zsafe)
    return ifelse(z > zero(z), val, oftype(z, -Inf))
end

# --- array unnormalised kernels (vectorised, Reactant-safe) ---
# Sum the per-element unnormalised kernel via broadcasting + reductions so we
# never need scalar `getindex` on a traced array. Out-of-support inputs are
# the user's responsibility (check `Dists.insupport(d, x)`); we don't mask
# them here because branchless masking adds arithmetic to every element.

@inline _unnormed_kernel_sum(::StdNormal, z) = -sum(abs2.(z)) / 2
@inline _unnormed_kernel_sum(::StdExponential, z) = -sum(z)
@inline _unnormed_kernel_sum(::StdUniform, z) = zero(eltype(z))
@inline function _unnormed_kernel_sum(d::StdInverseGamma, z)
    α = d.α
    log_z = log.(z)
    inv_z = inv.(z)
    return -sum((α .+ 1) .* log_z) - sum(inv_z)
end

# --- scale Jacobian / `loggamma` reductions used by `lognorm` ---
@inline _scale_logsum(σ::Number, n::Int) = n * log(σ)
@inline function _scale_logsum(σ::AbstractArray, _)
    tmp = log.(σ)
    return sum(tmp)
end

@inline _ig_loggamma_sum(α::Number, n::Int) = n * loggamma(α)
@inline function _ig_loggamma_sum(α::AbstractArray, _)
    tmp = loggamma.(α)
    return sum(tmp)
end


# ----- public unnormed_logpdf / lognorm interface -------------------------
# The two pieces of `logpdf(d, x) = unnormed_logpdf(d, x) + lognorm(d)`. The
# `lognorm` constant is data-independent and caching it across repeated
# `logpdf` calls (with varying `x`) is the whole point of the split.

"""
    unnormed_logpdf(d, x)

Returns the part of `logpdf(d, x)` that depends on `x`. The full log-density
is `unnormed_logpdf(d, x) + lognorm(d)`.
"""
function unnormed_logpdf end

"""
    lognorm(d)

Returns the data-independent log-normalisation constant of `d`. Useful for
caching when the normalisation is expensive — e.g. a `StdInverseGamma` whose
shape parameter is a large array, or an `AffineDistribution` with a per-pixel
`scale` whose `sum(log, scale)` term doesn't depend on the input.
"""
function lognorm end

# Scalar (N = 0)
unnormed_logpdf(d::StdNormal{T, 0}, x::Number) where {T} = _unnormed_kernel(d, x, 1)
function unnormed_logpdf(d::StdExponential{T, 0}, x::Number) where {T}
    return _unnormed_kernel(d, x, 1)
end
unnormed_logpdf(d::StdUniform{T, 0}, x::Number) where {T} = _unnormed_kernel(d, x, 1)
function unnormed_logpdf(d::StdInverseGamma{T, <:Number, 0}, x::Number) where {T}
    return _unnormed_kernel(d, x, 1)
end

# Array (N >= 1)
function unnormed_logpdf(d::StdNormal{T, N}, x::AbstractArray{<:Number, N}) where {T, N}
    return _unnormed_kernel_sum(d, x)
end
function unnormed_logpdf(d::StdExponential{T, N}, x::AbstractArray{<:Number, N}) where {T, N}
    return _unnormed_kernel_sum(d, x)
end
function unnormed_logpdf(d::StdUniform{T, N}, x::AbstractArray{<:Number, N}) where {T, N}
    return _unnormed_kernel_sum(d, x)
end
function unnormed_logpdf(
        d::StdInverseGamma{T, Tα, N}, x::AbstractArray{<:Number, N}
    ) where {T, Tα, N}
    return _unnormed_kernel_sum(d, x)
end

# `lognorm` for the Std bases. `length(d)` is the number of elements (1 for
# scalar, `prod(dims)` for arrays), so a single formula handles both regimes.
@inline lognorm(d::StdNormal) = -length(d) * oftype(zero(eltype(d)), log(2π) / 2)
@inline lognorm(d::StdExponential) = zero(eltype(d))
@inline lognorm(d::StdUniform) = zero(eltype(d))
@inline lognorm(d::StdInverseGamma) = -_ig_loggamma_sum(d.α, length(d))


# ----- cdf / quantile kernels ---------------------------------------------
# Closed-form arithmetic so they trace cleanly under Reactant for traced
# (i.e. non-`Real`) parameters and inputs. Wrapping `Distributions.jl` types
# would force `<:Real`, which excludes Reactant tracers.

@inline _std_cdf(::StdNormal, x) = (one(x) + erf(x / sqrt(oftype(x, 2)))) / 2
@inline _std_quantile(::StdNormal, p) = sqrt(oftype(p, 2)) * erfinv(2 * p - one(p))

# StdExponential (rate 1)
@inline _std_cdf(::StdExponential, x) = -expm1(-x)
@inline _std_quantile(::StdExponential, p) = -log1p(-p)

# StdUniform on [0, 1]
@inline _std_cdf(::StdUniform, x) = clamp(x, zero(x), one(x))
@inline _std_quantile(::StdUniform, p) = p

# StdInverseGamma(α): cdf(x) = Q(α, 1/x), where Q is the regularised upper
# incomplete gamma. SpecialFunctions provides both directions.
@inline function _std_cdf(d::StdInverseGamma, x)
    α = _at(d.α, 1)
    return last(SpecialFunctions.gamma_inc(α, inv(x), 0))
end
@inline function _std_quantile(d::StdInverseGamma, p)
    α = _at(d.α, 1)
    return inv(SpecialFunctions.gamma_inc_inv(α, one(p) - p, p))
end


# ----- Distributions interface for Std bases ------------------------------

# Scalar (N = 0) logpdf — composes the `unnormed_logpdf` + `lognorm` split.
Dists.logpdf(d::StdNormal{T, 0}, x::Number) where {T} = unnormed_logpdf(d, x) + lognorm(d)
function Dists.logpdf(d::StdExponential{T, 0}, x::Number) where {T}
    return unnormed_logpdf(d, x) + lognorm(d)
end
Dists.logpdf(d::StdUniform{T, 0}, x::Number) where {T} = unnormed_logpdf(d, x) + lognorm(d)
function Dists.logpdf(d::StdInverseGamma{T, <:Number, 0}, x::Number) where {T}
    return unnormed_logpdf(d, x) + lognorm(d)
end

# Array (N >= 1) logpdf. Distributions' top-level array `logpdf` only matches
# `AbstractArray{<:Real, N}`; Reactant traced eltypes are `<:Number` but not
# `<:Real`, so we provide our own `logpdf` for `<:Number` and an explicit
# `<:Real` override that breaks ambiguity with the Distributions fallback.
@inline _std_array_logpdf(d, x) = unnormed_logpdf(d, x) + lognorm(d)

function Dists._logpdf(d::StdExponential{T, N}, x::AbstractArray{<:Number, N}) where {T, N}
    return _std_array_logpdf(d, x)
end
function Dists._logpdf(d::StdUniform{T, N}, x::AbstractArray{<:Number, N}) where {T, N}
    return _std_array_logpdf(d, x)
end
function Dists._logpdf(
        d::StdInverseGamma{T, Tα, N}, x::AbstractArray{<:Number, N}
    ) where {T, Tα, N}
    return _std_array_logpdf(d, x)
end

function Dists.logpdf(d::StdExponential{T, N}, x::AbstractArray{<:Real, N}) where {T, N}
    @argcheck size(x) == size(d) "input/distribution size mismatch"
    return _std_array_logpdf(d, x)
end
function Dists.logpdf(d::StdUniform{T, N}, x::AbstractArray{<:Real, N}) where {T, N}
    @argcheck size(x) == size(d) "input/distribution size mismatch"
    return _std_array_logpdf(d, x)
end
function Dists.logpdf(
        d::StdInverseGamma{T, Tα, N}, x::AbstractArray{<:Real, N}
    ) where {T, Tα, N}
    @argcheck size(x) == size(d) "input/distribution size mismatch"
    return _std_array_logpdf(d, x)
end

function Dists.logpdf(d::StdExponential{T, N}, x::AbstractArray{<:Number, N}) where {T, N}
    @argcheck size(x) == size(d) "input/distribution size mismatch"
    return _std_array_logpdf(d, x)
end
function Dists.logpdf(d::StdUniform{T, N}, x::AbstractArray{<:Number, N}) where {T, N}
    @argcheck size(x) == size(d) "input/distribution size mismatch"
    return _std_array_logpdf(d, x)
end
function Dists.logpdf(
        d::StdInverseGamma{T, Tα, N}, x::AbstractArray{<:Number, N}
    ) where {T, Tα, N}
    @argcheck size(x) == size(d) "input/distribution size mismatch"
    return _std_array_logpdf(d, x)
end

# Sampling (CPU-only paths; `rand` is not on the Reactant hot path)
Random.rand(rng::AbstractRNG, ::StdNormal{T, 0}) where {T} = T(randn(rng))
function Random.rand(rng::AbstractRNG, ::StdExponential{T, 0}) where {T}
    return T(rand(rng, Dists.Exponential()))
end
function Random.rand(rng::AbstractRNG, ::StdUniform{T, 0}) where {T}
    return T(rand(rng, Dists.Uniform()))
end
function Random.rand(rng::AbstractRNG, d::StdInverseGamma{T, <:Number, 0}) where {T}
    return T(rand(rng, Dists.InverseGamma(Float64(d.α), 1.0)))
end

function Dists._rand!(
        rng::AbstractRNG, ::StdExponential{T, N}, x::AbstractArray{<:Real, N}
    ) where {T, N}
    rd = Dists.Exponential()
    @inbounds for i in eachindex(x)
        x[i] = rand(rng, rd)
    end
    return x
end
function Dists._rand!(
        rng::AbstractRNG, ::StdUniform{T, N}, x::AbstractArray{<:Real, N}
    ) where {T, N}
    rd = Dists.Uniform()
    @inbounds for i in eachindex(x)
        x[i] = rand(rng, rd)
    end
    return x
end
function Dists._rand!(
        rng::AbstractRNG, d::StdInverseGamma{T, Tα, N}, x::AbstractArray{<:Real, N}
    ) where {T, Tα, N}
    @inbounds for i in eachindex(x)
        αi = _at(d.α, i)
        x[i] = rand(rng, Dists.InverseGamma(Float64(αi), 1.0))
    end
    return x
end

# Insupport
Dists.insupport(::StdExponential, x::Number) = x >= 0
Dists.insupport(::StdUniform, x::Number) = (0 <= x <= 1)
Dists.insupport(::StdInverseGamma, x::Number) = x > 0
function Dists.insupport(d::StdExponential, x::AbstractArray)
    return size(d) == size(x) && all(>=(0), x)
end
function Dists.insupport(d::StdUniform, x::AbstractArray)
    return size(d) == size(x) && all(xi -> 0 <= xi <= 1, x)
end
function Dists.insupport(d::StdInverseGamma, x::AbstractArray)
    return size(d) == size(x) && all(>(0), x)
end

# Scalar moments. Note: `srf.jl` defines `mean(::StdNormal) = zeros(size(d))`
# which returns a 0-dim Array for N = 0, so we override here for the scalar.
Dists.mean(::StdNormal{T, 0}) where {T} = zero(T)
Dists.var(::StdNormal{T, 0}) where {T} = one(T)
Dists.std(::StdNormal{T, 0}) where {T} = one(T)
Dists.mean(::StdExponential{T, 0}) where {T} = one(T)
Dists.var(::StdExponential{T, 0}) where {T} = one(T)
Dists.mean(::StdUniform{T, 0}) where {T} = T(0.5)
Dists.var(::StdUniform{T, 0}) where {T} = T(1 // 12)
function Dists.mean(d::StdInverseGamma{T, <:Real, 0}) where {T}
    return d.α > 1 ? T(1 / (d.α - 1)) : T(Inf)
end
function Dists.var(d::StdInverseGamma{T, <:Real, 0}) where {T}
    return d.α > 2 ? T(1 / ((d.α - 1)^2 * (d.α - 2))) : T(Inf)
end

# Scalar CDF / quantile (no `<:Real` constraint, so Reactant traced inputs work).
Dists.cdf(d::StdNormal{T, 0}, x::Number) where {T} = _std_cdf(d, x)
Dists.cdf(d::StdExponential{T, 0}, x::Number) where {T} = _std_cdf(d, x)
Dists.cdf(d::StdUniform{T, 0}, x::Number) where {T} = _std_cdf(d, x)
function Dists.cdf(d::StdInverseGamma{T, <:Number, 0}, x::Number) where {T}
    return _std_cdf(d, x)
end
Dists.quantile(d::StdNormal{T, 0}, p::Number) where {T} = _std_quantile(d, p)
Dists.quantile(d::StdExponential{T, 0}, p::Number) where {T} = _std_quantile(d, p)
Dists.quantile(d::StdUniform{T, 0}, p::Number) where {T} = _std_quantile(d, p)
function Dists.quantile(d::StdInverseGamma{T, <:Number, 0}, p::Number) where {T}
    return _std_quantile(d, p)
end

# asflat for Std bases. The N >= 1 `StdNormal` form already lives in `srf.jl`.
HC.asflat(::StdNormal{T, 0}) where {T} = TV.asℝ
HC.asflat(::StdExponential{T, 0}) where {T} = TV.asℝ₊
HC.asflat(::StdUniform{T, 0}) where {T} = TV.as(Real, 0.0, 1.0)
HC.asflat(::StdInverseGamma{T, <:Number, 0}) where {T} = TV.asℝ₊
HC.asflat(d::StdExponential{T, N}) where {T, N} = TV.as(Array, TV.asℝ₊, size(d)...)
HC.asflat(d::StdUniform{T, N}) where {T, N} = TV.as(Array, TV.as(Real, 0.0, 1.0), size(d)...)
HC.asflat(d::StdInverseGamma{T, <:Any, N}) where {T, N} = TV.as(Array, TV.asℝ₊, size(d)...)
