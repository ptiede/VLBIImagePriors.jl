# StdNormal — standard zero-mean unit-variance normal of arbitrary shape.
# Pre-existed in `src/srf.jl` (used by `StationaryRandomField`); we moved it
# here for consistency with the other Std bases. The struct itself is
# unchanged so existing references in `srf.jl` and `markovrf/gmrf.jl` remain
# valid.

struct StdNormal{T, N} <: Dists.ContinuousDistribution{Dists.ArrayLikeVariate{N}}
    dims::Dims{N}
end
StdNormal(d::Dims{N}) where {N} = StdNormal{Float64, N}(d)
StdNormal(d::Int...) = StdNormal(d)

Base.size(d::StdNormal) = d.dims
Base.length(d::StdNormal) = prod(d.dims)
Base.eltype(::StdNormal{T}) where {T} = T
Dists.insupport(::StdNormal, ::Number) = true
Dists.insupport(::StdNormal, ::Real) = true
Dists.insupport(::StdNormal, x::AbstractArray) = true


# ----- log-pdf split ------------------------------------------------------
# `unnormed_logpdf(d, x)` returns only the data-dependent part; `lognorm(d)`
# returns the constant. `logpdf = unnormed_logpdf + lognorm`.

@inline _unnormed_kernel(::StdNormal, z, _) = -z * z / 2

# `sum(abs2, z)` is non-allocating on CPU and Reactant supports the
# mapreduce form (see existing test at `test/reactant.jl` / `srf.jl:412`).
@inline _unnormed_kernel_sum(::StdNormal, z) = -sum(abs2, z) / 2

unnormed_logpdf(d::StdNormal{T, 0}, x::Number) where {T} = _unnormed_kernel(d, x, 1)
function unnormed_logpdf(d::StdNormal{T, N}, x::AbstractArray{<:Number, N}) where {T, N}
    return _unnormed_kernel_sum(d, x)
end

@inline lognorm(d::StdNormal) = -length(d) * oftype(zero(eltype(d)), log(2π) / 2)


# ----- Distributions interface --------------------------------------------

Dists.logpdf(d::StdNormal{T, 0}, x::Number) where {T} = unnormed_logpdf(d, x) + lognorm(d)

# Three-method pattern. `<:Number` is the workhorse (covers Reactant traced
# eltypes); `<:Real` is required to break ambiguity with Distributions'
# fallback `logpdf(::Distribution{ArrayLikeVariate{N}}, ::AbstractArray{<:Real, M})`
# at `Distributions/.../common.jl:261`. Without the `<:Real` override Julia
# emits an ambiguity error for `Matrix{Float64}` inputs (verified empirically).
function Dists._logpdf(d::StdNormal{T, N}, x::AbstractArray{<:Number, N}) where {T, N}
    return unnormed_logpdf(d, x) + lognorm(d)
end
function Dists.logpdf(d::StdNormal{T, N}, x::AbstractArray{<:Real, N}) where {T, N}
    return unnormed_logpdf(d, x) + lognorm(d)
end
function Dists.logpdf(d::StdNormal{T, N}, x::AbstractArray{<:Number, N}) where {T, N}
    return unnormed_logpdf(d, x) + lognorm(d)
end


# ----- sampling -----------------------------------------------------------

Random.rand(rng::AbstractRNG, ::StdNormal{T, 0}) where {T} = T(randn(rng))
function Dists._rand!(
        rng::AbstractRNG, ::StdNormal{T, N}, x::AbstractArray{<:Real, N}
    ) where {T, N}
    return randn!(rng, x)
end


# ----- moments ------------------------------------------------------------

Dists.mean(::StdNormal{T, 0}) where {T} = zero(T)
Dists.var(::StdNormal{T, 0}) where {T} = one(T)
Dists.std(::StdNormal{T, 0}) where {T} = one(T)
Dists.mean(d::StdNormal) = zeros(eltype(d), size(d))
Dists.var(d::StdNormal) = ones(eltype(d), size(d))
Dists.cov(d::StdNormal) = I(length(d))


# ----- cdf / quantile -----------------------------------------------------

@inline _std_cdf(::StdNormal, x) = (one(x) + erf(x / sqrt(oftype(x, 2)))) / 2
@inline _std_quantile(::StdNormal, p) = sqrt(oftype(p, 2)) * erfinv(2 * p - one(p))

Dists.cdf(d::StdNormal, x::Number) = _std_cdf(d, x)
Dists.quantile(d::StdNormal, p::Number) = _std_quantile(d, p)


# ----- transforms ---------------------------------------------------------

HC.asflat(::StdNormal{T, 0}) where {T} = TV.asℝ
HC.asflat(d::StdNormal) = TV.as(Array, size(d)...)
HC.ascube(d::StdNormal) = HC.ArrayHC(d)
HC.inverse_eltype(::StdNormal{T}, y::Type) where {T} = promote_type(T, eltype(y))

function HC._step_transform(h::HC.ArrayHC{<:StdNormal}, p::AbstractVector, index)
    d = h.dist
    out = Dists.quantile.(Ref(d), p)
    return out, index + HC.dimension(h)
end
function HC._step_inverse!(
        x::AbstractVector, index, h::HC.ArrayHC{<:StdNormal}, y::AbstractVector
    )
    d = h.dist
    x .= Dists.cdf.(Ref(d), y)
    return index + HC.dimension(h)
end
