using SparseArrays
using LinearAlgebra

export GaussMarkovRF, GMRFCache


struct GMRFCache{A, TD, M}
    Λ::A
    D::TD
    λQ::M
end

function GMRFCache(mean)
    dims = size(mean)

    # build the 1D correlation matrices
    q1 = build_q1d(mean, dims[1])
    q2 = build_q1d(mean, dims[2])

    # We do this ordering because Julia like column major
    Λ = kron(q2, I(dims[1])) + kron(I(dims[2]), q1)
    D = spdiagm(ones(eltype(Λ), size(Λ,1)))
    λQ = eigenvals(dims)
    return GMRFCache(Λ, D, λQ)
end

# struct IGRMFCache{A, TD, M, C}
#     Λ::A
#     D::TD
#     λQ::M
# end

# function IGMRFCache(mean)
#     dims = size(mean)

#     # build the 1D correlation matrices
#     q1 = build_q1d(mean, dims[1])
#     q2 = build_q1d(mean, dims[2])

#     Λ = kron(q1, I(dims[2])) + kron(I(dims[1]), q2)
#     D = cholesky(Λ)
#     λQ = eigenvals(dims)
#     return GMRFCache(Λ, D, λQ)
# end

# struct IGaussMarkovRF{T,M<:AbstractMatrix{T},P,C,TDi} <: Dists.ContinuousMatrixDistribution
#     m::M
#     λ::P
#     cache::C
#     dims::TDi
# end

# Base.size(d::IGaussMarkovRF)  = size(d.m)
# Dists.mean(d::IGaussMarkovRF) = d.m
# Dists.cov(d::IGaussMarkovRF)  = inv(Array(Dists.invcov(d)))
# HC.asflat(d::IGaussMarkovRF) = TV.as(Matrix, size(d)...)

# function IGaussMarkovRF(mean::AbstractMatrix, λ, κ)
#     cache = IGMRFCache(mean)
#     dims = size(mean)
#     return IGaussMarkovRF(mean, λ, cache, dims)
# end

# IGaussMarkovRF(mean::AbstractMatrix, λ, cache::GMRFCache) = IGaussMarkovRF(mean, λ, cache, size(mean))

# function Dists._rand!(rng::AbstractRNG, d::IGaussMarkovRF, x::AbstractMatrix{<:Real})
#     Q = Dists.invcov(d)
#     cQ = cholesky(Q)
#     z = randn(rng, length(x))
#     x .= Dists.mean(d) .+ reshape(cQ\z, size(d))
# end

# Dists.insupport(::GaussMarkovRF, x::AbstractMatrix) = true



struct GaussMarkovRF{T,M<:AbstractMatrix{T},P,C,TDi} <: Dists.ContinuousMatrixDistribution
    m::M
    λ::P
    κ::P
    cache::C
    dims::TDi
end

Base.size(d::GaussMarkovRF)  = size(d.m)
Dists.mean(d::GaussMarkovRF) = d.m
Dists.cov(d::GaussMarkovRF)  = inv(Array(Dists.invcov(d)))


HC.asflat(d::GaussMarkovRF) = TV.as(Matrix, size(d)...)

function GaussMarkovRF(mean::AbstractMatrix, λ, κ)
    cache = GMRFCache(mean)
    dims = size(mean)
    return GaussMarkovRF(mean, λ, κ, cache, dims)
end

GaussMarkovRF(mean::AbstractMatrix, λ, κ, cache::GMRFCache) = GaussMarkovRF(mean, λ, κ, cache, size(mean))

function Dists._rand!(rng::AbstractRNG, d::GaussMarkovRF, x::AbstractMatrix{<:Real})
    Q = Dists.invcov(d)
    cQ = cholesky(Q)
    z = randn(rng, length(x))
    x .= Dists.mean(d) .+ reshape(cQ\z, size(d))
end

Dists.insupport(::GaussMarkovRF, x::AbstractMatrix) = true


function eigenvals(dims)
    m, n = dims
    ix = 1:m
    iy = 1:n
    return @. 2*(2 - cos(π*(ix - 1)/m)  - cos(π*(iy'-1)/n))
end


function Dists._logpdf(d::GaussMarkovRF, x::AbstractMatrix{<:Real})
    return unnormed_logpdf(d, x) + lognorm(d)
end


function lognorm(d::GaussMarkovRF)
    (;λ, κ) = d
    N = length(d)
    sum(log, 1 .+ (λ*κ).*d.cache.λQ)/2 - N*log(κ)/2 - Dists.log2π*N/2
end

function build_q1d(mean, n)
    d1 = similar(mean, n)
    fill!(d1, 2)
    d1[begin] = d1[end] = 1
    Q = spdiagm(-1=>fill(-oneunit(eltype(d1)), n-1), 0=>d1, 1=>fill(-oneunit(eltype(d1)), n-1))
    return Q
end

function Dists.invcov(d::GaussMarkovRF)
    (;λ, κ) = d
    return @. λ*d.cache.Λ + inv(κ)*d.cache.D
end

function unnormed_logpdf(d::GaussMarkovRF, I::AbstractMatrix)
    (;λ, κ) = d
    ΔI = d.m - I
    s = igrmf_1n(ΔI)
    return -(λ*s + inv(κ)*sum(abs2,ΔI))/2
end

# computes the intrinsic gaussian process of a 1-neighbor method
# this is equivalent to TSV regularizer
function igrmf_1n(I::AbstractMatrix)
    value = zero(eltype(I))
    for iy in axes(I,2), ix in axes(I,1)
        value = value + igrmf_1n_pixel(I, ix, iy)
    end
    return value
end


@inline function igrmf_1n_pixel(I::AbstractArray, ix::Integer, iy::Integer)
    if ix < lastindex(I, 1)
         @inbounds ΔIx = I[ix+1, iy] - I[ix, iy]
    else
        ΔIx = 0
    end

    if iy < lastindex(I, 2)
         @inbounds ΔIy = I[ix, iy+1] - I[ix, iy]
    else
        ΔIy = 0
    end

    return ΔIx^2 + ΔIy^2
end


@inline function igrmf_1n_grad_pixel(I::AbstractArray, ix::Integer, iy::Integer)
    nx = lastindex(I, 1)
    ny = lastindex(I, 2)

    i1 = ix
    j1 = iy
    i0 = i1 - 1
    j0 = j1 - 1
    i2 = i1 + 1
    j2 = j1 + 1

    grad = 0.0

    # For ΔIx = I[i+1,j] - I[i,j]
    if i2 < nx + 1
        @inbounds grad += -2 * (I[i2, j1] - I[i1, j1])
    end

    # For ΔIy = I[i,j+1] - I[i,j]
    if j2 < ny + 1
        @inbounds grad += -2 * (I[i1, j2] - I[i1, j1])
    end

    # For ΔIx = I[i,j] - I[i-1,j]
    if i0 > 0
        @inbounds grad += +2 * (I[i1, j1] - I[i0, j1])
    end

    # For ΔIy = I[i,j] - I[i,j-1]
    if j0 > 0
        @inbounds grad += +2 * (I[i1, j1] - I[i1, j0])
    end

    return grad
end





function ChainRulesCore.rrule(::typeof(igrmf_1n), x::AbstractArray)
    y = igrmf_1n(x)
    px = ProjectTo(x)
    function pullback(Δy)
        f̄bar = NoTangent()
        xbar = @thunk(igrmf_1n_grad(x) .* Δy)
        return f̄bar, px(xbar)
    end
    return y, pullback
end


@inline function igrmf_1n_grad(I::AbstractArray)
    grad = similar(I)
    for iy in axes(I,2), ix in axes(I,1)
        @inbounds grad[ix, iy] = igrmf_1n_grad_pixel(I, ix, iy)
    end
    return grad
end
