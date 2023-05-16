using SparseArrays
using LinearAlgebra
using ComradeBase

export GaussMarkovRF, GMRFCache

"""
    $(TYPEDEF)

A cache for a Gaussian Markov random field. This cache is useful
when defining a hierarchical prior for the GRMHD.

# Fields
$(FIELDS)

# Examples
```julia
julia> mean = zeros(2,2) # define a zero mean
julia> cache = GRMFCache(mean)
julia> prior_map(x) = GaussMarkovRF(mean, x[1], x[2], cache)
julia> d = HierarchicalPrior(prior_map, product_distribution([Uniform(-5.0,5.0), Uniform(0.0, 10.0)]))
julia> x0 = rand(d)
```
"""
struct GMRFCache{A, TD, M}
    """
    Intrinsic Gaussian Random Field pseudo-precison matrix.
    This is similar to the TSV regularizer
    """
    Λ::A
    """
    Gaussian Random Field diagonal precision matrix.
    This is similar to the L2-regularizer
    """
    D::TD
    """
    The eigenvalues of the Λ matrix which is needed to compute the
    log-normalization constant of the GMRF.
    """
    λQ::M
end

"""
    GMRFCache(mean::AbstractMatrix)

Contructs the [`GMRFCache`](@ref) from the mean image `mean`.
This is useful for hierarchical priors where you change the hyperparameters
of the [`GaussMarkovRF`](@ref), λ and `Σ`.
"""
function GMRFCache(mean::AbstractMatrix)
    dims = size(mean)

    # build the 1D correlation matrices
    q1 = build_q1d(mean, dims[1])
    q2 = build_q1d(mean, dims[2])

    # We do this ordering because Julia like column major
    Λ = kron(q2, I(dims[1])) + kron(I(dims[2]), q1)
    D = Diagonal(ones(eltype(Λ), size(Λ,1)))
    λQ = eigenvals(dims)
    return GMRFCache(Λ, D, λQ)
end

"""
    GMRFCache(m::AbstractModel, grid::AbstractDims; transform=identity)

Create a GMRF cache out of a `Comrade` model as the mean image.

# Arguments
 - `m`: Comrade model used for the mean image.
 - `grid`: The grid of points you want to create the model image on.

# Keyword arguments
 - `transform = identity`: A transform to apply to the image when creating the mean image. See the examples


# Example

```julia-repl
julia> m = GMRFCache(Gaussian(), imagepixels(10.0, 10.0, 64, 64))
julia> m = GMRFCache(Gaussian(), imagepixels(10.0, 10.0, 64, 64); transform=alr)
```
"""
function GMRFCache(m::ComradeBase.AbstractModel, grid::ComradeBase.AbstractDims; transform=identity)
    img = intensitymap(m, grid)
    return GMRFCache(transform(baseimage(img)))
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


"""
    $(TYPEDEF)

A image prior based off of the first-order Gaussian Markov random field.
This is similar to the combination of L₂ and TSV regularization and is equal to

    λ TSV(I-M) + Σ⁻¹L₂(I-M) + lognorm(λ, Σ)

where λ and Σ are given below and `M` is the mean image and `lognorm(λ,Σ)` is the
log-normalization of the random field and is needed to jointly infer `I` and the
hyperparameters λ, Σ.

# Fields
$(FIELDS)

# Examples

```julia
julia> mimg = zeros(6, 6) # The mean image
julia> λ, Σ = 2.0, 1.0
julia> d = GaussMarkovRF(mimg, λ, Σ)
julia> cache = GMRFCache(mimg) # now instead construct the cache
julia> d2 = GaussMarkovRF(mimg, λ, Σ, cache)
julia> invcov(d) ≈ invcov(d2)
true
```
"""
struct GaussMarkovRF{T,M<:AbstractMatrix{T},P,C,TDi} <: Dists.ContinuousMatrixDistribution
    """
    The mean image of the Gaussian Markov random field
    """
    m::M
    """
    The pixel correlation function of the random field. This is the
    TSV hyperparameter and controls the pixel-to-pixel correlation.
    As λ→0 this become a white-noise process.
    """
    λ::P
    """
    The variance of each pixel of the random field. As Σ→∞ we revert to a
    pure TSV prior, which is an improper or intrinsic Gaussian Markov random field.
    """
    Σ::P
    """
    The internal cache of the process. See [`GMRFCache`](@ref) for more information
    """
    cache::C
    """
    The dimensions of the image.
    """
    dims::TDi
end

Base.size(d::GaussMarkovRF)  = size(d.m)
Dists.mean(d::GaussMarkovRF) = d.m
Dists.cov(d::GaussMarkovRF)  = inv(Array(Dists.invcov(d)))


HC.asflat(d::GaussMarkovRF) = TV.as(Matrix, size(d)...)

"""
    GaussMarkovRF(mean::AbstractMatrix, λ, Σ)

Constructs a first order Gaussian Markov random field with mean image
`mean` and correlation `λ` and diagonal covariance `Σ`.
"""
function GaussMarkovRF(mean::AbstractMatrix, λ, Σ)
    cache = GMRFCache(mean)
    dims = size(mean)
    return GaussMarkovRF(mean, λ, Σ, cache, dims)
end

"""
    GaussMarkovRF(mean::ComradeBase.AbstractModel, grid::ComradeBase.AbstractDims, λ, Σ [,cache]; transform=identity)

Create a `GaussMarkovRF` object using a ComradeBase model.

# Arguments
 - `mean`: A ComradeBase model that will define the mean image
 - `grid`: The grid on which the image of the model will be created. This calls `ComradeBase.intensitymap`.
 - `λ`: The correlation length of the GMRF
 - `Σ`: The variance of the GMRF
 - `cache`: Optionally specify the precomputed GMRFCache

# Keyword Arguments
- `transform = identity`: A transform to apply to the image when creating the mean image. See the examples.

# Examples
```julia
julia> m1 = GaussMarkovRF(Gaussian(), imagepixels(10.0, 10.0, 128, 128), 5.0, 1.0; transform=alr)
julia> cache = GMRFCache(Gaussian(), imagepixels(10.0, 10.0, 128, 128), 5.0, 1.0; transform=alr)
julia> m2 = GaussMarkovRF(Gaussian(), imagepixels(10.0, 10.0, 128, 128), 5.0, 1.0, cache; transform=alr)
julia> m1 == m2
true
```

"""
function GaussMarkovRF(mean::ComradeBase.AbstractModel, grid::ComradeBase.AbstractDims, args...; transform=identity)
    img = intensitymap(mean, grid)
    return GaussMarkovRF(transform(baseimage(img)), args...)
end

"""
    GaussMarkovRF(mean::AbstractMatrix, λ, Σ, cache::GMRFCache)

Constructs a first order Gaussian Markov random field with mean image
`mean` and correlation `λ` and diagonal covariance `Σ` and the precomputed GMRFCache `cache`.
"""
GaussMarkovRF(mean::AbstractMatrix, λ, Σ, cache::GMRFCache) = GaussMarkovRF(mean, λ, Σ, cache, size(mean))

function Dists._rand!(rng::AbstractRNG, d::GaussMarkovRF, x::AbstractMatrix{<:Real})
    Q = Dists.invcov(d)
    cQ = cholesky(Q)
    z = randn(rng, length(x))
    x .= Dists.mean(d) .+ reshape(cQ\z, size(d))
end

Dists.insupport(::GaussMarkovRF, x::AbstractMatrix) = true


function eigenvals(dims)
    m, n = dims
    ix = 0:(m-1)
    iy = 0:(n-1)
    # Eigenvalues are the Σᵢⱼcos2π(ii'/m + jj'/n)
    return @. (4 - 2*cos(2π*ix/m) - 2*cos(2π*iy'/n))
end


function Dists._logpdf(d::GaussMarkovRF, x::AbstractMatrix{<:Real})
    return unnormed_logpdf(d, x) + lognorm(d)
end


function lognorm(d::GaussMarkovRF)
    (;λ, Σ) = d
    N = length(d)
    sum(log, (inv(d.Σ)/(π)*d.λ)*(inv(λ^2) .+ d.cache.λQ))/2 - Dists.log2π*N/2
    # sum(log, (inv(Σ) .+ λ*d.cache.λQ))/2 - Dists.log2π*N/2
end

function build_q1d(mean, n)
    d1 = similar(mean, n)
    fill!(d1, 2)
    d1[begin] = d1[end] = 2
    Q = spdiagm(-1=>fill(-oneunit(eltype(d1)), n-1), 0=>d1, 1=>fill(-oneunit(eltype(d1)), n-1), (n-1)=>[-oneunit(eltype(d1))], -(n-1)=>[-oneunit(eltype(d1))])
    return Q
end

function Dists.invcov(d::GaussMarkovRF)
    # return (d.λ*d.cache.Λ .+ d.Σ*d.cache.D)
    return (inv(d.Σ)/(π)).*(d.cache.Λ .+ inv(d.λ^2).*d.cache.D)
end

function unnormed_logpdf(d::GaussMarkovRF, I::AbstractMatrix)
    (;λ, Σ) = d
    ΔI = d.m - I
    s = igrmf_1n(ΔI)
    return -(inv(Σ)/(π)*d.λ)*(s + inv(λ^2)*sum(abs2,ΔI))/2
    # return -(λ*s + inv(Σ)*sum(abs2,ΔI))/2
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
        ΔIx = I[begin, iy] - I[ix,iy]
    end

    if iy < lastindex(I, 2)
         @inbounds ΔIy = I[ix, iy+1] - I[ix, iy]
    else
        ΔIy = I[ix, begin] - I[ix,iy]
    end

    ΔI2 = ΔIx^2 + ΔIy^2
    return ΔI2
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
    else
        @inbounds grad += -2*(I[begin, j1] - I[i1, j1])
    end


    # For ΔIy = I[i,j+1] - I[i,j]
    if j2 < ny + 1
        @inbounds grad += -2 * (I[i1, j2] - I[i1, j1])
    else
        @inbounds grad += -2*(I[i1, begin] - I[i1, j1])
    end

    # For ΔIx = I[i,j] - I[i-1,j]
    if i0 > 0
        @inbounds grad += 2 * (I[i1, j1] - I[i0, j1])
    else
        @inbounds grad += 2*(I[i1, j1] - I[end, j1])
    end

    # For ΔIy = I[i,j] - I[i,j-1]
    if j0 > 0
        @inbounds grad += 2 * (I[i1, j1] - I[i1, j0])
    else
        @inbounds grad += 2*(I[i1, j1] - I[i1, end])
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
