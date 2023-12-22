export MarkovRandomFieldCache

"""
    $(TYPEDEF)

A cache for a Markov random field.

# Fields
$(FIELDS)

# Examples
```julia
julia> mean = zeros(2,2) # define a zero mean
julia> cache = GRMFCache(mean)
julia> prior_map(x) = GaussMarkovRandomField(mean, x[1], x[2], cache)
julia> d = HierarchicalPrior(prior_map, product_distribution([Uniform(-5.0,5.0), Uniform(0.0, 10.0)]))
julia> x0 = rand(d)
```
"""
struct MarkovRandomFieldCache{A, TD, M}
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

struct ConditionalMarkov{B,C}
    cache::C
end

"""
    ConditionalMarkov(D, args...)

Creates a Conditional Markov measure, that behaves as a Julia functional. The functional
returns a probability measure defined by the arguments passed to the functional.

# Arguments

 - `D`: The base distribution or measure of the random field. Currently `Normal` and `TDist`
        are valid random fields
 - `args`: Additional arguments used to construct the Markov random field cache.
           See [`MarkovRandomFieldCache`](@ref) for more information.

# Example
```julia-repl
julia> grid = imagepixels(10.0, 10.0, 64, 64)
julia> ℓ = ConditionalMarkov(Normal, grid)
julia> d = ℓ(16) # This is now a distribution
julia> rand(d)
```
"""
function ConditionalMarkov(D::Type{<:Union{Dists.Normal, Dists.TDist, Dists.Exponential}}, args...)
    c = MarkovRandomFieldCache(args...)
    return ConditionalMarkov{D, typeof(c)}(c)
end


(c::ConditionalMarkov{<:Dists.Normal})(ρ)     = GaussMarkovRandomField(ρ, c.cache)
(c::ConditionalMarkov{<:Dists.TDist})(ρ, ν=1)   = TDistMarkovRandomField(ρ, ν, c.cache)


Base.size(c::MarkovRandomFieldCache) = size(c.λQ)

"""
    MarkovRandomFieldCache(mean::AbstractMatrix)

Contructs the [`MarkovRandomFieldCache`](@ref) cache.
"""
function MarkovRandomFieldCache(T::Type{<:Number}, dims::Dims{2})

    # build the 1D correlation matrices
    q1 = build_q1d(T, dims[1])
    q2 = build_q1d(T, dims[2])

    # We do this ordering because Julia like column major
    Λ = kron(q2, I(dims[1])) + kron(I(dims[2]), q1)
    D = Diagonal(ones(eltype(Λ), size(Λ,1)))
    λQ = eigenvals(dims)
    return MarkovRandomFieldCache(Λ, D, λQ)
end
MarkovRandomFieldCache(img::AbstractMatrix{T}) where {T} = MarkovRandomFieldCache(T, size(img))


"""
    MarkovRandomFieldCache(grid::AbstractDims)

Create a GMRF cache out of a `Comrade` model as the mean image.

# Arguments
 - `m`: Comrade model used for the mean image.
 - `grid`: The grid of points you want to create the model image on.

# Keyword arguments
 - `transform = identity`: A transform to apply to the image when creating the mean image. See the examples


# Example

```julia-repl
julia> m = MarkovRandomFieldCache(imagepixels(10.0, 10.0, 64, 64))
```
"""
function MarkovRandomFieldCache(grid::ComradeBase.AbstractDims)
    return MarkovRandomFieldCache(eltype(grid.X), size(grid))
end


# Compute the square manoblis distance or the <x,Qx> inner product.
function sq_manoblis(::MarkovRandomFieldCache, ΔI::AbstractMatrix, ρ)
    s = igrmf_1n(ΔI)
    return (s + inv(ρ^2)*sum(abs2, ΔI))
end

function LinearAlgebra.logdet(d::MarkovRandomFieldCache, ρ)
    return sum(log, (inv(ρ^2) .+ d.λQ))
end

Dists.invcov(d::MarkovRandomFieldCache, ρ) =  (d.Λ .+ d.D.*inv(ρ^2))

function eigenvals(dims)
    m, n = dims
    ix = 0:(m-1)
    iy = 0:(n-1)
    # Eigenvalues are the Σᵢⱼcos2π(ii'/m + jj'/n)
    return @. (4 - 2*cos(2π*ix/m) - 2*cos(2π*iy'/n))
end


function build_q1d(T, n)
    d1 = Vector{T}(undef, n)
    fill!(d1, 2)
    d1[begin] = d1[end] = 2
    Q = spdiagm(-1=>fill(-oneunit(eltype(d1)), n-1), 0=>d1, 1=>fill(-oneunit(eltype(d1)), n-1), (n-1)=>[-oneunit(eltype(d1))], -(n-1)=>[-oneunit(eltype(d1))])
    return Q
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
        @inbounds ΔIy = I[ix, begin] - I[ix,iy]
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
