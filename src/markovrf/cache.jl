export MarkovRandomFieldCache

"""
    $(TYPEDEF)

Stores the graph for a Markov random field. This stores the dependency graph
for neighbors for a Markov Random field. The Markov random field implemented is
near identical to 2D Matern process albeit with integer powers of the power spectrum.
Note the first order process is not actually a Matern process but rather a de Wiij process
and is related to a 2D random walk on a lattice. Higher order processes are build from the
first order process.


# Fields
$(FIELDS)

# Examples
```julia
julia> mean = zeros(2,2) # define a zero mean
julia> cache = MarkovRandomFieldCache(mean)
julia> prior_map(ρ) = GaussMarkovRandomField(ρ, cache)
julia> d = HierarchicalPrior(prior_map, product_distribution([Uniform(0.0, 10.0)]))
julia> x0 = rand(d)
```

## Warning

Currently only the first and second order processes are efficient. The others still scale better
than the usual Gaussian process but they have not been optimized to the same extent.

"""
struct MarkovRandomFieldCache{Order, A, TD, M}
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

    """
        MarkovRandomFieldCache([T=Float64], dims::Dims{2}; order=1)

    Constructs the cache for a `order` Markov Random Field with dimension `dims`.
    The first optional argument specifies the type used for the internal graph structure
    usually given by a sparse matrix of type `T`.

    The `order` keyword argument specifies the order of the Markov random process. The default
    is first order which is a de Wiij process and it equivalent to TSV and L₂ regularizers from
    RML imaging.

    """
    function MarkovRandomFieldCache(T::Type{<:Number}, dims::Dims{2}; order::Integer=1)
        order < 1 && ArgumentError("`order` parameter must be greater than or equal to 1, not $order")

        # build the 1D correlation matrices
        q1 = build_q1d(T, dims[1])
        q2 = build_q1d(T, dims[2])

        # We do this ordering because Julia like column major
        Λ = kron(q2, I(dims[1])) + kron(I(dims[2]), q1)
        D = Diagonal(ones(eltype(Λ), size(Λ,1)))
        λQ = eigenvals(dims)
        return new{order, typeof(Λ), typeof(D), typeof(λQ)}(Λ, D, λQ)
    end

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
function ConditionalMarkov(D::Type{<:Union{Dists.Normal, Dists.TDist, Dists.Exponential}}, args...; kwargs...)
    c = MarkovRandomFieldCache(args...; kwargs...)
    return ConditionalMarkov{D, typeof(c)}(c)
end




Base.size(c::MarkovRandomFieldCache) = size(c.λQ)


MarkovRandomFieldCache(dims::Dims{2}; order::Integer=1) = MarkovRandomField(Float64, dims; order)
MarkovRandomFieldCache(img::AbstractMatrix{T}; order::Integer=1) where {T} = MarkovRandomFieldCache(T, size(img); order)


"""
    MarkovRandomFieldCache(grid::AbstractDims; order=1)
    MarkovRandomFieldCache(img::AbstractMatrix; order=1)

Create a `order` Markov random field using the `grid` or `image` as its dimension.

The `order` keyword argument specifies the order of the Markov random process. The default
is first order which is a de Wiij process and it equivalent to TSV and L₂ regularizers from
RML imaging.

# Example

```julia-repl
julia> m = MarkovRandomFieldCache(imagepixels(10.0, 10.0, 64, 64))
```
"""
function MarkovRandomFieldCache(grid::ComradeBase.AbstractDims; order::Integer=1)
    return MarkovRandomFieldCache(eltype(grid.X), size(grid); order)
end

function κ(ρ, ::Val{1})
    return inv(ρ)
end

function κ(ρ, ::Val{N}) where {N}
    return sqrt(oftype(ρ, 8*(N-1)))*inv(ρ)
end


# Compute the square manoblis distance or the <x,Qx> inner product.
function sq_manoblis(::MarkovRandomFieldCache{1}, ΔI::AbstractMatrix, ρ)
    s = igrmf_1n(ΔI)
    κ² = κ(ρ, Val(1))^2
    return (s + κ²*sum(abs2, ΔI))
end

function sq_manoblis(::MarkovRandomFieldCache{2}, ΔI::AbstractMatrix, ρ)
    κ² = κ(ρ, Val(2))^2
    return igmrf_2n(ΔI, κ²)/mrfnorm(κ², Val(2))
end

function sq_manoblis(d::MarkovRandomFieldCache{N}, ΔI::AbstractMatrix, ρ) where {N}
    κ² = κ(ρ, Val(N))^2
    return dot(ΔI, (κ²*d.D + d.Λ)^(N), vec(ΔI))/mrfnorm(κ², Val(N))
end

function ChainRulesCore.rrule(::typeof(sq_manoblis), d::MarkovRandomFieldCache, ΔI, ρ)
    s = sq_manoblis(d, ΔI, ρ)
    prI = ProjectTo(ΔI)
    function _sq_manoblis_pullback(Δ)
        Δf = NoTangent()
        Δd = NoTangent()
        dI = zero(ΔI)

        ((_, _, dρ), ) = autodiff(Reverse, sq_manoblis, Active, Const(d), Duplicated(ΔI, dI), Active(ρ))

        dI .= Δ.*dI
        return Δf, Δd, prI(dI), Δ*dρ
    end
    return s, _sq_manoblis_pullback
end


function LinearAlgebra.logdet(d::MarkovRandomFieldCache{N}, ρ) where {N}
    κ² = κ(ρ, Val(N))^2
    a =  N*sum(d.λQ) do x
                log(κ² + x)
        end
    return a - length(d.λQ)*log(mrfnorm(κ², Val(N)))
end

# This is the σ to ensure we have a unit variance GMRF
function mrfnorm(::T, ::Val{1}) where {T<:Number}
    return one(T)
end


function mrfnorm(κ²::T, ::Val{2}) where {T<:Number}
    return convert(T, 4π)*κ²
end

function mrfnorm(κ²::T, ::Val{N}) where {T<:Number, N}
    return (N+1)*convert(T, 4π)*κ²^((N-1))
end


function Dists.invcov(d::MarkovRandomFieldCache{N}, ρ) where {N}
    κ² = κ(ρ, Val(N))^2
    return (d.Λ .+ d.D.*κ²)^N/mrfnorm(κ², Val(N))
end

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


function igmrf_2n(I::AbstractMatrix, κ²)
    value = zero(eltype(I))
    for iy in axes(I, 2), ix in axes(I,1)
        value = value + igrmf_2n_pixel(I, κ², ix, iy)
    end
    return value
end

@inline function igrmf_2n_pixel(I::AbstractArray, κ², ix::Integer, iy::Integer)
    value = (4 + κ²)*I[ix, iy]
    ΔIx  = ix < lastindex(I, 1)  ? I[ix+1, iy] : I[begin, iy]
    ΔIx += ix > firstindex(I, 1) ? I[ix-1, iy] : I[end, iy]

    ΔIy  = iy < lastindex(I, 2)  ? I[ix, iy+1] : I[ix, begin]
    ΔIy += iy > firstindex(I, 2) ? I[ix, iy-1] : I[ix, end]

    value = value - ΔIx - ΔIy
    return value*value
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
