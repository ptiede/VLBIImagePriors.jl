export MarkovRandomFieldGraph

#=

Stores the graph for a Markov random field. This stores the dependency graph
for neighbors for a Markov Random field. The Markov random field implemented is to match
the 2D Matern process albeit restricted integer powers smoothness paramters.

Note the first order process is not actually a Matern process but rather a de Wiij process
and is related to a 2D random walk on a lattice. Higher order processes are build from the
first order process.



# Examples
```julia
julia> mean = zeros(2,2) # define a zero mean
julia> cache = MarkovRandomFieldGraph(mean)
julia> prior_map(ρ) = GaussMarkovRandomField(ρ, cache)
julia> d = HierarchicalPrior(prior_map, product_distribution([Uniform(0.0, 10.0)]))
julia> x0 = rand(d)
```

## Warning

Currently only the first and second order processes are efficient. The others still scale better
than the usual Gaussian process but they have not been optimized to the same extent.

=#
struct MarkovRandomFieldGraph{Order, A, TD, M}
    """
    Intrinsic Gaussian Random Field pseudo-precison matrix.
    This is similar to the TSV regularizer
    """
    G::A
    """
    Gaussian Random Field diagonal precision matrix.
    This is similar to the L2-regularizer
    """
    D::TD
    """
    The eigenvalues of the G matrix which is needed to compute the
    log-normalization constant of the GMRF.
    """
    λQ::M

    """
        MarkovRandomFieldGraph([T=Float64], dims::Dims{2}; order=1)

    Constructs the graph for a `order` Markov Random Field with dimension `dims`.
    The first optional argument specifies the type used for the internal graph structure
    usually given by a sparse matrix of type `T`.

    The `order` keyword argument specifies the order of the Markov random process. The default
    is first order which is a de Wiij process and it equivalent to TSV and L₂ regularizers from
    RML imaging.

    """
    function MarkovRandomFieldGraph(T::Type{<:Number}, dims::Dims{2}; order::Integer=1)
        order < 1 && ArgumentError("`order` parameter must be greater than or equal to 1, not $order")

        # build the 1D correlation matrices
        q1 = build_q1d(T, dims[1])
        q2 = build_q1d(T, dims[2])

        # We do this ordering because Julia like column major
        G = kron(q2, I(dims[1])) + kron(I(dims[2]), q1)
        D = Diagonal(ones(eltype(G), size(G,1)))
        λQ = eigenvals(T, dims)
        return new{order, typeof(G), typeof(D), typeof(λQ)}(G, D, λQ)
    end

end



Base.size(c::MarkovRandomFieldGraph) = size(c.λQ)


MarkovRandomFieldGraph(dims::Dims{2}; order::Integer=1) = MarkovRandomFieldGraph(Float64, dims; order)
MarkovRandomFieldGraph(img::AbstractMatrix{T}; order::Integer=1) where {T} = MarkovRandomFieldGraph(T, size(img); order)


"""
    MarkovRandomFieldGraph(grid::AbstractRectiGrid; order=1)
    MarkovRandomFieldGraph(img::AbstractMatrix; order=1)

Create a `order` Markov random field using the `grid` or `image` as its dimension.

The `order` keyword argument specifies the order of the Markov random process and is generally
given by the precision matrix

    Qₙ = τ(κI + G)ⁿ

where `n = order`, I is the identity matrix, G is specified by the first order stencil

    .  -1  .
    -1  4  -1
    .   4  .

κ is the Markov process hyper-parameters. For `n=1` κ is related to the correlation length
ρ of the random field by

    ρ = 1/κ

while for `n>1` it is given by

    ρ = √(8(n-1))/κ

Note that κ isn't set in the `MarkovRandomFieldGraph`, but rather once the noise process is
set, i.e. one of the subtypes of [`MarkovRandomField`](@ref).

Finally τ is chosen so that the marginal variance of the random field is unity. For `n=1`

    τ = 1

for `n=2`

    τ = 4πκ²

and for `n>2` we have

    τ = (N+1)4π κ²⁽ⁿ⁺¹⁾

# Example

```julia-repl
julia> m = MarkovRandomFieldGraph(imagepixels(10.0, 10.0, 64, 64))
julia> ρ = 10 # correlation length
julia> d = GaussMarkovRandomField(ρ, m) # build the Gaussian Markov random field
```
"""
function MarkovRandomFieldGraph(grid::ComradeBase.AbstractRectiGrid; order::Integer=1)
    return MarkovRandomFieldGraph(eltype(grid.X), size(grid); order)
end

function Base.show(io::IO, d::MarkovRandomFieldGraph{O}) where {O}
    println(io, "MarkovRandomFieldGraph{$O}(")
    println(io, "dims: ", size(d))
    print(io, ")")
end

function κ(ρ, ::Val{1})
    return inv(ρ)
end

function κ(ρ, ::Val{N}) where {N}
    return sqrt(oftype(ρ, 8*(N-1)))*inv(ρ)
end


# Compute the square manoblis distance or the <x,Qx> inner product.
function sq_manoblis(::MarkovRandomFieldGraph{1}, ΔI::AbstractMatrix, ρ)
    κ² = κ(ρ, Val(1))^2
    return igmrf_1n(ΔI, κ²)
end

function sq_manoblis(::MarkovRandomFieldGraph{2}, ΔI::AbstractMatrix, ρ)
    κ² = κ(ρ, Val(2))^2
    return igmrf_2n(ΔI, κ²)/mrfnorm(κ², Val(2))
end

function sq_manoblis(d::MarkovRandomFieldGraph{N}, ΔI::AbstractMatrix, ρ) where {N}
    κ² = κ(ρ, Val(N))^2
    return dot(ΔI, (κ²*d.D + d.G)^(N), vec(ΔI))/mrfnorm(κ², Val(N))
end

@inline function LinearAlgebra.logdet(d::MarkovRandomFieldGraph{N}, ρ) where {N}
    κ² = κ(ρ, Val(N))^2
    a = zero(eltype(d.λQ))
    @fastmath @simd for i in eachindex(d.λQ)
        a += log(κ² + d.λQ[i])
    end
    return N*a - length(d.λQ)*log(mrfnorm(κ², Val(N)))
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


function scalematrix(d::MarkovRandomFieldGraph{N}, ρ) where {N}
    κ² = κ(ρ, Val(N))^2
    return (d.G .+ d.D.*κ²)^N/mrfnorm(κ², Val(N))
end

function eigenvals(T, dims)
    m, n = dims
    ix = 1:m
    iy = 1:n
    # Eigenvalues are the Σᵢⱼcos2π(ii'/m + jj'/n)
    return @. 4 + 2*cos(convert(T, π)*ix/(m+1)) + 2*cos(convert(T, π)*iy'/(n+1))
    # return @. (4 - 2*cos(2π*ix/m) - 2*cos(2π*iy'/n))
end

# function eigenvals(dims)
#     m, n = dims
#     ix = 0:(m-1)
#     iy = 0:(n-1)
#     # Eigenvalues are the Σᵢⱼcos2π(ii'/m + jj'/n)
#     return
#     return @. (4 - 2*cos(2π*ix/m) - 2*cos(2π*iy'/n))
# end



function build_q1d(T, n)
    d1 = Vector{T}(undef, n)
    fill!(d1, 2)
    d1[begin] = d1[end] = 2
    Q = spdiagm(-1=>fill(-oneunit(eltype(d1)), n-1), 0=>d1, 1=>fill(-oneunit(eltype(d1)), n-1))#, (n-1)=>[-oneunit(eltype(d1))], -(n-1)=>[-oneunit(eltype(d1))])
    return Q
end


function igmrf_2n(I::AbstractMatrix, κ²)
    value = zero(eltype(I))
    for iy in axes(I, 2), ix in axes(I,1)
        value = value + igmrf_qv(I, κ², ix, iy)^2
    end
    return value
end

@inline function igmrf_2n_pixel(I::AbstractArray, κ², ix::Integer, iy::Integer)
    value = (4 + κ²)*I[ix, iy]
    ΔIx  = ix < lastindex(I, 1)  ? I[ix+1, iy] : zero(eltype(I))
    ΔIx += ix > firstindex(I, 1) ? I[ix-1, iy] :

    ΔIy  = iy < lastindex(I, 2)  ? I[ix, iy+1] : I[ix, begin]
    ΔIy += iy > firstindex(I, 2) ? I[ix, iy-1] : I[ix, end]

    value = value - ΔIx - ΔIy
    return value*value
end



# computes the intrinsic gaussian process of a 1-neighbor method
# this is equivalent to TSV regularizer
function igmrf_1n(I::AbstractMatrix, κ²)
    value = zero(eltype(I))
    for iy in axes(I,2), ix in axes(I,1)
        value = value + igmrf_qv(I, κ², ix, iy)*I[ix, iy]
    end
    return value
end


@inline @inbounds function igmrf_qv(I::AbstractMatrix, κ², ix::Integer, iy::Integer)
    value = (4 + κ²)*I[ix, iy]
    if ix < lastindex(I, 1)
         value -= I[ix+1, iy]
    end

    if iy < lastindex(I, 2)
         value -= I[ix, iy+1]
    end

    if ix > firstindex(I, 1)
        value -= I[ix-1, iy]
    end

   if iy > firstindex(I, 2)
        value -= I[ix, iy-1]
   end

    return value
end


# @inline function igmrf_1n_grad_pixel(I::AbstractArray, ix::Integer, iy::Integer)
#     nx = lastindex(I, 1)
#     ny = lastindex(I, 2)

#     i1 = ix
#     j1 = iy
#     i0 = i1 - 1
#     j0 = j1 - 1
#     i2 = i1 + 1
#     j2 = j1 + 1

#     grad = 0.0

#     # For ΔIx = I[i+1,j] - I[i,j]
#     if i2 < nx + 1
#         @inbounds grad += -2 * (I[i2, j1] - I[i1, j1])
#     else
#         @inbounds grad += -2*(I[begin, j1] - I[i1, j1])
#     end


#     # For ΔIy = I[i,j+1] - I[i,j]
#     if j2 < ny + 1
#         @inbounds grad += -2 * (I[i1, j2] - I[i1, j1])
#     else
#         @inbounds grad += -2*(I[i1, begin] - I[i1, j1])
#     end

#     # For ΔIx = I[i,j] - I[i-1,j]
#     if i0 > 0
#         @inbounds grad += 2 * (I[i1, j1] - I[i0, j1])
#     else
#         @inbounds grad += 2*(I[i1, j1] - I[end, j1])
#     end

#     # For ΔIy = I[i,j] - I[i,j-1]
#     if j0 > 0
#         @inbounds grad += 2 * (I[i1, j1] - I[i1, j0])
#     else
#         @inbounds grad += 2*(I[i1, j1] - I[i1, end])
#     end

#     return grad
# end





# function ChainRulesCore.rrule(::typeof(igmrf_1n), x::AbstractArray)
#     y = igmrf_1n(x)
#     px = ProjectTo(x)
#     function pullback(Δy)
#         f̄bar = NoTangent()
#         xbar = @thunk(igmrf_1n_grad(x) .* Δy)
#         return f̄bar, px(xbar)
#     end
#     return y, pullback
# end


# @inline function igmrf_1n_grad(I::AbstractArray)
#     grad = similar(I)
#     for iy in axes(I,2), ix in axes(I,1)
#         @inbounds grad[ix, iy] = igmrf_1n_grad_pixel(I, ix, iy)
#     end
#     return grad
# end
