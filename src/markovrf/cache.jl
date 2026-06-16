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
struct MarkovRandomFieldGraph{Order, A, TD, M, E}
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
    Executor for the prior evalution
    """
    executor::E

    """
        MarkovRandomFieldGraph([T=Float64], dims::Dims{2}; order=1, executor = Serial())

    Constructs the graph for a `order` Markov Random Field with dimension `dims`.
    The first optional argument specifies the type used for the internal graph structure
    usually given by a sparse matrix of type `T`.

    The `order` keyword argument specifies the order of the Markov random process. The default
    is first order which is a de Wiij process and it equivalent to TSV and L₂ regularizers from
    RML imaging.

    """
    function MarkovRandomFieldGraph(T::Type{<:Number}, dims::Dims{2}; order::Integer = 1, executor = Serial())
        order < 1 && ArgumentError("`order` parameter must be greater than or equal to 1, not $order")

        # build the 1D correlation matrices
        q1 = build_q1d(T, dims[1])
        q2 = build_q1d(T, dims[2])

        # We do this ordering because Julia like column major
        G = kron(q2, I(dims[1])) + kron(I(dims[2]), q1)
        D = Diagonal(ones(eltype(G), size(G, 1)))
        λQ = eigenvals(T, dims, executor)
        return new{order, typeof(G), typeof(D), typeof(λQ), typeof(executor)}(G, D, λQ, executor)
    end

end


Base.size(c::MarkovRandomFieldGraph) = size(c.λQ)


MarkovRandomFieldGraph(dims::Dims{2}; order::Integer = 1, executor = Serial()) = MarkovRandomFieldGraph(Float64, dims; order, executor)
MarkovRandomFieldGraph(img::AbstractMatrix{T}; order::Integer = 1, executor = Serial()) where {T} = MarkovRandomFieldGraph(T, size(img); order, executor)


"""
    MarkovRandomFieldGraph(grid::AbstractRectiGrid; order=1)
    MarkovRandomFieldGraph(img::AbstractMatrix; order=1)

Create a `order` Markov random field using the `grid` or `image` as its dimension.

The `order` keyword argument specifies the order of the Markov random process and is generally
given by the precision matrix

    Qₙ = τ(κ²I + G)ⁿ

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
function MarkovRandomFieldGraph(grid::ComradeBase.AbstractRectiGrid; order::Integer = 1)
    return MarkovRandomFieldGraph(eltype(grid.X), size(grid); order)
end

function Base.show(io::IO, d::MarkovRandomFieldGraph{O}) where {O}
    println(io, "MarkovRandomFieldGraph{$O}(")
    println(io, "dims: ", size(d))
    return print(io, ")")
end

function κ(ρ, ::Val{1})
    return inv(ρ)
end

function κ(ρ, ::Val{N}) where {N}
    return sqrt(oftype(ρ, 8 * (N - 1))) * inv(ρ)
end


# Compute the square manoblis distance or the <x,Qx> inner product.
function sq_manoblis(d::MarkovRandomFieldGraph{1}, ΔI::AbstractMatrix, ρ)
    κ² = κ(ρ, Val(1))^2
    return igmrf_1n(ΔI, κ², d.executor) / mrfnorm(d, κ²)
end

function sq_manoblis(d::MarkovRandomFieldGraph{2}, ΔI::AbstractMatrix, ρ)
    κ² = κ(ρ, Val(2))^2
    return igmrf_2n(ΔI, κ², d.executor) / mrfnorm(d, κ²)
end

function sq_manoblis(d::MarkovRandomFieldGraph{N}, ΔI::AbstractMatrix, ρ) where {N}
    κ² = κ(ρ, Val(N))^2
    return dot(ΔI, (κ² * d.D + d.G)^(N), vec(ΔI)) / mrfnorm(d, κ²)
end

__f(κ², x) = log(κ² + x)


@inline function LinearAlgebra.logdet(d::MarkovRandomFieldGraph{N}, ρ) where {N}
    κ² = κ(ρ, Val(N))^2
    f2 = Base.Fix1(__f, κ²)
    a = sum(f2, d.λQ)
    return N * a - length(d.λQ) * log(mrfnorm(d, κ²))
end


# TODO
# Figure out the actual narmalization from the lattice helmholtz decomposition
# using the specific boundary conditions I am using.

# This is the σ to ensure we have a unit variance GMRF
function mrfnorm(d::MarkovRandomFieldGraph{1}, κ²::T) where {T <: Number}
    return (κ² + 1) #Empirical rule
end


function mrfnorm(d::MarkovRandomFieldGraph{2}, k::T) where {T <: Number}
    Tπ = T(π)
    n, m = size(d)
    λ0 = 4 + 2 * cos(Tπ * n / (n + 1)) + 2 * cos(Tπ * m / (m + 1))
    return T(4π) * (k + λ0) #Empirical rule
end

function mrfnorm(d::MarkovRandomFieldGraph{N}, k::T) where {N, T <: Number}
    Tπ = T(π)
    n, m = size(d)
    λ0 = 4 + 2 * cos(Tπ * n / (n + 1)) + 2 * cos(Tπ * m / (m + 1))
    return T(4π) * (k + λ0)^(N - 1) #Empirical rule
end


function scalematrix(d::MarkovRandomFieldGraph{N}, ρ) where {N}
    κ² = κ(ρ, Val(N))^2
    return (d.G .+ d.D .* κ²)^N / mrfnorm(d, κ²)
end

function eigenvals(T, dims, ::Any)
    m, n = dims
    ix = m:-1:1 #Reverse the order to match the DST conventions
    iy = n:-1:1 #Reverse the order to match the DST conventions
    # Because G is a Kronecker product of two tri-diagonal matrices
    return @. 4 + 2 * cos(convert(T, π) * ix / (m + 1)) + 2 * cos(convert(T, π) * iy' / (n + 1))
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
    Q = spdiagm(-1 => fill(-oneunit(eltype(d1)), n - 1), 0 => d1, 1 => fill(-oneunit(eltype(d1)), n - 1)) #, (n-1)=>[-oneunit(eltype(d1))], -(n-1)=>[-oneunit(eltype(d1))])
    return Q
end


# Walks the grid in 9 unconditional regions (interior + 4 edges + 4 corners),
# computing the per-cell qv with the right number of in-bounds neighbors, and
# accumulating `f(qv, Ic)`. Used by both igmrf_1n (f = qv*Ic) and igmrf_2n
# (f = qv*qv). No `@trace track_numbers=false if` anywhere; all iterators are plain `UnitRange{Int}`
# bound before the loops.
@inline function _igmrf_walk(I::AbstractMatrix, κ², f::F) where {F}
    c = 4 + κ²
    nx, ny = size(I)
    xitr_inner = 2:(nx - 1)
    yitr_inner = 2:(ny - 1)
    value = zero(eltype(I))

    # Interior — 4 neighbors
    @trace track_numbers = false for iy in yitr_inner
        @trace track_numbers = false for ix in xitr_inner
            Ic = rgetindex(I, ix, iy)
            qv = c * Ic
            qv -= rgetindex(I, ix + 1, iy)
            qv -= rgetindex(I, ix - 1, iy)
            qv -= rgetindex(I, ix, iy + 1)
            qv -= rgetindex(I, ix, iy - 1)
            value += f(qv, Ic)
        end
    end

    # Top edge (iy = 1)
    @trace track_numbers = false for ix in xitr_inner
        Ic = rgetindex(I, ix, 1)
        qv = c * Ic
        qv -= rgetindex(I, ix + 1, 1)
        qv -= rgetindex(I, ix - 1, 1)
        qv -= rgetindex(I, ix, 2)
        value += f(qv, Ic)
    end

    # Bottom edge (iy = ny)
    @trace track_numbers = false for ix in xitr_inner
        Ic = rgetindex(I, ix, ny)
        qv = c * Ic
        qv -= rgetindex(I, ix + 1, ny)
        qv -= rgetindex(I, ix - 1, ny)
        qv -= rgetindex(I, ix, ny - 1)
        value += f(qv, Ic)
    end

    # Left edge (ix = 1)
    @trace track_numbers = false for iy in yitr_inner
        Ic = rgetindex(I, 1, iy)
        qv = c * Ic
        qv -= rgetindex(I, 2, iy)
        qv -= rgetindex(I, 1, iy + 1)
        qv -= rgetindex(I, 1, iy - 1)
        value += f(qv, Ic)
    end

    # Right edge (ix = nx)
    @trace track_numbers = false for iy in yitr_inner
        Ic = rgetindex(I, nx, iy)
        qv = c * Ic
        qv -= rgetindex(I, nx - 1, iy)
        qv -= rgetindex(I, nx, iy + 1)
        qv -= rgetindex(I, nx, iy - 1)
        value += f(qv, Ic)
    end

    # Four corners
    Ic = rgetindex(I, 1, 1)
    qv = c * Ic
    qv -= rgetindex(I, 2, 1)
    qv -= rgetindex(I, 1, 2)
    value += f(qv, Ic)

    Ic = rgetindex(I, nx, 1)
    qv = c * Ic
    qv -= rgetindex(I, nx - 1, 1)
    qv -= rgetindex(I, nx, 2)
    value += f(qv, Ic)

    Ic = rgetindex(I, 1, ny)
    qv = c * Ic
    qv -= rgetindex(I, 2, ny)
    qv -= rgetindex(I, 1, ny - 1)
    value += f(qv, Ic)

    Ic = rgetindex(I, nx, ny)
    qv = c * Ic
    qv -= rgetindex(I, nx - 1, ny)
    qv -= rgetindex(I, nx, ny - 1)
    value += f(qv, Ic)

    return value
end

# 1-neighbor: Σ qv · I  (TSV regularizer)
_red1n(qv, Ic) = qv * Ic
igmrf_1n(I::AbstractMatrix, κ², ::Any) = _igmrf_walk(I, κ², _red1n)

# 2-neighbor: Σ qv²
_red2n(qv, _) = qv * qv
igmrf_2n(I::AbstractMatrix, κ², ::Any) = _igmrf_walk(I, κ², _red2n)


@inline function igmrf_qv(I::AbstractMatrix, κ², ix, iy)
    value = (4 + κ²) * rgetindex(I, ix, iy)
    @trace track_numbers = false if ix < lastindex(I, 1)
        value -= rgetindex(I, ix + 1, iy)
    end

    @trace track_numbers = false if iy < lastindex(I, 2)
        value -= rgetindex(I, ix, iy + 1)
    end

    @trace track_numbers = false if ix > firstindex(I, 1)
        value -= rgetindex(I, ix - 1, iy)
    end

    @trace track_numbers = false if iy > firstindex(I, 2)
        value -= rgetindex(I, ix, iy - 1)
    end

    return value
end
