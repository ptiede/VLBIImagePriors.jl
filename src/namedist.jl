export NamedDist

struct TupleDist{N, D<:NTuple{N, Dists.Distribution}} <: Dists.ContinuousMultivariateDistribution
    dists::D
end

Base.length(::TupleDist{N}) where {N} = N

function Dists.logpdf(d::TupleDist{N}, x::Tuple) where {N}
    dists = d.dists
    sum(map((dist, acc) -> Dists.logpdf(dist, acc), dists, x))
end

function Dists.rand(rng::AbstractRNG, d::TupleDist{N}) where {N}
    return ntuple(i->rand(rng, d.dists[i]), N)
end

function Dists.rand(rng::AbstractRNG, d::TupleDist, n::Int)
    map(1:n) do _
        rand(rng, d)
    end
end


function Dists.rand(rng::AbstractRNG, d::TupleDist, n::Dims)
    map(CartesianIndices(n)) do I
        rand(rng, d)
    end
end

HC.asflat(d::TupleDist{N}) where {N} = HC.asflat(d.dists)
HC.ascube(d::TupleDist{N}) where {N} = HC.ascube(d.dists)




struct NamedDist{Names, D} <: Dists.ContinuousMultivariateDistribution
    dists::D
end

Base.getproperty(d::NamedDist{N}, s::Symbol) where {N} = getproperty(NamedTuple{N}(getfield(d, :dists)), s)
Base.propertynames(::NamedDist{N}) where {N} = N
Base.length(::NamedDist{N}) where {N} = length(N)

"""
    NamedDist(d::NamedTuple{N})
    NamedDist(;dists...)

A Distribution with names `N`. This is useful to construct a set of random variables
with a set of names attached to them.

```julia-repl
julia> d = NamedDist((a=Normal(), b = Uniform(), c = MvNormal(randn(2), rand(2))))
julia> x = rand(d)
(a = 0.13789342, b = 0.2347895, c = [2.023984392, -3.09023840923])
julia> logpdf(d, x)
```

Note that NamedDist values passed to NamedDist can also be abstract collections of
distributions as well
```julia-repl
julia> d = NamedDist(a = Normal(),
                     b = MvNormal(ones(2)),
                     c = (Uniform(), InverseGamma())
                     d = (a = Normal(), Beta)
                    )
```
How this is done internally is considered an implementation detail and is not part of the
public interface.
"""
function NamedDist(d::NamedTuple{N}) where {N}
    d = values(d)
    dd = map(_distize, d)
    return NamedDist{N,typeof(dd)}(dd)
end

NamedDist(;kwargs...) = NamedDist((;kwargs...))

function Base.show(io::IO, d::NamedDist{N}) where {N}
    dists = getfield(d, :dists)
    if length(N) < 4
        show(io, NamedTuple{N}(getfield(d, :dists)))
    else
        println(io, "($(N[1]) = $(dists[1]), $(N[2]) = $(dists[2]), $(N[3]) = $(dists[3]), ...)")
    end
end


# Now this is a bunch of convienence stuff to automatically convert collection of distributions
# to something NamedDist will like
_distize(d::Dists.Distribution) = d
_distize(d::NTuple{N, <:Dists.Distribution}) where {N} = TupleDist(d)
_distize(d::Tuple) = TupleDist(map(_distize, d))
_distize(d::AbstractArray{<:Dists.Distribution}) = Dists.product_distribution(d)
_distize(d::NamedTuple{N}) where {N} = NamedDist(NamedTuple{N}(map(_distize, d)))


function Dists.logpdf(d::NamedDist{N}, x::NamedTuple{N}) where {N}
    vt = values(x)
    dists = getfield(d, :dists)
    sum(map((dist, acc) -> Dists.logpdf(dist, acc), dists, vt))
end

function Dists.logpdf(d::NamedDist{N}, x::NamedTuple{M}) where {N,M}
    xsub = select(x, N)
    return Dists.logpdf(d, xsub)
end

function Dists.rand(rng::AbstractRNG, d::NamedDist{N}) where {N}
    return NamedTuple{N}(map(x->rand(rng, x), getfield(d, :dists)))
end

function Dists.rand(rng::AbstractRNG, d::NamedDist{Na}, n::Int) where {Na}
    map(1:n) do _
        rand(rng, d)
    end
end


function Dists.rand(rng::AbstractRNG, d::NamedDist{Na}, n::Dims) where {Na}
    map(CartesianIndices(n)) do I
        rand(rng, d)
    end
end

HC.asflat(d::NamedDist{N}) where {N} = HC.asflat(NamedTuple{N}(getfield(d, :dists)))
HC.ascube(d::NamedDist{N}) where {N} = HC.ascube(NamedTuple{N}(getfield(d, :dists)))

# DensityInterface.DensityKind(::NamedDist) = DensityInterface.IsDensity()
# DensityInterface.logdensityof(d::NamedDist, x) = Dists.logpdf(d, x)
