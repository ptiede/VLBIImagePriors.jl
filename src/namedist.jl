export NamedDist


struct NamedDist{Names, D} <: Dists.ContinuousMultivariateDistribution
    dists::D
end

"""
    NamedDist(d::NamedTuple{N})

A Distribution with names `N`. This is useful to construct a set of random variables
with a set of names attached to them.

# Example
```julia-repl
julia> d = NamedDist((a=Normal(), b = Uniform(), c = MvNormal(randn(2), rand(2))))
julia> x = rand(d)
(a = 0.13789342, b = 0.2347895, c = [2.023984392, -3.09023840923])
julia> logpdf(d, x)
```
"""
function NamedDist(d::NamedTuple{N}) where {N}
    d = values(d)
    return NamedDist{N,typeof(d)}(d)
end

function Dists.logpdf(d::NamedDist{N}, x::NamedTuple{N}) where {N}
    vt = values(x)
    dists = d.dists
    sum(map((dist, acc) -> Dists.logpdf(dist, acc), dists, vt))
end

function Dists.logpdf(d::NamedDist{N}, x::NamedTuple{M}) where {N,M}
    xsub = select(x, N)
    return Dists.logpdf(d, xsub)
end

function Dists.rand(rng::AbstractRNG, d::NamedDist{N}) where {N}
    return NamedTuple{N}(map(x->rand(rng, x), d.dists))
end

function Dists.rand(rng::AbstractRNG, d::NamedDist{Na}, n::Dims) where {Na}
    map(CartesianIndices(n)) do I
        rand(rng, d)
    end
end

HypercubeTransform.asflat(d::NamedDist{N}) where {N} = asflat(NamedTuple{N}(d.dists))
HypercubeTransform.ascube(d::NamedDist{N}) where {N} = ascube(NamedTuple{N}(d.dists))

DensityInterface.DensityKind(::NamedDist) = DensityInterface.IsDensity()
DensityInterface.logdensityof(d::NamedDist, x) = Dists.logpdf(d, x)
