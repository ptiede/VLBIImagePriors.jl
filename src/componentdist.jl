export ComponentDist

using ComponentArrays

struct ComponentDist{Names, D, A} <: Dists.ContinuousMultivariateDistribution
    dists::D
    axis::A
end

Base.getproperty(d::ComponentDist{N}, s::Symbol) where {N} = getproperty(NamedTuple{N}(getfield(d, :dists)), s)
Base.getproperty(d::ComponentDist, ::Val{N}) where {N} = getproperty(d, N)
Base.propertynames(::ComponentDist{N}) where {N} = N
Base.length(d::ComponentDist) = mapreduce(length, +, getfield(d, :dists))

"""
    ComponentDist(d::NamedTuple{N})
    ComponentDist(;dists...)

A Distribution with names `N`. This is useful to construct a set of random variables
with a set of names attached to them.

```julia-repl
julia> d = ComponentDist((a=Normal(), b = Uniform(), c = MvNormal(randn(2), rand(2))))
julia> x = rand(d)
(a = 0.13789342, b = 0.2347895, c = [2.023984392, -3.09023840923])
julia> logpdf(d, x)
```

Note that ComponentDist values passed to ComponentDist can also be abstract collections of
distributions as well
```julia-repl
julia> d = ComponentDist(a = Normal(),
                     b = MvNormal(ones(2)),
                     c = (Uniform(), InverseGamma())
                     d = (a = Normal(), Beta)
                    )
```
How this is done internally is considered an implementation detail and is not part of the
public interface.
"""
function ComponentDist(d::NamedTuple{N}) where {N}
    d = values(d)
    dd = map(_distize, d)
    axis = getaxes(ComponentVector(NamedTuple{N}(map(rand, dd))))
    return ComponentDist{N,typeof(dd), typeof(axis)}(dd, axis)
end


function Dists.logpdf(d::ComponentDist{N}, x::NamedTuple{N}) where {N}
    vt = values(x)
    dists = getfield(d, :dists)
    sum(map((dist, acc) -> Dists.logpdf(dist, acc), dists, vt))
end

function Dists.logpdf(d::ComponentDist{N}, x::ComponentArray) where {N}
    s = mapreduce(+, valkeys(x)) do k
        Dists.logpdf(getproperty(d, k), getproperty(x, k))
    end
    return s
end

function Dists.rand(rng::AbstractRNG, d::ComponentDist)
    x = ComponentVector(zeros(length(d)), getfield(d, :axis))
    for v in valkeys(x)
        setproperty!(x, v, Dists.rand(rng, getproperty(d, v)))
    end
    return x
end
