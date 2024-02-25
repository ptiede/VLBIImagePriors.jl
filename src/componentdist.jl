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
    dd = map(_distize_comp, d)
    axis = getaxes(ComponentVector(NamedTuple{N}(map(rand, dd))))
    return ComponentDist{N,typeof(dd), typeof(axis)}(dd, axis)
end

_distize_comp(d::Dists.Distribution) = d
_distize_comp(d::NTuple{N, <:Dists.Distribution}) where {N} = TupleDist(d)
_distize_comp(d::Tuple) = TupleDist(map(_distize, d))
_distize_comp(d::AbstractArray{<:Dists.Distribution}) = Dists.product_distribution(d)
_distize_comp(d::NamedTuple{N}) where {N} = ComponentDist(NamedTuple{N}(map(_distize_comp, d)))


ComponentDist(;kwargs...) = ComponentDist((;kwargs...))


function Dists.logpdf(d::ComponentDist{N}, x::NamedTuple{N}) where {N}
    vt = values(x)
    dists = getfield(d, :dists)
    sum(map((dist, acc) -> Dists.logpdf(dist, acc), dists, vt))
end

# TODO do I really need  generated function for this?
@generated function Dists.logpdf(d::ComponentDist{N}, x::ComponentArray) where {N}
    exprs = [:(Dists.logpdf(d.$k, x.$k)) for k in N]
    return :((+($(exprs...))))
end

flexible_setproperty!(d::ComponentVector, ::Val{k}, v) where {k} = setproperty!(d, Val(k), v); nothing
@generated function flexible_setproperty!(d::ComponentVector, ::Val{k}, v::NamedTuple{N}) where {k, N}
    exprs = []
    for n in N
        sym = QuoteNode(Symbol("$n"))
        push!(exprs, :(flexible_setproperty!(d.$k, Val($sym), v.$n)))
    end

    return quote
        $(exprs...)
        return nothing
    end
end

function Dists.rand(rng::AbstractRNG, d::ComponentDist)
    x = ComponentVector(zeros(length(d)), getfield(d, :axis))
    for v in valkeys(x)
        r = Dists.rand(rng, getproperty(d, v))
        flexible_setproperty!(x, v, r)
    end
    return x
end

function HC.asflat(d::ComponentDist{N}) where {N}
    dists = getfield(d, :dists)
    trfs = HC.asflat(NamedTuple{N}(fieldvalues(dists))).transformations
    return TV.as(getfield(d, :axis), trfs)
end
