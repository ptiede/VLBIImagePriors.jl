export HierarchicalPrior

struct HierarchicalPrior{M, P} <: Dists.ContinuousMultivariateDistribution
    priormap::M
    hyperprior::P
end

Base.length(d::HierarchicalPrior) = length(d.priormap(rand(d.hyperprior))) + length(d.hyperprior)

function Base.show(io::IO, d::HierarchicalPrior)
    println(io, "HierarchicalPrior(")
    print(io, "\tmap: \n\t", d.priormap)
    println(io, "\thyper prior: \n\t", d.hyperprior)
    return println(io, ")")
end


function Dists.logpdf(d::HierarchicalPrior, x::NamedTuple)
    hp = x.hyperparams
    p = x.params
    pr = d.priormap(hp)
    return Dists.logpdf(pr, p) + Dists.logpdf(d.hyperprior, hp)
end

function Dists.rand(rng::AbstractRNG, d::HierarchicalPrior)
    hp = rand(rng, d.hyperprior)
    dp = d.priormap(hp)
    p = rand(rng, dp)
    return (params = p, hyperparams = hp)
end

function Dists.rand(rng::AbstractRNG, d::HierarchicalPrior, n::Dims)
    return map(CartesianIndices(n)) do I
        rand(rng, d)
    end
end

function Dists.rand(rng::AbstractRNG, d::HierarchicalPrior, n::Int)
    return map(1:n) do I
        rand(rng, d)
    end
end


HC.asflat(d::HierarchicalPrior) = TV.as((params = HC.asflat(d.priormap(rand(d.hyperprior))), hyperparams = HC.asflat(d.hyperprior)))
