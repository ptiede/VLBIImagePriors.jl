using Enzyme
Enzyme.API.runtimeActivity!(true)

struct Uniform{T}
    a::T
    b::T
end
logpdf(d::Uniform, ::Real) = -log(d.b - d.a)

struct Normal{T}
    μ::T
    σ::T
end
logpdf(d::Normal, x::Real) = -(x - d.μ)^2 / (2 * d.σ^2)

struct ProductDist{V}
    dists::V
end
function logpdf(d::ProductDist, x::Vector)
    dists = d.dists
    s = zero(eltype(x))
    for i in eachindex(x)
        s += logpdf(dists[i], x[i])
    end
    return s
end

struct NamedDist{Names, D<:NamedTuple{Names}}
    dists::D
end

function logpdf(d::NamedDist{N}, x::NamedTuple{N}) where {N}
    dists = d.dists
    return logpdf(dists.a, x.a) + logpdf(dists.b, x.b)
end

@noinline function logpdf_ref(d::NamedDist{N}, x::Ref) where {N}
    return logpdf(d, x[])
end

d = NamedDist((;a = Normal(0.0, 1.0), b = ProductDist([Uniform(0.0, 1.0), Uniform(0.0, 1.0)])))
p = (a = 1.0, b = [0.5, 0.5])
dp = Enzyme.make_zero(p)
logpdf(d, p)
autodiff(Reverse, logpdf, Active, Const(d), Duplicated(p, dp))
autodiff(Reverse, logpdf_ref, Active, Const(d), Duplicated(Ref(p), Ref(dp)))
