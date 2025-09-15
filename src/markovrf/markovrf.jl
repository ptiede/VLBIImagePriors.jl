using SparseArrays
using LinearAlgebra
using ComradeBase

export scalematrix, corrparam, graph

"""
    $(TYPEDEF)

An abstract type for a `MarkovRandomField`. We assume that the distribution is
of the form

    p(x | ρ) = N(detQ(ρ)) f(xᵀQ(ρ)x),

where `f` is a function and `N` is the normalization of the distribution, and ρ is the
correlation parameter.

To implement the informal interface e.g., `MyRF <: MarkovRandomField`, the user must implement

  - `lognorm(d::MyRF)`: Which computes the log of the normalization constant `N`
  - `unnormed_logpdf(d::MyRF, x::AbstractMatrix)`: Which computes f(xᵀQx)
  - `Distributions._rand!(rng::AbstractRNG, d::MyRF, x::AbstractMatrix)`: To enable sampling
     from the prior

Additionally, there are a number of auto-generated function that can be overwritten:
  - `graph(d::MyRF)`: Which returns the graph structure of the Markov Random Field.
     The default returns the property `d.graph`.
  - `corrparam(d::MyRF)`: Which returns the correlation parameter ρ of the Markov Random Field.
     The default returns the property `d.ρ`.
  - `Base.size(d::MyRF)`: Which returns the size of the distribution. This defaults to the
     size of the graph cache.
  - `scalematrix(d::MyRF)`: Which computes the scale matrix `Q`, of the random field. The
     default is to forward to the `scalematrix(graph(d), corrparm(d))`.
  - `(c::ConditionalMarkov{<:MyRF})(ρ)`: To map from a correlation to the distribution
  - `HypercubeTransform.asflat(d::MyRF)`: To map from parameter space to a flattened version.
     The default is `TransformVariables.as(Matrix, size(d)...)`
  - `Distributions.insupport(d::MyRF, x::AbstractMatrix)` which checks if `x` is in the
     support of `d`. The default is to always return true.
  - `LinearAlgebra.logdet(d::MyRF)` which computes the log determinant of `Q`. This defaults to
     `logdet(graph(d), corrparam(d))`.

Finally additional optional methods are:
  - `Distributions.mean(d::MyRF)`: Which computes the mean of the process
  - `Distributions.cov(d::MyRF)`: Which computes the covariance matrix of the process.
  - `Distributions.invcov(d::MyRF)`: Computes the precision matrix of the random field


For an example implementation see e.g., [GaussMarkovRandomField](@ref)
"""
abstract type MarkovRandomField <: Dists.ContinuousMatrixDistribution end

function Base.show(io::IO, d::T) where {T <: MarkovRandomField}
    s = "$T"
    t = split(s, "{")[1]
    println(io, "$t(")
    println(io, "Graph: ", d.graph)
    println(io, "Correlation Parameter: ", d.ρ)
    return print(io, ")")
end

"""
    graph(m::MarkovRandomField)

Returns the graph or graph cache of the Markov Random field `m`.
"""
@inline graph(m::MarkovRandomField) = m.graph

"""
    corrparam(m::MarkovRandomField)

Returns the correlation parameter of the Markov Random field `m`. For details
about the correlation parmeter see [`MarkovRandomFieldGraph`](@ref).
"""
@inline corrparam(m::MarkovRandomField) = m.ρ

LinearAlgebra.logdet(m::MarkovRandomField) = logdet(graph(m), corrparam(m))


Base.size(m::MarkovRandomField) = size(graph(m))

"""
    scalematrix(m::MarkovRandomField)

Return the scale matrix for the `Markov Random field`. For a Gaussian Markov
random field this corresponds to the precision matrix of the Gaussian field.

For other random processes this is the `metric` of the inner product, i.e. `Q`
in

    xᵀQx

which is the distance from the origin to `x` using the metric `Q`.
"""
@inline scalematrix(m::MarkovRandomField) = scalematrix(graph(m), corrparam(m))

# Assume that the distribution support is Rᴺ
HC.asflat(d::MarkovRandomField) = TV.as(Matrix, size(d)...)

Dists.insupport(::MarkovRandomField, x::AbstractMatrix) = true


function Dists._logpdf(d::MarkovRandomField, x::AbstractMatrix{<:Real})
    return unnormed_logpdf(d, x) + lognorm(d)
end


include("cache.jl")
include("conditional.jl")
include("gmrf.jl")
include("studentTrf.jl")
include("exponential.jl")
include("noncentered.jl")
