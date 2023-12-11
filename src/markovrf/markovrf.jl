using SparseArrays
using LinearAlgebra
using ComradeBase

abstract type MarkovRandomField <: Dists.ContinuousMatrixDistribution end


Dists.insupport(::MarkovRandomField, x::AbstractMatrix) = true


function Dists._logpdf(d::MarkovRandomField, x::AbstractMatrix{<:Real})
    return unnormed_logpdf(d, x) + lognorm(d)
end


function Dists.invcov(d::MarkovRandomField)
    return Dists.invcov(d.cache, d.ρ)
end


include("cache.jl")
include("gmrf.jl")
include("studentTrf.jl")
