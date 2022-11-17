using BesselK


struct LogitMatern{G,D<:NamedTuple} <: Dists.ContinuousMultivariateDistribution
    distance::G
    dims::D
end

Base.size(d::LogitMatern) = length.(values(d.dims) - 1)
