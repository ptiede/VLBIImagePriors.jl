export matern

# TODO Fix FFT's to work with Enzyme rather than using the rrule from ChainRules
struct StationaryMatern{TΛ, P}
    kx::TΛ
    ky::TΛ
    p::P
    function StationaryMatern(T::Type{<:Number}, dims::Dims{2})
        kx = fftfreq(dims[1], one(T))*π
        ky = fftfreq(dims[2], one(T))*π
        plan = FFTW.plan_fft!(zeros(Complex{T}, dims))
        return new{typeof(kx), typeof(plan)}(kx, ky, plan)
    end
end

function Base.show(io::IO, x::StationaryMatern)
    println(io, "StationaryMatern")
    println(io, "\tBase type: $(eltype(x.kx))")
    println(io, "\tsize:      ($(size(x.kx,1)), $(size(x.ky,1)))")
end

function Serialization.serialize(s::Serialization.AbstractSerializer, cache::StationaryMatern)
    Serialization.writetag(s.io, Serialization.OBJECT_TAG)
    Serialization.serialize(s, typeof(cache))
    Serialization.serialize(s, cache.kx)
    Serialization.serialize(s, cache.ky)
end

function Serialization.deserialize(s::AbstractSerializer, ::Type{<:StationaryMatern})
    kx = Serialization.deserialize(s)
    ky = Serialization.deserialize(s)
    return StationaryMatern(eltype(kx), (length(kx), length(ky)))
end


@fastmath function (θ::StationaryMatern)(x::AbstractArray, ρ::NTuple{2,Number}, ξ::Number, ν::Number)
    (;kx, ky, p) = θ
    ρx, ρy = ρ
    @assert size(x) == (length(kx), length(ky))
    T = promote_type(eltype(x), typeof(ρ[1]), typeof(ν))
    κ = T(sqrt(8*ν))
    κ2 = κ*κ
    τ = κ^ν*sqrt(ν)*convert(T, π)/sqrt(prod(size(x)))
    ns = similar(x , Complex{eltype(x)})
    expp = -(ν+1)/2
    s, c = sincos(ξ)
    @inbounds for i in eachindex(ky), j in eachindex(kx)
        rx = c*kx[j] - s*ky[i]
        ry = s*kx[j] + c*ky[i]    
        ns[j,i] = τ*sqrt(ρx*ρy)*x[j,i]*(κ2 + (ρx*rx)^2 + (ρy*ry)^2)^expp/2
    end
    p*ns
    rast = (real.(ns) .+ imag.(ns))
    return rast
end

@fastmath function (θ::StationaryMatern)(x::AbstractArray, ρ::Number, ν::Number)
    return θ(x, (ρ, ρ), zero(ρ), ν)
end


function std_dist(d::StationaryMatern)
    StdNormal{eltype(d.kx),2}((length(d.kx), length(d.ky)))
end


"""
    matern([T=Float64], dims::Dims{2})
    matern([T=Float64], dims::Int...)

Creates an approximate Matern Gaussian process that approximates the Matern process
on a regular grid which cyclic boundary conditions. This function returns a tuple of
two objects
 - A functor `f` of type `StationaryMatern` that iid-Normal noise to a draw from the Matern process.
   The functor call arguments are `f(s, ρ, ν)` where `s` is the random white Gaussian noise with
   dimension `dims`, `ρ` is the correlation length, and `ν` is Matern smoothness parameter
 - The a set of `prod(dims)` standard Normal distributions that can serve as the noise generator
   for the process.

# Example

```julia-repl
julia> transform, dstd = matern((32, 32))
julia> draw_matern = transform(rand(dstd), 10.0, 2.0)
julia> draw_matern_aniso = transform(rand(dstd), (10.0, 5.0), π/4 2.0) # anisotropic Matern
julia> ones(32, 32) .+ 5.* draw_matern # change the mean and variance of the field
```
"""
function matern(T::Type{<:Number}, dims::Dims{2})
    d = StationaryMatern(T, dims)
    return d, std_dist(d)
end

matern(dims::Dims{2}) = matern(Float64, dims)
matern(T::Type{<:Number}, dims::Vararg{Int}) = matern(T, dims)
matern(dims::Vararg{Int}) = matern(dims)

"""
    matern(img::AbstractMatrix)

Creates an approximate Matern Gaussian process with dimension `size(img)`

"""
matern(img::AbstractMatrix{T}) where {T} = matern(T, size(img))


struct StdNormal{T, N} <: Dists.ContinuousDistribution{Dists.ArrayLikeVariate{N}}
    dims::Dims{N}
end

StdNormal(d::Dims{N}) where {N} = StdNormal{Float64, N}(d)

Base.size(d::StdNormal) = d.dims
Base.length(d::StdNormal) = prod(d.dims)
Base.eltype(::StdNormal{T}) where {T} = T
Dists.insupport(::StdNormal, x::AbstractVector) = true

HC.asflat(d::StdNormal) = TV.as(Array, size(d)...)
Dists.mean(d::StdNormal) = zeros(size(d))
Dists.cov(d::StdNormal)  = Diagonal(prod(size(d)))


function Dists._logpdf(d::StdNormal{T, N}, x::AbstractArray{T, N}) where {T<:Real, N}
    return __logpdf(d, x)
end
Dists._logpdf(d::StdNormal{T, 2}, x::AbstractMatrix{T}) where {T<:Real} = __logpdf(d, x)


__logpdf(d::StdNormal, x) = -sum(abs2, x)/2 - prod(d.dims)*Dists.log2π/2


function Dists._rand!(rng::AbstractRNG, ::StdNormal{T, N}, x::AbstractArray{T, N}) where {T<: Real, N}
    return randn!(rng, x)
end
