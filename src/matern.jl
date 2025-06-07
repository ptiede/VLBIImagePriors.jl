export matern

# TODO Fix FFT's to work with Enzyme rather than using the rrule from ChainRules
struct StationaryMatern{TΛ, E<:Union{Serial, ThreadsEx}, P}
    kx::TΛ
    ky::TΛ
    executor::E
    p::P
    function StationaryMatern(T::Type{<:Number}, dims::Dims{2}; executor=Serial())
        kx = fftfreq(dims[1], one(T))*π
        ky = fftfreq(dims[2], one(T))*π
        plan = FFTW.plan_fft!(zeros(Complex{T}, dims); flags=FFTW.MEASURE)
        return new{typeof(kx), typeof(executor), typeof(plan)}(kx, ky, executor, plan)
    end
end

ComradeBase.executor(d::StationaryMatern) = getfield(d, :executor)

function Base.show(io::IO, x::StationaryMatern)
    println(io, "StationaryMatern")
    println(io, "\tBase type: $(eltype(x.kx))")
    println(io, "\tsize:      ($(size(x.kx,1)), $(size(x.ky,1)))")
    println(io, "\texec:      $(x.executor)")
end

function Serialization.serialize(s::Serialization.AbstractSerializer, cache::StationaryMatern)
    Serialization.writetag(s.io, Serialization.OBJECT_TAG)
    Serialization.serialize(s, typeof(cache))
    Serialization.serialize(s, cache.kx)
    Serialization.serialize(s, cache.ky)
    Serialization.serialize(s, cache.executor)
end

function Serialization.deserialize(s::AbstractSerializer, ::Type{<:StationaryMatern})
    kx = Serialization.deserialize(s)
    ky = Serialization.deserialize(s)
    executor = Serialization.deserialize(s)
    return StationaryMatern(eltype(kx), (length(kx), length(ky)); executor)
end


@fastmath function (θ::StationaryMatern)(x::AbstractArray, ρ::NTuple{2,Number}, ξ::Number, ν::Number)
    (;kx, ky, p) = θ
    ρx, ρy = ρ
    @assert size(x) == (length(kx), length(ky))
    T = promote_type(eltype(x), typeof(ρ[1]), typeof(ν))
    κ = T(sqrt(8*ν))
    κ2 = κ*κ
    τ = κ^ν*sqrt(ν*convert(T, π))/sqrt(prod(size(x)))
    ns = similar(x , Complex{eltype(x)})
    expp = -(ν+1)/2
    s, c = sincos(ξ)
    e = executor(θ)

    @threaded e for i in eachindex(ky)
        for j in eachindex(kx)
            @inbounds rx = c*kx[j] - s*ky[i]
            @inbounds ry = s*kx[j] + c*ky[i]
            @inbounds ns[j,i] = τ*sqrt(ρx*ρy)*x[j,i]*(κ2 + (ρx*rx)^2 + (ρy*ry)^2)^expp
        end
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
    matern([T=Float64], dims::Dims{2}; executor=Serial())
    matern([T=Float64], dims::Int...; executor=Serial())

Creates an approximate Matern Gaussian process that approximates the Matern process
on a regular grid which cyclic boundary conditions. This function returns a tuple of
two objects
 - A functor `f` of type `StationaryMatern` that iid-Normal noise to a draw from the Matern process.
   The functor call arguments are `f(s, ρ, ν)` where `s` is the random white Gaussian noise with
   dimension `dims`, `ρ` is the correlation length, and `ν` is Matern smoothness parameter
 - The a set of `prod(dims)` standard Normal distributions that can serve as the noise generator
   for the process.

# Example

## Arguments

- `[T::Float64]`: Optional element type of the matern process. Default is `Float64`.
- `dims::Dims{2}`: The dimensions of the Matern process. This is a tuple of two integers.

or 

- `grid::AbstractRectiGrid`: A grid object that the Matern process is defined on. 

## Keyword arguments



```julia-repl
julia> transform, dstd = matern((32, 32))
julia> draw_matern = transform(rand(dstd), 10.0, 2.0)
julia> draw_matern_aniso = transform(rand(dstd), (10.0, 5.0), π/4 2.0) # anisotropic Matern
julia> ones(32, 32) .+ 5.* draw_matern # change the mean and variance of the field
```
"""
function matern(T::Type{<:Number}, dims::Dims{2}; executor=Serial())
    d = StationaryMatern(T, dims; executor=executor)
    return d, std_dist(d)
end

function matern(grid::ComradeBase.AbstractRectiGrid)
    return matern(eltype(grid), size(grid); executor=executor(grid))
end

matern(dims::Dims{2}; executor=Serial()) = matern(Float64, dims; executor=executor)
matern(T::Type{<:Number}, dims::Vararg{Int}; executor=Serial()) = matern(T, dims; executor=executor)
matern(dims::Vararg{Int}; executor=Serial()) = matern(dims; executor)

"""
    matern(img::AbstractMatrix)

Creates an approximate Matern Gaussian process with dimension `size(img)`

"""
matern(img::AbstractMatrix{T}; executor=Serial()) where {T} = matern(T, size(img); executor)


struct StdNormal{T, N} <: Dists.ContinuousDistribution{Dists.ArrayLikeVariate{N}}
    dims::Dims{N}
end

StdNormal(d::Dims{N}) where {N} = StdNormal{Float64, N}(d)

Base.size(d::StdNormal) = d.dims
Base.length(d::StdNormal) = prod(d.dims)
Base.eltype(::StdNormal{T}) where {T} = T
Dists.insupport(::StdNormal, x::AbstractVector) = true

HC.asflat(d::StdNormal) = TV.as(Array, size(d)...)
HC.ascube(d::StdNormal) = HC.ArrayHC(d)

function HC._step_transform(h::HC.ArrayHC{<:StdNormal}, p::AbstractVector, index)
    d = Dists.Normal()
    out = Dists.quantile.(Ref(d), p)
    return out, index+HC.dimension(h)
end

function HC._step_inverse!(x::AbstractVector, index, h::HC.ArrayHC{<:StdNormal}, y::AbstractVector)
    d = Dists.Normal()
    x .= Dists.cdf.(Ref(d), y)
    return index+HC.dimension(h)
end

Dists.mean(d::StdNormal) = zeros(size(d))
Dists.cov(d::StdNormal)  = Diagonal(prod(size(d)))


function Dists._logpdf(d::StdNormal{T, N}, x::AbstractArray{T, N}) where {T<:Real, N}
    return __logpdf(d, x)
end
Dists._logpdf(d::StdNormal{T, 2}, x::AbstractMatrix{T}) where {T<:Real} = __logpdf(d, x)


# __logpdf(d::StdNormal, x) = -sum(abs2, x)/2 - prod(d.dims)*Dists.log2π/2

function __logpdf(d::StdNormal, x)
    s = zero(eltype(x))
    for i in eachindex(x)
        s += abs2(x[i])
    end
    return -s/2 - prod(d.dims)*Dists.log2π/2
end


function Dists._rand!(rng::AbstractRNG, ::StdNormal{T, N}, x::AbstractArray{T, N}) where {T<: Real, N}
    return randn!(rng, x)
end
