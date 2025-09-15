export MaternPS, SqExpPS, RationalQuadPS, ScaledPS, MarkovPS, 
       StationaryRandomFieldPlan, 
       StationaryRandomField, genfield, std_dist, 
       matern

struct StationaryRandomFieldPlan{TΛ, E, P}
    kx::TΛ
    ky::TΛ
    executor::E
    p::P
    function StationaryRandomFieldPlan(T::Type{<:Number}, dims::Dims{2}; executor=Serial())
        kx = fftfreq(dims[1], one(T))*π
        ky = fftfreq(dims[2], one(T))*π
        plan = FFTW.plan_fft!(zeros(Complex{T}, dims); flags=FFTW.MEASURE)
        return new{typeof(kx), typeof(executor), typeof(plan)}(kx, ky, executor, plan)
    end
end

Base.eltype(d::StationaryRandomFieldPlan) = eltype(d.kx)
Base.size(d::StationaryRandomFieldPlan) = (length(d.kx), length(d.ky))
Base.length(d::StationaryRandomFieldPlan) = prod(size(d))
ComradeBase.executor(d::StationaryRandomFieldPlan) = getfield(d, :executor)


"""
    StationaryRandomFieldPlan(g::RectiGrid{<:ComradeBase.SpatialDims})

Sets up the plan for generating a stationary random field on the given grid `g`.
Currently we only support 2D spatial grids but this will be relaxed in the future.
"""
function StationaryRandomFieldPlan(g::RectiGrid{<:ComradeBase.SpatialDims})
    T = eltype(g)
    ex = executor(g)
    ng = size(g)
    if !(ex isa Serial || ex isa ThreadsEx)
        ex = Serial()
        @warn "Executor type $(typeof(ex)) not supported, defaulting to Serial()"
    end
    return StationaryRandomFieldPlan(T, ng; executor=ex)
end


function Base.show(io::IO, x::StationaryRandomFieldPlan)
    println(io, "StationaryRandomFieldPlan")
    println(io, "\tBase type: $(eltype(x.kx))")
    println(io, "\tsize:      ($(size(x.kx,1)), $(size(x.ky,1)))")
    println(io, "\texec:      $(x.executor)")
end

function Serialization.serialize(s::Serialization.AbstractSerializer, cache::StationaryRandomFieldPlan)
    Serialization.writetag(s.io, Serialization.OBJECT_TAG)
    Serialization.serialize(s, typeof(cache))
    Serialization.serialize(s, cache.kx)
    Serialization.serialize(s, cache.ky)
    Serialization.serialize(s, cache.executor)
end

function Serialization.deserialize(s::AbstractSerializer, ::Type{<:StationaryRandomFieldPlan})
    kx = Serialization.deserialize(s)
    ky = Serialization.deserialize(s)
    executor = Serialization.deserialize(s)
    return StationaryRandomFieldPlan(eltype(kx), (length(kx), length(ky)); executor)
end

"""
    AbstractPowerSpectrum


Defines a spectral density or power spectrum for a stationary random field.

To define your own RF a user needs to subtype `AbstractPowerSpectrum` and define the method
[`VLBIImagePriors.ampspectrum`](@ref).

"""
abstract type AbstractPowerSpectrum end

"""
    ampspectrum(ps::AbstractPowerSpectrum, ks)

Computes the square root of the power spectrum at the wavenumbers `ks`.
Note we typically want to assume that the power spectrum is normalized such that its integral is
2π when integrated over the entire frequency space.
"""
function amplitudespectrum end

"""
    ScaledPS(ps::AbstractPowerSpectrum, τ, ξ=0)
Defines a power spectrum that is a scaled and rotated version of another power spectrum `ps`.
The scaling is defined by `τ` stretches the PS in the y-direction by `τ` and the rotation is defined
by the angle `ξ` in radians CCW.
"""
struct ScaledPS{P<:AbstractPowerSpectrum, T} <: AbstractPowerSpectrum
    ps::P
    τ::T 
    s::T
    c::T
    function ScaledPS(ps::A, τ, ξ) where {A}
        s, c = sincos(ξ)
        T = promote_type(typeof(τ), typeof(ξ))
        return new{A, T}(ps, convert(T, τ), convert(T, s), convert(T, c))
    end
end

function ScaledPS(ps::AbstractPowerSpectrum, τ)
    return ScaledPS(ps, τ, zero(τ))
end

function ampspectrum(p::ScaledPS, ks)
    (; ps, τ, s, c) = p
    kx, ky = ks
    kx2 = c*kx - s*ky
    ky2 = (s*kx + c*ky)/τ
    return ampspectrum(ps, (kx2, ky2))/sqrt(τ)
end




"""
    MaternPS(ρ, ν)

Defines a Matern power spectrum with correlation length `ρ` and smoothness parameter `ν`.
The power spectrum is given by
    S(k) = τ * (κ² + k²)^(-(ν + 1)/2)
where `κ = sqrt(8ν)/ρ` and `τ = κ^ν * sqrt(νπ)`.
"""
struct MaternPS{T} <: AbstractPowerSpectrum
    τ::T
    κ2::T
    ν::T
    function MaternPS(ρ, ν)
        @assert ρ > zero(ρ) "Correlation length ρ must be positive"
        @assert ν > zero(ν) "Smoothness parameter ν must be positive"
        T = promote_type(typeof(ρ), typeof(ν))
        κ = T(sqrt(8*ν)/ρ)
        κ2 = κ*κ
        τ = κ^ν*sqrt(ν*convert(T, π))
        return new{T}(τ, κ2, ν)
    end
end

@inline function ampspectrum(ps::MaternPS, ks)
    (; τ, κ2, ν) = ps
    kx, ky = ks
    expp = -(ν+1)/2
    return τ*(κ2 + kx^2 + ky^2)^expp
end

"""
    SqExpPS(ρ)

Defines a squared exponential (Gaussian) power spectrum with correlation length `ρ`.
The power spectrum is given by
    S(k) = ρ * exp(-ρ^2 * k^2 / 4)
where the factor `ρ` is to roughly ensure the same marginal variance.
"""
struct SqExpPS{T} <: AbstractPowerSpectrum
    ρ::T
end

@inline function ampspectrum(ps::SqExpPS{T}, ks) where {T}
    (; ρ) = ps
    kx, ky = ks
    return exp(-(kx^2 + ky^2)*inv(4)*ρ^2)*ρ
end


"""
    RationalQuadPS(ρ, α)

Defines a rational quadratic power spectrum with correlation length `ρ` and shape parameter `α`.
The power spectrum is given by
    S(k) = (αρ) * (1 + (αρ^2/2) * k^2)^(-(α + 1)/2)

"""
struct RationalQuadPS{T} <: AbstractPowerSpectrum
    ρ::T
    α::T
end

@inline function ampspectrum(ps::RationalQuadPS{T}, ks) where {T}
    (; ρ, α) = ps
    kx, ky = ks
    αρ = α*ρ
    return (αρ)*(1 + αρ*ρ/2*(kx^2 + ky^2))^(-(α + 1)/2)
end

"""
    MarkovPS(ρ::NTuple{N,T}) where {T, N}

Defines a power spectrum corresponding to a Nth order Markov Random Field with coefficents `ρs`.
The power spectrum is given by
    S(k) = Norm / (1 + Σₙ (ρₙ * k^2)^n)
where norm is given by sqrt(Σₙ ρₙ) to roughly ensure the same marginal variance.
"""
struct MarkovPS{T, N} <: AbstractPowerSpectrum
    ρs::NTuple{N,T}
end

@inline function ampspectrum(ps::MarkovPS{T, N}, ks) where {T, N}
    (; ρs) = ps
    kx, ky = ks
    k2 = kx^2 + ky^2
    terms = ntuple(Val(N)) do n
        (ρs[n]*k2)^n
    end
    norm = ntuple(Val(N)) do n
        m = (n == 1 ? 2 : n) 
        ρs[n]*n*sin(T(π)/m)
    end
    return sqrt(sum(norm))/sqrt(1 + reduce(+, terms))
end

"""
    StationaryRandomField(ps::AbstractPowerSpectrum, plan::StationaryRandomFieldPlan)

Creates a stationary random field defined by the power spectrum `ps` and the evaluation `plan`.
Note that by default the plan assumes periodic boundary conditions.
"""
struct StationaryRandomField{PS<:AbstractPowerSpectrum, P}
    ps::PS
    plan::P
end

"""
    genfield(rf::StationaryRandomField, z::AbstractArray)

Generates a stationary random field from the standardized normal random field `z`
using the power spectrum and plan defined in `rf`. Typically the input `z` should be 
a draw from `std_dist(rf)`.
"""
function genfield(rf::StationaryRandomField, z::AbstractArray)
    ps = rf.ps
    (;kx, ky, p) = rf.plan
    e = executor(rf.plan)

    ns = similar(z , Complex{eltype(z)})
    @threaded e for i in eachindex(ky)
        for j in eachindex(kx)
            @inbounds ns[j, i] = ampspectrum(ps, (kx[j], ky[i]))*z[j,i]
        end
    end

    p*ns
    rast = (real.(ns) .+ imag.(ns))./sqrt(prod(size(z)))
    return rast
end


"""
    matern([T=Float64], dims::Dims{2}; executor=Serial())
    matern([T=Float64], dims::Int...; executor=Serial())


!!! warn
    This is deprecated and will be removed in the next release. Please use the more general
    StationaryRandomField and MaternPS types instead.

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
function matern(T::Type{<:Number}, dim::Dims{2}; executor=Serial())
    plan = StationaryRandomFieldPlan(T, dim; executor=executor)
    f = StationaryMatern(plan)
    return f, std_dist(plan)
end
matern(dims::Dims{2}; executor=Serial()) = matern(Float64, dims; executor=executor)

function matern(g::ComradeBase.RectiGrid{<:ComradeBase.SpatialDims})
    T = eltype(g)
    ex = executor(g)
    ng = size(g)
    return matern(T, ng; executor=ex)
end


struct StationaryMatern{P}
    plan::P
end

function (m::StationaryMatern)(z, ρ::NTuple{2}, ξ, ν)
    τ = ρ[1]/ρ[2]
    ps = ScaledPS(MaternPS(ρ[1], ν), τ, ξ)
    rf = StationaryRandomField(ps, m.plan)
    return genfield(rf, z)
end

function (m::StationaryMatern)(z, ρ, ν)
    ps = MaternPS(ρ, ν)
    rf = StationaryRandomField(ps, m.plan)
    return genfield(rf, z)
end




"""
    std_dist(d::StationaryRandomField)
    std_dist(d::StationaryRandomFieldPlan)

Returns the standardized normal distribution corresponding to the stationary random field `d`
"""
function std_dist(d::StationaryRandomField)
    return std_dist(d.plan)
end

function std_dist(d::StationaryRandomFieldPlan)
    StdNormal{eltype(d.kx),2}((length(d.kx), length(d.ky)))
end

struct StdNormal{T, N} <: Dists.ContinuousDistribution{Dists.ArrayLikeVariate{N}}
    dims::Dims{N}
end

StdNormal(d::Dims{N}) where {N} = StdNormal{Float64, N}(d)

Base.size(d::StdNormal) = d.dims
Base.length(d::StdNormal) = prod(d.dims)
Base.eltype(::StdNormal{T}) where {T} = T
Dists.insupport(::StdNormal, x::AbstractArray) = true

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
Dists.cov(d::StdNormal)  = I(length(d))


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
