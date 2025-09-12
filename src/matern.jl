export MaternPS, SqExpPS, RationalQuadPS, StationaryRandomField, StationaryRandomFieldPlan, genfield, std_dist

# TODO Fix FFT's to work with Enzyme rather than using the rrule from ChainRules
struct StationaryRandomFieldPlan{TΛ, E<:Union{Serial, ThreadsEx}, P}
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

ComradeBase.executor(d::StationaryRandomFieldPlan) = getfield(d, :executor)

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


abstract type AbstractPowerSpectrum end

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

@inline function ampspectrum(ps::MaternPS, kx, ky)
    (; τ, κ2, ν) = ps
    expp = -(ν+1)/2
    return τ*(κ2 + kx^2 + ky^2)^expp
end


struct SqExpPS{T} <: AbstractPowerSpectrum
    ρ::T
end

@inline function ampspectrum(ps::SqExpPS{T}, kx, ky) where {T}
    (; ρ) = ps
    return exp(-(kx^2 + ky^2)*inv(4)*ρ^2)*ρ
end

struct RationalQuadPS{T} <: AbstractPowerSpectrum
    ρ::T
    α::T
end

@inline function ampspectrum(ps::RationalQuadPS{T}, kx, ky) where {T}
    (; ρ, α) = ps
    αρ = α*ρ
    return (αρ)*(1 + αρ*ρ/2*(kx^2 + ky^2))^(-(α + 1)/2)
end

"""
    StationaryRandomField(ps::AbstractPowerSpectrum, plan::StationaryRandomFieldPlan)

Creates a stationary random field defined by the power spectrum `ps` and the evaluation plan, which
is typically an FFT.
"""
struct StationaryRandomField{PS<:AbstractPowerSpectrum, P}
    ps::PS
    plan::P
end


function genfield(rf::StationaryRandomField, z::AbstractArray)
    ps = rf.ps
    (;kx, ky, p) = rf.plan
    e = executor(rf.plan)

    ns = similar(z , Complex{eltype(z)})
    @threaded e for i in eachindex(ky)
        for j in eachindex(kx)
            @inbounds ns[j, i] = ampspectrum(ps, kx[j], ky[i])*z[j,i]
        end
    end

    p*ns
    rast = (real.(ns) .+ imag.(ns))./sqrt(prod(size(z)))
    return rast
end

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
