export to_real, to_simplex, CenteredLR, AdditiveLR

abstract type LogRatioTransform end

struct CenteredLR <: LogRatioTransform end
struct AdditiveLR <: LogRatioTransform end

function to_simplex(t::LogRatioTransform, x)
    y = similar(x)
    to_simplex!(t::LogRatioTransform, y, x)
    return y
end

function to_simplex!(::AdditiveLR, y, x)
    alrinv!(y, x)
    return nothing
end

function to_simplex!(::CenteredLR, y, x)
    clrinv!(y, x)
    return nothing
end

function to_real(t::LogRatioTransform, y)
    x = similar(y)
    to_real!(t, x, y)
    return x
end

function to_real!(::AdditiveLR, x, y)
    alr!(x, y)
    return nothing
end

function to_real!(::CenteredLR, x, y)
    clr!(x, y)
    return nothing
end

"""
    clrinv!(x, y)

Computes the additive logit transform inplace. This converts from
ℜⁿ → Δⁿ where Δⁿ is the n-simplex


# Notes
This function is mainly to transform the GaussMarkovRF to live on the simplex.
"""
function clrinv!(x, y)
    x .= exp.(y)
    tot = sum(x)
    x .= x./tot
    nothing
end

"""
    alrinv!(x, y)

Computes the additive logit transform inplace. This converts from
ℜⁿ → Δⁿ where Δⁿ is the n-simplex


# Notes
This function is mainly to transform the GaussMarkovRF to live on the simplex.
In order to preserve the nice properties of the GRMF namely the log det we
only use `y[begin:end-1]` elements and the last one is not included in the transform.
This shouldn't be a problem since the additional parameter is just a dummy in that case
and the Gaussian prior should ensure it is easy to sample from.

"""
function alrinv!(x, y)
    x[end] = 1
    # Skip the last element
    x[begin:end-1] .= exp.(@view y[begin:end-1])
    tot = sum(x)
    itot = inv(tot)
    x .*= itot
    nothing
end

using ChainRulesCore
function ChainRulesCore.rrule(::typeof(to_simplex), t::LogRatioTransform, y)
    x = to_simplex(t, y)
    function _to_simplex_pullback(Δ)
        Δf = NoTangent()
        dx = zero(x)
        dx .= unthunk(Δ)
        Δy = zero(y)


        Enzyme.autodiff(Reverse, to_simplex!, Const, Const(t), Duplicated(x, dx), Duplicated(y, Δy))
        return (Δf, NoTangent(), Δy)
    end
    return x, _to_simplex_pullback
end

"""
    clr!(x, y)

Compute the inverse alr transform. That is `x` lives in ℜⁿ and `y`, lives in Δⁿ
"""
function clr!(x, y)
    @assert sum(y) ≈ 1 "$(sum(x)) is not unity"
    x .= log.(y)
    x .= x .- sum(x)/length(x)
    return nothing
end

"""
    alr!(x, y)

Compute the inverse alr transform. That is `x` lives in ℜⁿ and `y`, lives in Δⁿ
"""
function alr!(x, y)
    @assert sum(y) ≈ 1 "$(sum(y)) is not unity"
    x[begin:end-1] .= log.(@view y[begin:end-1]) .- log(y[end])
    x[end] = 0
    return nothing
end
