export alr, alr!, alrinv

"""
    alr!(x, y)

Computes the additive logit transform inplace. This is useful for moving
from a normal distribution to the logistic normal, which lives
on the simplex.


# Notes
This function is mainly to transform the GaussMarkovRF to live on the simplex.
In order to preserve the nice properties of the GRMF namely the log det we
only use `y[begin:end-1]` elements and the last one is not included in the transform.
This shouldn't be a problem since the additional parameter is just a dummy in that case
and the Gaussian prior should ensure it is easy to sample from.

"""
function alr!(x, y)
    x[end] = 1
    tot = one(eltype(x))
    # Skip the last element
    x[begin:end-1] .= exp.(@view y[begin:end-1])
    tot = sum(x)
    itot = inv(tot)
    x .*= itot
    nothing
end

"""
    alr(y)

Computes the additive logit transform of the parameter `y`. This is useful for moving
from a normal distribution to the logistic normal, which lives
on the simplex.

```julia-repl
julia> y = randn(10)
julia> x = alr(y)
julia> sum(x) ≈ 1
true

# Notes
This function is mainly to transform the GaussMarkovRF to live on the simplex.
In order to preserve the nice properties of the GRMF namely the log det we
only use `y[begin:end-1]` elements and the last one is not included in the transform.
This shouldn't be a problem since the additional parameter is just a dummy in that case
and the Gaussian prior should ensure it is easy to sample from.
"""
function alr(y)
    x = similar(y)
    alr!(x, y)
    return x
end

using ChainRulesCore
function ChainRulesCore.rrule(::typeof(alr), y)
    x = alr(y)
    function _alr_pullback(Δ)
        Δf = NoTangent()
        dx = zero(x)
        dx .= unthunk(Δ)
        Δy = zero(y)


        Enzyme.autodiff(Reverse, alr!, Const, Duplicated(x, dx), Duplicated(y, Δy))
        return (Δf, Δy)
    end
    return x, _alr_pullback
end

"""
    alrinv(x)

Compute the inverse alr transform. Note that we fill the last element with zero.
"""
function alrinv(x)
    @assert sum(x) ≈ 1 "$(sum(x)) is not unity"
    y = zero(x)
    y[begin:end-1] .= log.(@view x[begin:end-1]) .- log(x[end])
    return y
end
