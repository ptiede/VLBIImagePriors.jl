"""
    ImageSimplex(ny,nx)

This defines a transformation from ℝⁿ⁻¹ to the `n` probability simplex defined on an matrix
with dimension `ny×nx`. This is a more natural
transformation for rasterized images, which are most naturally represented as a matrix.

# Notes
Much of this code was inspired by [TransformVariables](https://github.com/tpapp/TransformVariables.jl).
However, we have specified custom `rrules` using Enzyme as a backend. This allowed the simplex
transform to be used with Zygote and we achieved an order of magnitude speedup when computing
the pullback of the simplex transform.
"""
struct ImageSimplex <: TV.VectorTransform
    n::Int
    dims::Dims{2}
    function ImageSimplex(dims::Dims{2})
        n = prod(dims)
        @argcheck n ≥ 1 "Dimension of simplex should be positive"
        new(prod(dims), dims)
    end
end

function ImageSimplex(nx::Int, ny::Int)
    return ImageSimplex((nx, ny))
end

HC.dimension(t::ImageSimplex) = t.n - 1

function simplex_fwd(flag::TV.LogJacFlag, t::ImageSimplex, y::AbstractArray)
    x = similar(y, t.n+1)
    flagbool = flag isa TV.NoLogJac ? true : false
    simplex_fwd!(x, y, flagbool)
    return x
end

function simplex_fwd(t::ImageSimplex, y::AbstractArray)
    x = similar(y, t.n+1)
    simplex_fwd!(x, y, true)
    return reshape(@view(x[begin:end-1]), t.dims[1], t.dims[2])
end

function ChainRulesCore.rrule(::typeof(simplex_fwd), flag::TV.LogJacFlag, t::ImageSimplex, y::AbstractArray)
    x = simplex_fwd(flag, t, y)
    py = ProjectTo(y)
    function _simplex_fwd_pullback(ΔX)
        Δf = NoTangent()
        Δflag = NoTangent()
        Δt = NoTangent()
        dx = zero(ΔX)
        dx .= unthunk(ΔX)
        Δy = zero(y)

        f = (flag isa TV.NoLogJac) ? true : false
        #copy is because sometimes y is a subarray :(
        Enzyme.autodiff(simplex_fwd!, Const, Duplicated(x, dx), Duplicated(copy(y), Δy), Const(f))
        return (Δf, Δflag, Δt, py(Δy))
    end
    return x, _simplex_fwd_pullback
end

# ReverseDiff.@grad_from_chainrules simplex_fwd(flag, t, y::TrackedArray)


function simplex_fwd!(x::AbstractArray, y::AbstractArray, flag::Bool)
    #@argcheck length(x) == length(y) + 2
    n = length(x)-1
    x[end] = zero(eltype(x))
    stick = one(eltype(x))
    for i in eachindex(y)
        z = logistic(y[i] - log(n-i))
        x[i] = stick*z

        if !(flag)
            x[end] += log(stick) + log(z*(1-z))
        end
        stick *= 1-z
    end
    x[end-1] = stick

    return nothing
end

function TV.transform_with(flag::TV.LogJacFlag, t::ImageSimplex, y::AbstractVector, index)
    n = t.n
    x = simplex_fwd(flag, t, @view y[index:index+t.n-2])
    ℓ = TV.logjac_zero(flag, eltype(y))
    if !(flag isa TV.NoLogJac)
        ℓ = x[end]
    end
    return (reshape(@view(x[begin:end-1]), t.dims[1], t.dims[2]), ℓ, index+n-1)
end

TV.inverse_eltype(::ImageSimplex, y::AbstractMatrix) = TV.extended_eltype(y)


function TV.inverse_at!(x::AbstractVector, index, t::ImageSimplex, y::AbstractMatrix)
    return TV.inverse_at!(x, index, TV.UnitSimplex(t.n), reshape(y,:))
end
