export SphericalUnitVector, AngleTransform


"""
    AngleTransform

A transformation that moves two vector `x` and `y` to an angle `θ`.  Note that is `x` and
`y` are normally distributed then the resulting distribution in `θ` is uniform on the circle.
"""
struct AngleTransform <: TV.VectorTransform end

TV.dimension(t::AngleTransform) = 2

function TV.transform_with(flag::TV.LogJacFlag, ::AngleTransform, y::AbstractVector, index)
    T = TV.robust_eltype(y)
    ℓi = TV.logjac_zero(flag, T)
    x1 = y[index]
    x2 = y[index+1]
    r = sqrt(x1^2 + x2^2)
    # Use log-normal with μ = 0, σ = 1/4
    σ = oftype(r, 1/4)
    if !(flag isa TV.NoLogJac)
        lr = log(r)
        ℓi = -lr^2*inv(2*σ^2) - lr
    end

    return atan(x1, x2), ℓi, index+2
end

function ChainRulesCore.rrule(::typeof(TV.transform_with), flag::TV.LogJacFlag, t::TV.ArrayTransformation{<:AngleTransform}, y::AbstractVector, index)
    out = TV.transform_with(flag, t, y, index)
    py = ProjectTo(y)
    function _transform_with_arrayangle_pb(Δ)
        Δy = zero(y)
        Δx = Δ[1]
        Δℓ = Δ[2]
        for i in index:2:(index+TV.dimension(t)-1)
            y1 = y[i]
            y2 = y[i+1]
            r = hypot(y1, y2)
            ix = (i+2-index)÷2
            Δy[i] =  Δx[ix]*y2/r^2
            Δy[i+1]   = -Δx[ix]*y1/r^2
            if !(flag isa TV.NoLogJac)
                σ = oftype(r, 1/4)
                dpdr = -inv(r)*(log(r)/σ^2 + 1)
                Δy[i] += Δℓ*dpdr*y1/r
                Δy[i+1] += Δℓ*dpdr*y2/r
            end
        end
        return NoTangent(), NoTangent(), NoTangent(), py(Δy), NoTangent()
    end
    return out, _transform_with_arrayangle_pb
end

function TV.inverse_at!(x, index, ::AngleTransform, y::Number)
    x[index:index+1] .= sincos(y)
    return index+2
end

TV.inverse_eltype(::AngleTransform, x) = float(typeof(x))


"""
    SphericalUnitVector{N}()

A transformation from a set of `N+1` vectors to the `N` sphere. The set of `N+1` vectors
are inherently assumed to be `N+1` a distributed according to a unit multivariate Normal
distribution.

# Notes
For more information about this transformation see the Stan [manual](https://mc-stan.org/docs/reference-manual/unit-vector.html).
In the future this may be depricated when [](https://github.com/tpapp/TransformVariables.jl/pull/67) is merged.
"""
struct SphericalUnitVector{N} <: TV.VectorTransform
    function SphericalUnitVector{N}() where {N}
        TV.@argcheck N ≥ 1 "Dimension should be positive."
        new{N}()
    end
end

TV.dimension(::SphericalUnitVector{N}) where {N} = N+1

TV.inverse_eltype(::SphericalUnitVector{N}, x) where {N} = TV.inverse_eltype(x)
TV.inverse_eltype(::TV.ArrayTransformation{<:SphericalUnitVector}, x::NTuple{N, T}) where {N,T} = float(eltype(first(x)))


function TV.transform_with(flag::TV.LogJacFlag, ::SphericalUnitVector{N}, y::AbstractVector, index) where {N}
    T = TV.robust_eltype(y)
    index2 = index + N +1
    # normalized vector
    vy = NTuple{N+1,T}(@view(y[index:(index2-1)]))
    sly = sum(abs2, vy)

    x = sly > 0 ? vy ./ sqrt(sly) : ntuple(i->(i==1 ? one(T) : zero(T)), N+1)
    # jacobian term
    ℓi = TV.logjac_zero(flag, T)

    if !(flag isa TV.NoLogJac)
        ℓi -= sly/2
    end

    return x, ℓi, index2
end

function TV.transform_with(flag::TV.LogJacFlag, t::TV.ArrayTransformation{<:SphericalUnitVector{N}}, y::AbstractVector, index) where {N}
    (;inner_transformation, dims) = t
    # NOTE not using index increments as that somehow breaks type inference
    d = TV.dimension(inner_transformation) # length of an element transformation
    len = prod(dims)              # number of elements
    𝐼 = reshape(range(index; length = len, step = d), dims)
    xℓ = map(index -> ((x, ℓ, _) = TV.transform_with(flag, inner_transformation, y, index); (x, ℓ)), 𝐼)
    ℓz = TV.logjac_zero(flag, TV.robust_eltype(y))
    index′ = index + d * len
    ntuple(i->getindex.(first.(xℓ), i), N+1), isempty(xℓ) ? ℓz : ℓz + sum(last, xℓ), index′
end



function TV.inverse_at!(x::AbstractArray, index, t::TV.ArrayTransformation{<:SphericalUnitVector{N}}, y::NTuple) where {N}
    @assert length(y) == N + 1 "Length of y must be equal to N + 1"
    index2 = index + TV.dimension(t)
    ix = 1
    for i in index:(N+1):(index+TV.dimension(t)-1)
        for j in 1:(N+1)
            x[i+j-1] = y[j][ix]
        end
        ix+=1
    end
    return index2
end

function ChainRulesCore.rrule(::typeof(TV.transform_with), flag::TV.LogJacFlag, T::SphericalUnitVector{N}, y::AbstractVector, index) where {N}
    res = TV.transform_with(flag, T, y, index)
    py = ProjectTo(y)
    function _spherical_unit_transform(Δ)
        Δf = NoTangent()
        Δflag = NoTangent()
        ΔT = NoTangent()
        Δindex = NoTangent()
        Δy = zero(y)
        ysub = @view(y[index:(index+N)])
        ny = norm(ysub)
        Δy[index:(index+N)] .= Δ[1]./ny .- (sum(Δ[1].*ysub).*ysub/ny^3)
        if !(flag isa TV.NoLogJac)
            Δy[index:(index+N)] .+= -Δ[2].*ysub
        end
        return Δf, Δflag, ΔT, py(Δy), Δindex
    end
    return res, _spherical_unit_transform
end

function ChainRulesCore.rrule(::typeof(TV.transform_with), flag::TV.LogJacFlag, t::TV.ArrayTransformation{<:SphericalUnitVector{N}}, y::AbstractVector, index) where {N}
    out = TV.transform_with(flag, t, y, index)
    py = ProjectTo(y)
    function _transform_with_arraysuv_pb(Δ)
        Δy = zero(y)
        Δx = Δ[1]
        Δℓ = Δ[2]
        for (ix, i) in enumerate(index:(N+1):(index+TV.dimension(t)-N-1))
            ysub = @view y[i:(i+N)]
            ny = norm(ysub)
            dx = ntuple(i->Δx[i][ix], Val(N+1))
            # Δy[i:(i+N)] .= dx./ny .- (sum(dx.*ysub).*ysub./ny^3)
            red = mapreduce(.*, +, dx, ysub)
            Δy[i:(i+N)] .= dx./ny .- red.*ysub./ny^3
            if !(flag isa TV.NoLogJac)
                Δy[i:(i+N)] .= @view(Δy[i:(i+N)]) .- Δℓ.*ysub
            end
        end
        return NoTangent(), NoTangent(), NoTangent(), py(Δy), NoTangent()
    end
    return out, _transform_with_arraysuv_pb
end
