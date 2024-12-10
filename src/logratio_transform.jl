export to_real, to_simplex, CenteredLR, AdditiveLR

abstract type LogRatioTransform end

"""
    CenteredLR <: LogRatioTransform

Defines the centered log-ratio transform. The `clr` transformation moves from the
simplex Sⁿ → Rⁿ and is given by
```
clr(x) = [log(x₁/g(x)) ... log(xₙ/g(x))]
```
where `g(x) = (∏xᵢ)ⁿ⁻¹` is the geometric mean. The inverse transformation is given by
the softmax function and is only defined on a subset of the domain otherwise it is not injective
```
clr⁻¹(x) = exp.(x)./sum(exp, x).
```

# Notes
As mentioned above this transformation is bijective on the entire codomain of the function.
However, unlike the additive log-ratio transform it does not treat any pixel as being special.
"""
struct CenteredLR <: LogRatioTransform end

"""
    AdditiveLR <: LogRatioTransform

Defines the additive log-ratio transform. The `clr` transformation moves from the
simplex Sⁿ → ``R^{n-1}`` and is given by
```
alr(x) = [log(x₁/xₙ) ... log(xₙ/xₙ)],
```
where `g(x) = (∏xᵢ)ⁿ⁻¹` is the geometric mean. The inverse transformation is given by
```
alr⁻¹(x) = exp.(x)./(1 + sum(x[1:n-1])).
```
"""
struct AdditiveLR <: LogRatioTransform end

"""
    to_simplex(t::LogRatioTransform, x)

Transform the vector `x` assumed to be a real valued array to the simplex using the
log-ratio transform `t`. See `subtypes(LogRatioTransform)` for a list of possible
transformations.

The inverse of this transform is given by [`to_real(t, y)`](@ref) where `y` is a vector that
sums to unity, i.e. it lives on the simplex.

# Example
```julia
julia> x = randn(100)
julia> to_simplex(CenteredLR(), x)
julia> to_simplex(AdditiveLR(), x)


```
"""
@inline function to_simplex(t::LogRatioTransform, x)
    y = similar(x)
    to_simplex!(t::LogRatioTransform, y, x)
    return y
end

"""
    to_simplex!(t::LogRatioTransform, y, x)

Transform the vector `x` assumed to be a real valued array to the simplex using the
log-ratio transform `t` and stores the value in `y`.

The inverse of this transform is given by [`to_real!(t, x, y)`](@ref) where `y` is a vector that
sums to unity, i.e. it lives on the simplex.

# Example
```julia
julia> x = randn(100)
julia> to_simplex(CenteredLR(), x)
julia> to_simplex(AdditiveLR(), x)


```
"""
@inline function to_simplex!(::AdditiveLR, y, x)
    alrinv!(y, x)
    return nothing
end

@inline function to_simplex!(::CenteredLR, y, x)
    clrinv!(y, x)
    return nothing
end

"""
    to_real(t::LogRatioTransform, y)

Transform the value `u` that lives on the simplex to a value in the real vector space.
See `subtypes(LogRatioTransform)` for a list of possible
transformations.

The inverse of this transform is given by [`to_simplex(t, x)`](@ref).

# Example
```julia
julia> y = randn(100)
julia> y .= y./sum(y)
julia> to_real(CenteredLR(), y)
julia> to_real(AdditiveLR(), y)
"""
@inline function to_real(t::LogRatioTransform, y)
    # @argcheck sum(y) ≈ 1
    x = similar(y)
    to_real!(t, x, y)
    return x
end

@inline function to_real!(::AdditiveLR, x, y)
    alr!(x, y)
    return nothing
end

@inline function to_real!(::CenteredLR, x, y)
    clr!(x, y)
    return nothing
end


"""
    clrinv!(x, y)

Computes the additive logit transform inplace. This converts from
ℜⁿ → Δⁿ where Δⁿ is the n-simplex



# Notes
This function is mainly to transform the GaussMarkovRandomField to live on the simplex.
"""
@inline @fastmath function clrinv!(x, y)
    maxx = _noadmaximum(y) # We don't need to AD since the gradient is independent of this
    x .= exp.(y .- maxx) # This is for numerical stability. Prevents overflow
    tot = _fastsum(x)
    x .= x./tot
    nothing
end

@noinline function _noadmaximum(x)
    return maximum(x)
end
EnzymeRules.inactive(::typeof(_noadmaximum), args...) = nothing

@inline function _fastsum(x)
    tot = zero(eltype(x))
    @simd for i in eachindex(x)
        tot += x[i]
    end
    return tot
end

"""
    alrinv!(x, y)

Computes the additive logit transform inplace. This converts from
ℜⁿ → Δⁿ where Δⁿ is the n-simplex


# Notes
This function is mainly to transform the GaussMarkovRandomField to live on the simplex.
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



checkx(x) = @argcheck sum(x) ≈ 1
EnzymeRules.inactive(::typeof(checkx), args...) = nothing

"""
    clr!(x, y)

Compute the inverse alr transform. That is `x` lives in ℜⁿ and `y`, lives in Δⁿ
"""
function clr!(x, y)
    checkx(y)
    x .= log.(y)
    x .= x .- sum(x)/length(x)
    return nothing
end

"""
    alr!(x, y)

Compute the inverse alr transform. That is `x` lives in ℜⁿ and `y`, lives in Δⁿ
"""
function alr!(x, y)
    checkx(y)
    x[begin:end-1] .= log.(@view y[begin:end-1]) .- log(y[end])
    x[end] = 0
    return nothing
end
