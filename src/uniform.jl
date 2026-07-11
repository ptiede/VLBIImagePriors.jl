export ImageUniform, ImageSphericalUniform

"""
    ImageUniform(a::Number, b::Number, nx, ny)
    ImageUniform(nx, ny)

A uniform distribution over an `nx × ny` image where each pixel is
independently `Uniform(a, b)`. Thin wrapper around `VLBIUniform(a, b, (nx, ny))`
— see that constructor for the underlying `AffineDistribution{<:StdUniform, 2}`.
"""
ImageUniform(a::Number, b::Number, nx::Integer, ny::Integer) = VLBIUniform(a, b, (nx, ny))
ImageUniform(nx::Integer, ny::Integer) = VLBIUniform(0.0, 1.0, (nx, ny))

"""
    ImageSphericalUniform(nx, ny)

Construct a distribution where each image pixel is a 3-sphere uniform variable. This is
useful for polarization where the stokes parameters are parameterized on the 3-sphere.

Currently we use a struct of vectors memory layout. That is the image is described by three
matrices `(X,Y,Z)` grouped together as a tuple, where each matrix is one direction on the sphere, and
we require `norm((X,Y,Z)) == 1`.
"""
struct ImageSphericalUniform{T} <: Dists.ContinuousMatrixDistribution
    nx::Int
    ny::Int
end

ImageSphericalUniform(nx::Int, ny::Int) = ImageSphericalUniform{Float64}(nx, ny)

transport_node(d::ImageSphericalUniform, ::TVFlat) = TV.as(Matrix, spherical_unit_vector(2), d.nx, d.ny)


Base.size(d::ImageSphericalUniform) = (d.nx, d.ny)

function Dists.logpdf(::ImageSphericalUniform, X::NTuple{3, T}) where {T <: AbstractMatrix}
    return -length(X[1]) * log(4π)
end

function Dists.rand!(rng::Random.AbstractRNG, ::ImageSphericalUniform, X::NTuple{3, T}) where {T <: AbstractMatrix}
    for i in eachindex(X...)
        x = randn(rng)
        y = randn(rng)
        z = randn(rng)
        r = hypot(x, y, z)
        X[1][i] = x / r
        X[2][i] = y / r
        X[3][i] = z / r
    end
    return X
end

Dists.rand!(d::ImageSphericalUniform, X) = Dists.rand!(Random.default_rng(), d, X)

function Dists.rand(rng::Random.AbstractRNG, d::ImageSphericalUniform)
    r1 = randn(rng, d.nx, d.ny)
    r2 = randn(rng, d.nx, d.ny)
    r3 = randn(rng, d.nx, d.ny)

    for i in eachindex(r1, r2, r3)
        r = hypot(r1[i], r2[i], r3[i])
        r1[i] /= r
        r2[i] /= r
        r3[i] /= r
    end

    return (r1, r2, r3)
end


# struct ImageSphericalUniform{T} <: Dists.Distribution{Dists.ArrayLikeVariate{3}, Dists.Continuous}
#     nx::Int
#     ny::Int
# end

# ImageSphericalUniform(nx::Int, ny::Int) = ImageSphericalUniform{Float64}(nx, ny)

# HC.asflat(d::VLBIImagePriors.ImageSphericalUniform) = TV.as(Array, SphericalUnitVector{2}(), 3, d.nx, d.ny)


# Base.size(d::ImageSphericalUniform) = (3, d.nx, d.ny)

# function Dists._logpdf(d::ImageSphericalUniform, ::AbstractArray{T, 3}) where {T<:Number}
#     return -d.nx*d.ny*log(4π)
# end

# function Dists._rand!(rng::Random.AbstractRNG, ::ImageSphericalUniform, R::AbstractArray{T, 3}) where {T<:Number}
#     X = @view R[1,:,:]
#     Y = @view R[2,:,:]
#     Z = @view R[3,:,:]

#     for i in eachindex(X,Y,Z)
#         x = randn(rng)
#         y = randn(rng)
#         z = randn(rng)
#         r = hypot(x, y, z)
#         X[i] = x/r
#         Y[i] = y/r
#         Z[i] = z/r
#     end
#     return R
# end

# # Dists.rand!(d::ImageSphericalUniform, X) = Dists.rand!(Random.default_rng(), d, X)

# # function Dists.rand(rng::Random.AbstractRNG, d::ImageSphericalUniform)
# #     R = randn(rng, d.nx, d.ny, 3)
# #     X = @view R[:,:,1]
# #     Y = @view R[:,:,2]
# #     Z = @view R[:,:,3]
# #     for i in eachindex(X,Y,Z)
# #         r = hypot(X[i], Y[i], Z[i])
# #         X[i] /= r
# #         Y[i] /= r
# #         Z[i] /= r
# #     end

# #     return R
# # end
# #
# (ChainRules rrule removed — the log-density is constant; Enzyme/Reactant handle it.)
