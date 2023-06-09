export CenteredRegularizer, CenterImage


"""
    CenteredRegularizer(x, y, σ, p)

Regularizes a general image prior `p` such that the center of light is close the the origin
of the imag. After regularization the log density of the prior is modified to

```math
    \\log p(I) \\to \\log p(I) - \\frac{(x_C^2 + y_C^2)^2}{2\\sigma\\^2} N_x N_y
```
where `N_x` and `N_y` are the number of pixels in the `x` and `y` direction of the image,
and ``x_C, y_C`` are the center of light of the image `I`.
"""
struct CenteredRegularizer{I,S,D} <: Dists.ContinuousMatrixDistribution
    x::I
    y::I
    σ::S
    distI::D
    function CenteredRegularizer(x, y, σ, dist)
        @assert length(x) == size(dist, 1) "Number of pixels in x direction does not match `x`"
        @assert length(y) == size(dist, 2) "Number of pixels in x direction does not match `y`"

        xp, yp = promote(x, y)
        return new{typeof(xp), typeof(σ), typeof(dist)}(x, y, σ, dist)
    end
end

Base.size(d::CenteredRegularizer) = size(d.distI)

HC.asflat(d::CenteredRegularizer) = HC.asflat(d.distI)

Dists.insupport(d::CenteredRegularizer, x::AbstractMatrix) = Dists.insupport(d.distI, x)

function lcol(d::CenteredRegularizer, img)
    dx = zero(eltype(img))
    dy = zero(eltype(img))
    for i in axes(img, 2), j in axes(img,1)
        dx += d.x[j]*img[j,i]
        dy += d.y[i]*img[j,i]
    end
    return -(dx^2 + dy^2)/(2*d.σ^2)*prod(size(img))
end

function ChainRulesCore.rrule(::typeof(lcol), d::CenteredRegularizer, img)
    f = lcol(d, img)
    pimg = ProjectTo(img)
    function _lcol_pullback(Δ)
        dimg = zero(img)
        autodiff(Reverse, lcol, Active, Const(d), Duplicated(copy(img), dimg))
        return (NoTangent(), NoTangent(), pimg(Δ*dimg))
    end
    return (f, _lcol_pullback)
end

function Dists._logpdf(d::CenteredRegularizer, x::AbstractMatrix{<:Real})
    return Dists.logpdf(d.distI, x) + lcol(d, x)
end

struct CenterImage{K,D}
    kernel::K
    dims::D
    function CenterImage(X, Y)
        # normalize by the step to make sure the numerics are alright
        x = ones(length(Y)).*X'./step(X)
        y = Y.*ones(length(X))'./step(Y)
        XY = zeros(2, length(x))
        XY[1,:] .= reshape(x, :)
        XY[2,:] .= reshape(y, :)
        C = nullspace(XY)
        dims = (length(X), length(Y))
        return new{typeof(C), typeof(dims)}(C*C', dims)
    end
end

"""
    CenterImage(X, Y)
    CenterImage(img::IntensityMap)
    CenterImage(grid::ComradeBase.AbstractDims)

Constructs a projections operator that will take an arbritrary image and return
a transformed version whose centroid is at the origin.

# Arguments
  - `X`, `Y`: the iterators for the image pixel locations
  - `img`: A `IntensityMap` whose grid is the same as the image you want to center
  - `grid`: The grid that the image is defined on.


# Example
```julia-repl
julia> using ComradeBase: centroid, imagepixels
julia> grid = imagepixels(μas2rad(100.0), μas2rad(100.0), 256, 256)
julia> K = CenterImage(grid)
julia> img = IntensityMap(rand(256, 256), grid)
julia> centroid(img)
(1.34534e-10, 5.23423e-11)
julia> cimg = K(img)
julia> centroid(cimg)
(1.231e-14, -2.43e-14)
```

# Note
This center image works using a linear projection operator. This means that is does
not necessarily preserve the image total flux and the positive definiteness of the image.
In practise we find that the deviation from the original flux, and the amount of negative
flux is small.

# Explanation
The centroid is constructed by
```
X ⋅ I = 0
Y ⋅ I = 0
```
where `I` is the flattened image of length `N`, and `X` and `Y` are the image pixel locations.
This can be simplified into the matrix equation
```
XY ⋅ I = 0
```
where `XY` is the `2×N` matrix whose first for is given by `X` and second is `Y`. The space
of centered images is then given by the null space of `XY`. Given this we can form a matrix `C`
which is the kernel matrix of the nullspace whose columns are the orthogonal basis vectors of
the null space of `XY`. Using this we can construct a centered image projection opertor by
```
K = CC'.
```
Our centered image is then given by `I₀ = KI`.
"""
CenterImage(img::ComradeBase.IntensityMap) = CenterImage(img.X, img.Y)
CenterImage(img::ComradeBase.AbstractDims) = CenterImage(img.X, img.Y)

function (c::CenterImage)(img::AbstractMatrix)
    return center_kernel(c.kernel, img)
end

center_kernel(K, img) = reshape(K*vec(img), size(img))

using ChainRulesCore
function ChainRulesCore.rrule(::typeof(center_kernel), K::AbstractMatrix, img::AbstractMatrix)
    out = center_kernel(K, img)
    pimg = ProjectTo(img)
    function _center_pullback(Δ)
        Δc = NoTangent()
        ΔK  = NoTangent()
        Δimg = pimg(reshape(K'*vec(Δ), size(img)))
        return Δc, ΔK, Δimg
    end
    return out, _center_pullback
end
