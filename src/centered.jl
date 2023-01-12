export CenteredImage


"""
    CenteredImage(x, y, σ, p)

Regularizes a general image prior `p` such that the center of light is close the the origin
of the imag. After regularization the log density of the prior is modified to

```math
    \\log p(I) \\to \\log p(I) - \\frac{(x_C^2 + y_C^2)^2}{2\\sigma\\^2} N_x N_y
```
where `N_x` and `N_y` are the number of pixels in the `x` and `y` direction of the image,
and ``x_C, y_C`` are the center of light of the image `I`.
"""
struct CenteredImage{I,S,D} <: Dists.ContinuousMatrixDistribution
    x::I
    y::I
    σ::S
    distI::D
end

Base.size(d::CenteredImage) = size(d.distI)

HC.asflat(d::CenteredImage) = HC.asflat(d.distI)

Dists.insupport(d::CenteredImage, x::AbstractMatrix) = Dists.insupport(d.distI, x)

function lcol(d::CenteredImage, img)
    dx = zero(eltype(img))
    dy = zero(eltype(img))
    for i in axes(img, 2), j in axes(img,1)
        dx += d.x[j]*img[j,i]
        dy += d.y[i]*img[j,i]
    end
    return -(dx^2 + dy^2)/(2*d.σ^2)*prod(size(img))
end

function ChainRulesCore.rrule(::typeof(lcol), d::CenteredImage, img)
    f = lcol(d, img)
    pimg = ProjectTo(img)
    function _lcol_pullback(Δ)
        dimg = zero(img)
        autodiff(Reverse, lcol, Active, Const(d), Duplicated(copy(img), dimg))
        return (NoTangent(), NoTangent(), pimg(Δ*dimg))
    end
    return (f, _lcol_pullback)
end

function Dists._logpdf(d::CenteredImage, x::AbstractMatrix{<:Real})
    return Dists.logpdf(d.distI, x) + lcol(d, x)
end
