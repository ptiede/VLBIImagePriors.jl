export CenteredImage

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
        dx += d.x[i]*img[j,i]
        dy += d.y[j]*img[j,i]
    end
    return -(dx^2 + dy^2)/(2*d.σ^2)*prod(size(img))
end

function ChainRulesCore.rrule(::typeof(lcol), d::CenteredImage, img)
    f = lcol(d, img)
    function _lcol_pullback(Δ)
        dimg = zero(img)
        autodiff(Reverse, lcol, Active, Const(d), Duplicated(copy(img), dimg))
        return (NoTangent(), NoTangent(), Δ*dimg)
    end
    return (f, _lcol_pullback)
end

function Dists._logpdf(d::CenteredImage, x::AbstractMatrix{<:Real})
    return Dists.logpdf(d.distI, x) + lcol(d, x)
end
