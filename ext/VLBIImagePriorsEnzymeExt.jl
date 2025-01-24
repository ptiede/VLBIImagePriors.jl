module VLBIImagePriorsEnzymeExt

using Enzyme
using VLBIImagePriors
using ChainRulesCore 
import TransformVariables as TV
using AbstractFFTs

function ChainRulesCore.rrule(::typeof(VLBIImagePriors.lcol), d::CenteredRegularizer, img)
    f = VLBIImagePriors.lcol(d, img)
    pimg = ProjectTo(img)
    function _lcol_pullback(Δ)
        dimg = zero(img)
        autodiff(Reverse, VLBIImagePriors.lcol, Active, Const(d), Duplicated(copy(img), dimg))
        return (NoTangent(), NoTangent(), pimg(Δ*dimg))
    end
    return (f, _lcol_pullback)
end

function ChainRulesCore.rrule(::typeof(TV.transform_with), flag::TV.LogJacFlag, t::TV.ArrayTransformation{<:TV.ScalarTransform}, y::AbstractVector, index)
    out = TV.transform_with(flag, t, y, index)
    function _transform_with_array(Δ)
        ysub = y[index:index+TV.dimension(t)-1]
        Δx = unthunk(Δ[1])
        Δlj = unthunk(Δ[2])
        dy = zero(ysub)
        x = similar(ysub, length(ysub)+1)
        dx = similar(ysub, length(ysub)+1)
        if Δx isa ZeroTangent
            dx[begin:end-1] .= 0.0
        else
            dx[begin:end-1] .= reshape(Δx, :)
        end

        dx[end] = Δlj
        Enzyme.autodiff(Reverse, VLBIImagePriors._transform_with_loop!, Const, Const(flag), Const(t.inner_transformation), Duplicated(x, dx), Duplicated(ysub, dy))
        Δy = zero(y)
        Δy[index:index+TV.dimension(t)-1] .= dy
        return NoTangent(), NoTangent(), NoTangent(), Δy, NoTangent()
    end
    return out, _transform_with_array
end

function ChainRulesCore.rrule(::typeof(VLBIImagePriors.sq_manoblis), d::MarkovRandomFieldGraph, ΔI, ρ)
    s = VLBIImagePriors.sq_manoblis(d, ΔI, ρ)
    prI = ProjectTo(ΔI)
    function _sq_manoblis_pullback(Δ)
        Δf = NoTangent()
        Δd = NoTangent()
        dI = zero(ΔI)

        ((_, _, dρ), ) = autodiff(Reverse, VLBIImagePriors.sq_manoblis, Active, Const(d), Duplicated(ΔI, dI), Active(ρ))

        dI .= Δ.*dI
        return Δf, Δd, prI(dI), Δ*dρ
    end
    return s, _sq_manoblis_pullback
end

function ChainRulesCore.rrule(::typeof(VLBIImagePriors.simplex_fwd), flag::TV.LogJacFlag, t::VLBIImagePriors.ImageSimplex, y::AbstractArray)
    x = VLBIImagePriors.simplex_fwd(flag, t, y)
    py = ProjectTo(y)
    function _simplex_fwd_pullback(ΔX)
        Δf = NoTangent()
        Δflag = NoTangent()
        Δt = NoTangent()
        dx = zero(x)
        if !(unthunk(ΔX) isa AbstractZero)
            dx .= unthunk(ΔX)
        end
        Δy = zero(y)

        f = (flag isa TV.NoLogJac) ? true : false
        #copy is because sometimes y is a subarray :(
        Enzyme.autodiff(Reverse, VLBIImagePriors.simplex_fwd!, Const, Duplicated(x, dx), Duplicated(copy(y), Δy), Const(f))
        return (Δf, Δflag, Δt, py(Δy))
    end
    return x, _simplex_fwd_pullback
end

using ChainRulesCore
function ChainRulesCore.rrule(::typeof(VLBIImagePriors.to_simplex), t::VLBIImagePriors.LogRatioTransform, y)
    x = VLBIImagePriors.to_simplex(t, y)
    function _to_simplex_pullback(Δ)
        Δf = NoTangent()
        dx = zero(x)
        dx .= unthunk(Δ)
        Δy = zero(y)
        Enzyme.autodiff(Reverse, VLBIImagePriors.to_simplex!, Const, Const(t), Duplicated(x, dx), Duplicated(y, Δy))
        return (Δf, NoTangent(), Δy)
    end
    return x, _to_simplex_pullback
end

function ChainRulesCore.rrule(::typeof(VLBIImagePriors.to_real), t::VLBIImagePriors.LogRatioTransform, y)
    x = VLBIImagePriors.to_real(t, y)
    function _to_simplex_pullback(Δ)
        Δf = NoTangent()
        dx = zero(x)
        dx .= unthunk(Δ)
        Δy = zero(y)
        Enzyme.autodiff(Reverse, VLBIImagePriors.to_real!, Const, Const(t), Duplicated(x, dx), Duplicated(y, Δy))
        return (Δf, NoTangent(), Δy)
    end
    return x, _to_simplex_pullback
end




end