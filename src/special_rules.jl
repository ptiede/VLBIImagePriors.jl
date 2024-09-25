function _transform_with_loop!(flag, t, xout, ysub)
    xout[end] = 0
    for i in eachindex(ysub)
        x, ℓ, _ = TV.transform_with(flag, t, ysub, i)
        xout[i] = x
        if flag === TV.LogJac()
            xout[end] += ℓ
        end
        # xout[end] += ℓ
    end
    return nothing
end


# This doesn't work for some reason and drops gradients
# @noinline function _enzyme_trf_lj2!(out1::Ref, out2::Ref, flag, in::AbstractVector, t::TV.TransformTuple, index)
#     ylj = TV.transform_with(flag, t, in, index)
#     out1[] = ylj[1]
#     out2[] = ylj[2]
#     return nothing
# end

# _detangent(x) = x
# _detangent(x::AbstractArray) = x
# _detangent(x::FillArrays.Fill) = (fill(first(x), size(x)))
# _detangent(x::Tuple) = map(_detangent, x)
# _detangent(x::NamedTuple) = map(_detangent, x)
# _detangent(x::Tangent) = _detangent(getfield(x, :backing))


# function ChainRulesCore.rrule(::typeof(TV.transform_with), f::TV.LogJacFlag, t::TV.TransformTuple{<:NamedTuple{N}}, x::AbstractVector, index) where {N}
#     out = TV.transform_with(f, t, x, index)
#     pr = ProjectTo(x)
#     function _trf_lj_pullback_nt(Δ)
#         Δlj = unthunk(Δ[2])
#         Δy = unthunk(Δ[1])
#         dy = _detangent(Δy)
#         Δx = zero(x)
#         Enzyme.autodiff(Reverse, _enzyme_trf_lj2!, Const, Duplicated(Ref(out[1]), Ref(dy)), Duplicated(Ref(zero(Δlj)), Ref(Δlj)), Const(f), Duplicated(x, Δx), Const(t), Const(index))

#         return NoTangent(), NoTangent(), NoTangent(), pr(Δx), NoTangent()
#     end
#     return out, _trf_lj_pullback_nt
# end
