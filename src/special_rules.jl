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

EnzymeRules.inactive(::typeof(FFTW.assert_applicable), args...) = nothing
EnzymeRules.inactive_type(::Type{<:FFTW.Plan}) = true

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, 
                                      func::EnzymeRules.Const{typeof(FFTW.unsafe_execute!)},
                                      ::Type{RT},
                                      plan::EnzymeRules.Annotation{<:FFTW.cFFTWPlan{<:FFTW.fftwComplex, I, true}},
                                      X::EnzymeRules.Annotation{<:StridedArray{<:FFTW.fftwComplex}},
                                      Y::EnzymeRules.Annotation{<:StridedArray{<:FFTW.fftwComplex}}) where {I, RT}

    if !(typeof(plan) <: EnzymeRules.Const)
        throw(ArgumentError("Plan in FFTW.unsafe_execute! is not Const"))
    end

    cache_plan = plan

    # Now FFTW may overwrite the input to we need to check and cache it if needed
    cache_X = if EnzymeRules.overwritten(config)[3]
        X
    else
        nothing
    end 

    cache_Y = if EnzymeRules.overwritten(config)[4]
        Y
    else
        nothing
    end

    primal = if EnzymeRules.needs_primal(config)
        Y.val
    else
        nothing
    end

    shadow = if EnzymeRules.needs_shadow(config)
        Y.dval
    else
        nothing
    end

    # now evaluate the function
    func.val(plan.val, X.val, Y.val)

    cache = (cache_plan, cache_X, cache_Y)
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(config::EnzymeRules.RevConfig, 
                             func::EnzymeRules.Const{typeof(FFTW.unsafe_execute!)},
                             ::Type{RT}, cache,
                             plan::EnzymeRules.Annotation{<:FFTW.cFFTWPlan{<:FFTW.fftwComplex, I, true}},
                             X::EnzymeRules.Annotation{<:StridedArray{<:FFTW.fftwComplex}},
                             Y::EnzymeRules.Annotation{<:StridedArray{<:FFTW.fftwComplex}},
                            ) where {I, RT}

    # Now FFTW may overwrite the input to we need to check and cache it if needed
    N = EnzymeRules.width(config)
    Xfwd = EnzymeRules.overwritten(config)[3] ? cache[2] : X 
    Yfwd = EnzymeRules.overwritten(config)[4] ? cache[3] : Y
    planfwd = cache[1]
    if !isa(Y, EnzymeRules.Const)
        cache_plan, cache_X, cache_Y = cache
        # I need something to hold the cache
        for b in 1:N
            dY = N==1 ? Yfwd.dval : Yfwd.dval[b]
            dX = N==1 ? Xfwd.dval : X.dval[b]
            planfwd.val'*dY # compute the adjoint plan and move forward
            dY .*= length(dY)
            # dX .= tmp
            # dY .= zero(eltype(dX))
        end
    end
    return (nothing, nothing, nothing)
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
