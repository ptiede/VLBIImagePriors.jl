# sadness type piracy this needs to be fixed but it requires a rewrite of transform variables to be non-allocating
function ChainRulesCore.rrule(::typeof(TV.transform_with), flag::TV.LogJacFlag, t::TV.ArrayTransform{<:TV.ScalarTransform}, y::AbstractVector, index)
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
        Enzyme.autodiff(Reverse, _transform_with_loop!, Const, Const(flag), Const(t.transformation), Duplicated(x, dx), Duplicated(ysub, dy))
        Δy = zero(y)
        Δy[index:index+TV.dimension(t)-1] .= dy
        return NoTangent(), NoTangent(), NoTangent(), Δy, NoTangent()
    end
    return out, _transform_with_array
end

function _transform_with_loop!(flag, t, xout, ysub)
    xout[end] = 0
    for i in eachindex(ysub)
        x, ℓ, _ = TV.transform_with(flag, t, ysub, i)
        xout[i] = x
        xout[end] += ℓ
    end
    return nothing
end
