using VLBIImagePriors
import VLBIImagePriors as RP
using Plots
using Distributions
import TransformVariables as TV
using ChainRulesCore

x = -3:0.5:3
f(x,y) = exp(-(x^2+y^2)/2)
I = f.(x', x)
I /= sum(I)

heatmap(x,x,I)
t = RP.ImageSimplex(size(I)...)
y = TV.inverse(t, I)

dy = MvNormal(ones(length(y)))

function alr(t, y)
    x = similar(y, length(y)+1)
    tot = one(eltype(y))
    x[end] = 1.0
    @simd for i in eachindex(y)
        @inbounds x[i] = exp(y[i])
        @inbounds tot += x[i]
    end
    return reshape(x./tot, t.dims...)
end
