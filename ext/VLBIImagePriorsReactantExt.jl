module VLBIImagePriorsReactantExt 

using VLBIImagePriors
using Reactant 
using ComradeBase
using LinearAlgebra
using FFTW

using Reactant: AnyTracedRArray, RNumber

# function VLBIImagePriors.igmrf_1n(I::AnyTracedRArray, κ², ::ComradeBase.ReactantEx)
#     VLBIImagePriors.igmrf_1n(I, κ², KernelAbstractions.get_backend(I))
# end

# function VLBIImagePriors.igmrf_2n(I::AnyTracedRArray, κ², ::ComradeBase.ReactantEx)
#     VLBIImagePriors.igmrf_2n(I, κ², KernelAbstractions.get_backend(I))
# end


# This is currently required because Reactant has a performance issue with mapreduce
# on non-traced arrays
function LinearAlgebra.logdet(d::MarkovRandomFieldGraph{N}, ρ::RNumber) where {N}
    κ² = VLBIImagePriors.κ(ρ, Val(N))^2
    tmp = log.(κ² .+ d.λQ)
    a = sum(tmp)
    return N * a - length(d.λQ) * log(VLBIImagePriors.mrfnorm(d, κ²))
end

# Needs Reactant support for Ref during broadcasting
as(ps::VLBIImagePriors.AbstractPowerSpectrum, kx, ky) = VLBIImagePriors.ampspectrum(ps, (kx, ky))
function VLBIImagePriors._spectrum!(::ComradeBase.ReactantEx, ns::Reactant.AnyTracedRArray, ps::VLBIImagePriors.AbstractPowerSpectrum, kx, ky)
    ns .= as.(Ref(ps), kx, ky')
end

@inline @inbounds function VLBIImagePriors.igmrf_qv(I::Reactant.AnyTracedRMatrix, κ², ix, iy)
    value = (4 + κ²) * @allowscalar(I[ix, iy])
    @trace if ix < lastindex(I, 1)
        value -= @allowscalar(I[ix + 1, iy])
    end

    @trace if iy < lastindex(I, 2)
        value -= @allowscalar(I[ix, iy + 1])
    end

    @trace if ix > firstindex(I, 1)
        value -= @allowscalar(I[ix - 1, iy])
    end

    @trace  if iy > firstindex(I, 2)
        value -= @allowscalar(I[ix, iy - 1])
    end

    return value
end

function VLBIImagePriors.igmrf_1n(I::Reactant.AnyTracedRMatrix, κ², ::ComradeBase.ReactantEx)
    value = zero(eltype(I))
    @trace for iy in axes(I, 2)
        @trace for ix in axes(I, 1)
            value += VLBIImagePriors.igmrf_qv(I, κ², ix, iy) * @allowscalar(I[ix, iy])
        end
    end
    return value
end

function VLBIImagePriors.igmrf_2n(I::Reactant.AnyTracedRMatrix, κ², ::ComradeBase.ReactantEx)
    value = zero(eltype(I))
    @trace for iy in axes(I, 2)
        @trace for ix in axes(I, 1)
            value += VLBIImagePriors.igmrf_qv(I, κ², ix, iy)^2
        end
    end
    return value
end



end