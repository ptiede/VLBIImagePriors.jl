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

end