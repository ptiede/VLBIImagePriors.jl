module VLBIImagePriorsReactantExt 

using VLBIImagePriors
using Reactant 
using KernelAbstractions
using ComradeBase
using LinearAlgebra
using FFTW

using Reactant: RArray, RNumber

function VLBIImagePriors.igmrf_1n(I::RArray, κ², ::Serial)
    VLBIImagePriors.igmrf_1n(I, κ², KernelAbstractions.get_backend(I))
end

function VLBIImagePriors.igmrf_2n(I::RArray, κ², ::Serial)
    VLBIImagePriors.igmrf_2n(I, κ², KernelAbstractions.get_backend(I))
end


# This is currently required because Reactant has a performance issue with mapreduce
# on non-traced arrays
function LinearAlgebra.logdet(d::MarkovRandomFieldGraph{N}, ρ::RNumber) where {N}
    κ² = VLBIImagePriors.κ(ρ, Val(N))^2
    tmp = log.(κ² .+ d.λQ)
    a = sum(tmp)
    return N * a - length(d.λQ) * log(VLBIImagePriors.mrfnorm(d, κ²))
end

# TODO upsteam to Reactant (need to be able to compute fft in place and when using a plan)
function VLBIImagePriors.myfft!(p, x::RArray)
    y = fft(x)
    copyto!(x, y)
    return x
end



end