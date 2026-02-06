module VLBIImagePriorsKernelAbstractionsExt

using VLBIImagePriors
using KernelAbstractions
using ComradeBase

@kernel function igmrf_kernel!(tmp, x, p)
    i0, j0 = @index(Global, NTuple)
    i = i0 + 1
    j = j0 + 1
    value = (4 + p) * x[i, j]

    value -= x[i + 1, j]
    value -= x[i, j + 1]
    value -= x[i - 1, j]
    value -= x[i, j - 1]

    tmp[i, j] = value
end


function igmrf_ka(I::AbstractMatrix, p, order)
    bk = KernelAbstractions.get_backend(I)
    sz = size(I)
    # We split this into edges because Reactant raising
    # requires pure kernels so no conditional loads

    # The simplest thing to do is to allocate a slightly larger array
    # and fill the edges with zeros
    padI = similar(I, ntuple(n -> sz[1] + 2, 2))
    fill!(padI, 0)
    tmp = zero(padI)

    padI[2:(end - 1), 2:(end - 1)] .= I

    kernel! = igmrf_kernel!(bk)
    kernel!(tmp, padI, p; ndrange = size(I))

    return igmrf_ka_reduce(tmp, padI, order)
end

function igmrf_ka_reduce(tmp, I, ::Val{1})
    return mapreduce(splat(*), +, zip(tmp, I))
end

function igmrf_ka_reduce(tmp, I, ::Val{2})
    return sum(abs2, tmp)
end

function VLBIImagePriors.igmrf_2n(I::AbstractMatrix, κ², ::KernelAbstractions.Backend)
    return igmrf_ka(I, κ², Val(2))
end

function VLBIImagePriors.igmrf_1n(I::AbstractMatrix, κ², ::KernelAbstractions.Backend)
    return igmrf_ka(I, κ², Val(1))
end

function VLBIImagePriors.eigenvals(T, dims, bkend::KernelAbstractions.Backend)
    buf = allocate(bkend, T, dims)
    copyto!(buf, VLBIImagePriors.eigenvals(T, dims, Serial()))
    return buf
end

@kernel function spectrum_kernel!(ns, ps, kx, ky)
    i, j = @index(Global, NTuple)
    ns[i, j] = VLBIImagePriors.ampspectrum(ps, (kx[i], ky[j]))
end

function VLBIImagePriors._spectrum!(bk::KernelAbstractions.Backend, ns, ps::VLBIImagePriors.AbstractPowerSpectrum, kx, ky)
    bk = KernelAbstractions.get_backend(ns)
    kernel! = spectrum_kernel!(bk)
    return kernel!(ns, ps, kx, ky; ndrange = size(ns))
end


end
