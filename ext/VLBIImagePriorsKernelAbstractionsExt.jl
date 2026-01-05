module VLBIImagePriorsKernelAbstractionsExt

using VLBIImagePriors
using KernelAbstractions
using ComradeBase

@kernel function igmrf_kernel!(tmp, x, p)
    i,j = @index(Global, NTuple)
    value = (4 + p) * x[i,j]

    if i < lastindex(x, 1)
        value -= x[i + 1, j]
    end

    if j < lastindex(x, 2)
        value -= x[i, j + 1]
    end

    if i > firstindex(x, 1)
        value -= x[i - 1, j]
    end

    if j > firstindex(x, 2)
        value -= x[i, j - 1]
    end

    tmp[i,j] = value
end

function igmrf_ka(I::AbstractMatrix, p, order)
    bk = KernelAbstractions.get_backend(I)
    tmp = similar(I)
    kernel! = igmrf_kernel!(bk)
    kernel!(tmp, I, p; ndrange=size(I))
    return igmrf_ka_reduce(tmp, I, order)
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





end
