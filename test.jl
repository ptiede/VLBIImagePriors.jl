using TestEnv; TestEnv.activate()

ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

using KernelAbstractions, Reactant, CUDA

Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true


@kernel function igmrf_kernel!(tmp, x, p)
    i, j = @index(Global, NTuple)
    value = (4 + p) * x[i, j]

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

    tmp[i, j] = value
end

function igmrf_ka(I::AbstractMatrix, p)
    bk = KernelAbstractions.get_backend(I)
    tmp = similar(I)
    kernel! = igmrf_kernel!(bk)
    kernel!(tmp, I, p; ndrange = size(I))
    return tmp
end

xr = Reactant.to_rarray(ones(64, 64))
ar = ConcreteRNumber(10.0)
@compile raise = true igmrf_ka(xr, ar)
