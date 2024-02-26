struct ComponentTransform{T, Ax} <: TV.VectorTransform
    transformations::T
    axes::Ax
    dimension::Int
    function ComponentTransform(transformations::T, ax
                            ) where {N, S <: TV.NTransforms, T <: NamedTuple{N, S}}
        new{T, typeof(ax)}(transformations, ax, TV._sum_dimensions(transformations))
    end
end

TV.dimension(tt::ComponentTransform) = tt.dimension

TV.as(ax::Axis, transformations) = ComponentTransform(transformations, ax)
TV.as(ax::Tuple{Axis}, transformations) = ComponentTransform(transformations, ax)

function TV.transform_with(flag::TV.LogJacFlag, tt::ComponentTransform, x::AbstractVector, index)
    data, index2 = _transform_components(flag, tt, x, index)
    out = ComponentVector(data[begin:end-1], tt.axes)
    ℓ = TV.logjac_zero(flag, eltype(out))
    if flag isa TV.LogJac
        ℓ = data[end]
    end
    return out, ℓ, index2
end

function _transform_components(flag::TV.LogJacFlag, tt::ComponentTransform, x, index)
    (;transformations, axes) = tt
    data = similar(x, lastindex(axes[1])+1)
    index2 = transform_components!(data, axes, flag, transformations, x, index)
    return data, index2
end

Base.@constprop :aggressive getvalproperty(tt::NamedTuple, k) = getproperty(tt, k)


@generated function transform_components!(data, axis, flag::TV.LogJacFlag, transformation::NamedTuple{N}, x, index) where {N}
    exprs = []
    push!(exprs, :(out = ComponentVector(@view(data[begin:end-1]), axis)))
    for k in N
        trf_sym = Symbol("trf_$k")
        y_sym = Symbol("y_$k")
        index_sym = Symbol("index_$k")
        ℓ_sym = Symbol("ℓ_$k")
        sym = QuoteNode(Symbol("$k"))
        push!(exprs, :($(trf_sym) = transformation.$k))
        push!(exprs, :(($(y_sym), $(ℓ_sym), $(index_sym)) = TV.transform_with(flag, $(trf_sym), x, index)))
        push!(exprs, :(index = $(index_sym)))
        push!(exprs, :(flexible_setproperty!(out, Val(Symbol($sym)), $(y_sym))))
        if flag isa TV.LogJac
            push!(exprs, :(data[end] += $(ℓ_sym)))
        end
    end
    return quote
        $(exprs...)
        return index
    end
end

function ChainRulesCore.rrule(::typeof(_transform_components), flag::TV.LogJacFlag, tt::ComponentTransform, x, index)
    data, index = _transform_components(flag, tt, x, index)
    px = ProjectTo(x)
    function _transform_components_pullback(Δ)
        Δdata = similar(data)
        Δdata .= unthunk(Δ[1])
        dd = zero(data)

        Δf = NoTangent()
        Δflag = NoTangent()
        Δt = NoTangent()
        Δindex = NoTangent()
        axes = tt.axes
        trfs = tt.transformations
        Δx = zero(x)
        autodiff(Reverse, transform_components!, Const, Duplicated(dd, Δdata), Const(axes), Const(flag), Const(trfs), Duplicated(x, Δx), Const(index))
        @info Δx
        return (Δf, Δflag, Δt, px(Δx), Δindex)
    end
    return (data, index), _transform_components_pullback
end
