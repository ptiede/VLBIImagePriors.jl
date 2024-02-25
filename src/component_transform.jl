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
    # @info "HERE"
    y, ℓ, index2 = _transform_components(flag, tt, x, index)
    # @info typeof(out)
    return y, ℓ, index2
end

function _transform_components(flag::TV.LogJacFlag, tt::ComponentTransform, x, index)
    (;transformations, axes) = tt
    out = ComponentVector(zeros(lastindex(axes[1])), axes)
    ℓ = transform_components!(out, flag, transformations, x, index)
    return out, ℓ, index + TV.dimension(tt) + 1
end

Base.@constprop :aggressive getvalproperty(tt::NamedTuple, k) = getproperty(tt, k)


@generated function transform_components!(out::ComponentVector, flag::TV.LogJacFlag, transformation::NamedTuple{N}, x, index) where {N}
    exprs = []
    logjac = Expr(:call, :+)
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
        push!(logjac.args, :($ℓ_sym))
    end
    return quote
        $(exprs...)
        return $(logjac)
    end
end

function test(out, flag, transformation::Type{<:NamedTuple{N}}, x, index) where {N}
    exprs = []
    logjac = Expr(:call, :+)
    for k in N
        trf_sym = Symbol("trf_$k")
        y_sym = Symbol("y_$k")
        index_sym = Symbol("index_$k")
        ℓ_sym = Symbol("ℓ_$k")
        sym = QuoteNode(Symbol("$k"))
        push!(exprs, :($(trf_sym) = transformation.$k))
        push!(exprs, :(($(y_sym), $(ℓ_sym), $(index_sym)) = TV.transform_with(flag, $(trf_sym), x, index)))
        push!(exprs, :(index = $(index_sym)))
        push!(exprs, :(flexible_setproperty!(out, Symbol($sym), $(y_sym))))
        push!(logjac.args, :($ℓ_sym))
    end
    return quote
        $(exprs...)
        return $(logjac)
    end
end

function ChainRulesCore.rrule(::typeof(_transform_components), flag::TV.LogJacFlag, tt::ComponentTransform, x, index)
    y, ℓ, index = _transform_components(flag, tt, x, index)
    px = ProjectTo(x)
    function _transform_components_pullback(Δ)
        Δout = Δ[1]
        Δout .= unthunk(Δ[1])
        Δℓ = unthunk(Δ[2])

        Δf = NoTangent()
        Δflag = NoTangent()
        Δt = NoTangent()
        Δindex = NoTangent()
        yy = zero(y)
        Δy = zero(y)
        Δy .= Δout

        Δx = zero(x)
        ℓ2 = autodiff(Reverse, transform_components!, Duplicated(yy, Δy), Const(flag), Const(tt.transformations), Duplicated(x, Δx), Const(index))
        @info Δx
        return (Δf, Δflag, Δt, px(Δx), Δindex)
    end
    return (y, ℓ, index), _transform_components_pullback
end
