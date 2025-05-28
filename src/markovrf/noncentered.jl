export standardize, centerdist, centerdist!

struct NonCenteredMarkovTransform{O, G<:MarkovRandomFieldGraph{O}, P}
    graph::G
    trf::P
end

function NonCenteredMarkovTransform(g::MarkovRandomFieldGraph; flag=FFTW.MEASURE)
    p = FFTW.plan_r2r!(copy(g.λQ), FFTW.RODFT00; flags=flag)
    return NonCenteredMarkovTransform(g, p)
end

function Serialization.serialize(s::Serialization.AbstractSerializer, cache::NonCenteredMarkovTransform)
    Serialization.writetag(s.io, Serialization.OBJECT_TAG)
    Serialization.serialize(s, typeof(cache))
    Serialization.serialize(s, cache.graph)
    Serialization.serialize(s, cache.trf.flags)
end

function Serialization.deserialize(s::AbstractSerializer, ::Type{<:NonCenteredMarkovTransform})
    graph = Serialization.deserialize(s)
    flag = Serialization.deserialize(s)
    return NonCenteredMarkovTransform(graph; flag=flag)
end


"""
    standardize(c::MarkovRandomField; flag=FFTW.MEASURE)

Transforms a `MarkovRandomField` into its non-centered form. This returns a tuple where the 
first element is a transform `t` that converts from the standardized space to the correlated 
Markov Random Field space, and the second element is the standard distribution, e.g., for a 
`GMRF` this is IID set of standard normals with the same size as the input Markov Random Field.

To transform the realization of the standarized Markov Random Field back to the original
space use the [`centerdist`](@ref) function.
"""
standardize(c::MarkovRandomField; flag=FFTW.MEASURE) = NonCenteredMarkovTransform(graph(c); flag), std_dist(c)


"""
    centerdist!(out, c::NonCenteredMarkovTransform, ρ, z::AbstractArray{<:Real}) where {Order, N}

The in-place version of [`centerdist`](@ref).
"""
function centerdist!(out::AbstractArray{<:Real}, c::NonCenteredMarkovTransform{Order}, ρ, z::AbstractArray{<:Real, N}) where {Order, N}
    κ² = κ(ρ, Val(Order))^2
    # numerator is the normaliztion of the MRF
    # denominator is to make the DST orthonormal
    sz = prod(ntuple(i -> size(c)[i] + 1, Val(N)))
    nm = sqrt(mrfnorm(κ², Val(Order))/(4*sz))
    g = graph(c)
    Λ = g.λQ

    if Order == 1
        out .= inv.(sqrt.(Λ .+ κ²)).*z.*nm
    elseif Order == 2
        out .= inv.(Λ .+ κ²).*z.*nm
    else
        out .=  ((Λ .+ κ²).^(-Order/2)).*z.*nm
    end
    
    c.trf*out
    return nothing
end

"""
    centerdist(c::NonCenteredMarkovTransform, ρ, z)
Transforms the standardized Markov Random Field `z` back to the original space
using the non-centered transform `c` and the correlation length `ρ`. The output is a new array
with the same size as `z` containing the transformed values.

Note this is substantially more efficient than the manual Cholesky decomposition and backsolve
approach, since we are able to use the fact that the MRF has a particularlly simple eigenspace 
decomposition.
"""
function centerdist(c::NonCenteredMarkovTransform{Order}, ρ, z::AbstractArray{<:Real}) where {Order}
    out = similar(z)
    centerdist!(out, c, ρ, z)
    return out
end

function invcenterdist(c::NonCenteredMarkovTransform, ρ, z::AbstractArray{<:Real})
    out = similar(z)
    invcenterdist!(out, c, ρ, z)
    return out
end

function invcenterdist!(out::AbstractArray{<:Real}, c::NonCenteredMarkovTransform{Order}, ρ, z::AbstractArray{<:Real}) where {Order}
    κ² = κ(ρ, Val(Order))^2
    # numerator is the normaliztion of the MRF
    # denominator is to make the DST orthonormal
    sz = prod(ntuple(i -> size(c)[i] + 1, Val(ndims(z))))
    nm = sqrt(mrfnorm(κ², Val(Order))*(4*sz))
    g = graph(c)
    Λ = g.λQ

    out .= z
    c.trf*out

    if Order == 1
        out .= sqrt.(Λ .+ κ²).*out./nm
    elseif Order == 2
        out .= (Λ .+ κ²).*out./nm
    else
        out .= ((Λ .+ κ²).^(Order/2)).*out./nm
    end
    
    return nothing
end

function Base.size(c::NonCenteredMarkovTransform{G, P}) where {G, P}
    return size(c.graph)
end

function graph(c::NonCenteredMarkovTransform{G, P}) where {G, P}
    return c.graph
end
