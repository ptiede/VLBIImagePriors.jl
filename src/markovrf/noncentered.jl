export standardize, centerdist, centerdist!, noncenterdist, noncenterdist!

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
    standardize(c::MarkovRandomFieldGraph; flag=FFTW.MEASURE)

Transforms a `MarkovRandomFieldGraph` into its non-centered form. This returns the 
transform `t` that converts an array from the standardized Markov Random Field space, i.e.
white noise, to the correlated sample. This is essentially precaching the whitening transform.

To transform the realization of the standarized Markov Random Field back to the original
space use the [`centerdist`](@ref) function. To transform the realization of the original
Markov Random Field to the standardized space use the [`noncenterdist`](@ref) function.
"""
standardize(c::MarkovRandomFieldGraph; flag=FFTW.MEASURE) = NonCenteredMarkovTransform(graph(c); flag)

"""
    standardize(c::MarkovRandomFieldGraph; flag=FFTW.MEASURE)

Computes the transformation for the `MarkovRandomField` that converts it into its non-centered form. 
This returns the transform and the standardized distribution of the Markov Random Field. Note that
if the `MarkovRandomField` depends on a hyperparameter such as the correlation length `ρ`, the 
parameter should be passed into the centerdist function. 


To transform the realization of the standarized Markov Random Field back to the original
space use the [`centerdist`](@ref) function. To transform the realization of the original
Markov Random Field to the standardized space use the [`noncenterdist`](@ref) function.
"""
standardize(c::GaussMarkovRandomField; flag=FFTW.MEASURE) = standardize(graph(c); flag), std_dist(c)


"""
    centerdist!(out, c::NonCenteredMarkovTransform, ρ, z::AbstractArray{<:Real}) where {Order, N}

The in-place version of [`centerdist`](@ref).
"""
function centerdist!(out::AbstractArray{<:Real}, c::NonCenteredMarkovTransform{Order}, ρ, z::AbstractArray{<:Real, N}) where {Order, N}
    κ² = κ(ρ, Val(Order))^2
    # numerator is the normaliztion of the MRF
    # denominator is to make the DST orthonormal
    sz = prod(ntuple(i -> size(c)[i] + 1, Val(N)))
    g = graph(c)
    nm = sqrt(mrfnorm(g, κ²)/(4*sz))
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

"""
    noncenterdist(c::NonCenteredMarkovTransform, ρ, z)
Transforms the Markov Random Field `z` back uncorrelated standardized space, i.e. whitened noise.

Note this is substantially more efficient than the manual Cholesky decomposition
approach, since we are able to use the fact that the MRF has a particularlly simple eigenspace 
decomposition.
"""
function noncenterdist(c::NonCenteredMarkovTransform, ρ, z::AbstractArray{<:Real})
    out = similar(z)
    noncenterdist!(out, c, ρ, z)
    return out
end

"""
    noncenterdist!(out, c::NonCenteredMarkovTransform, ρ, z::AbstractArray{<:Real}) where {Order}
Transforms the Markov Random Field `z` back uncorrelated standardized space, i.e. whitened noise,
inplace into `out`.
"""
function noncenterdist!(out::AbstractArray{<:Real}, c::NonCenteredMarkovTransform{Order}, ρ, z::AbstractArray{<:Real}) where {Order}
    κ² = κ(ρ, Val(Order))^2
    # numerator is the normaliztion of the MRF
    # denominator is to make the DST orthonormal
    sz = prod(ntuple(i -> size(c)[i] + 1, Val(ndims(z))))
    g = graph(c)
    nm = sqrt(mrfnorm(g, κ²)*(4*sz))
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
