export ConditionalMarkov

struct ConditionalMarkov{B,C}
    cache::C
end

"""
    ConditionalMarkov(D, args...)

Creates a Conditional Markov measure, that behaves as a Julia functional. The functional
returns a probability measure defined by the arguments passed to the functional.

# Arguments

 - `D`: The <: `MarkovRandomField` that defines the underlying measure
 - `args`: Additional arguments used to construct the Markov random field cache.
           See [`MarkovRandomFieldGraph`](@ref) for more information.

# Example
```julia-repl
julia> grid = imagepixels(10.0, 10.0, 64, 64)
julia> ℓ = ConditionalMarkov(GaussMarkovRandomField, grid)
julia> d = ℓ(16) # This is now a distribution
julia> rand(d)
```
"""
function ConditionalMarkov(D::Type{<:MarkovRandomField}, args...; kwargs...)
    c = MarkovRandomFieldGraph(args...; kwargs...)
    return ConditionalMarkov{D, typeof(c)}(c)
end

function Base.show(io::IO, x::ConditionalMarkov{B}) where {B}
    println(io, "ConditionalMarkov:")
    println(io, "\tRandom Field: $(B)")
    println(io, "\tGraph: ", x.cache)
end
