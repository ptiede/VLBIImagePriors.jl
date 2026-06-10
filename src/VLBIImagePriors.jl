module VLBIImagePriors

using Reexport

using ArgCheck
using Bessels
using ComradeBase
@reexport using DensityInterface
import Distributions as Dists
using DocStringExtensions
using FFTW
import FillArrays
using LinearAlgebra
using Random
using SpecialFunctions: loggamma, erf, erfinv
using SpecialFunctions
using StatsFuns
import TransformVariables as TV
using ReactantCore
using ComradeBase: rgetindex, rsetindex!

# ProbabilityTransports is now the home of the standardized distributions, the
# affine machinery, the truncated wrapper and the angular family. Re-export them
# so the VLBIImagePriors public API is unchanged, and use its transport interface
# (`transport_node`/`transport_to`) in place of HypercubeTransform.
@reexport using ProbabilityTransports
const PT = ProbabilityTransports
import ProbabilityTransports: transport_node, unnormed_logpdf, lognorm
using ProbabilityTransports: StdNormal, StdExponential, StdUniform, StdInverseGamma, StdTDist,
    TVFlat, AffineTransform, ScaleShift, PushforwardTransport, PushforwardDistribution,
    spherical_unit_vector, _std_cdf, _std_quantile

# Backwards-compatible alias for the truncated wrapper.
const VLBITruncated = ProbabilityTransports.Truncated
export VLBITruncated

using ComradeBase: @threaded


include("imagesimplex.jl")
include("dirichlet.jl")
include("special_rules.jl")
include("distributions/distributions.jl")
include("uniform.jl")
include("markovrf/markovrf.jl")
include("srf.jl")
include("heirarchical.jl")
include("logratio_transform.jl")
include("hypercube_compat.jl")


end
