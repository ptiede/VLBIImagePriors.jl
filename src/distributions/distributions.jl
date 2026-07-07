# VLBI's array/matrix-scale `AffineDistribution` (image-prior machinery). The
# standardized base distributions (`StdNormal`, …), the truncated wrapper and the
# `unnormed_logpdf`/`lognorm` split now live in ProbabilityTransports and are
# imported in `VLBIImagePriors.jl`. This file keeps only the affine wrapper and
# the user-facing `VLBI*` constructors.

export VLBIGaussian, VLBIExponential, VLBIUniform, VLBIInverseGamma, VLBITDist
export AffineDistribution
export unnormed_logpdf, lognorm

include("affine.jl")
include("constructors.jl")
include("beta.jl")
