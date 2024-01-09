```@meta
CurrentModule = VLBIImagePriors
```

# VLBIImagePriors

This package implements a number of priors that are helpful when imaging VLBI data. These priors include commonly used Bayesian Stokes I imaging priors such as 
  - Log Uniform prior from [Broderick et al. 2020](https://iopscience.iop.org/article/10.3847/1538-4357/ab9c1f)
  - Dirichlet prior from [Pesce 2021](https://iopscience.iop.org/article/10.3847/1538-3881/abe3f8/pdf)
  - Gaussian Markov Random Field (see [Markov Random Field Priors](@ref))

For polarized imaging we also include a number of useful priors that parameterize the unit n-sphere which are required to parameterize the Poincaré sphere [Polarization Priors](@ref).

In addition we include a [`NamedDist`](@ref) which is a distribution composed of `NamedTuples`. This distribution also attempts to be rather smart and does certain conversions automatically, such at converting an array and/or tuple of distributions to a `Distributions.jl` object. 

Documentation for [VLBIImagePriors](https://github.com/ptiede/VLBIImagePriors.jl).

```@index
```

```@autodocs
Modules = [VLBIImagePriors]
```
