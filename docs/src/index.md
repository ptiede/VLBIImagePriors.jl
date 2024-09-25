```@meta
CurrentModule = VLBIImagePriors
```

# VLBIImagePriors

This package implements a number of priors that are helpful when imaging VLBI data. These priors include commonly used Bayesian Stokes I imaging priors such as 
  - Log Uniform prior from [Broderick et al. 2020](https://iopscience.iop.org/article/10.3847/1538-4357/ab9c1f)
  - Dirichlet prior from [Pesce 2021](https://iopscience.iop.org/article/10.3847/1538-3881/abe3f8/pdf)
  - Gaussian Markov Random Field (see [Markov Random Field Priors](@ref))

For polarized imaging we also include a number of useful priors that parameterize the unit n-sphere which are required to parameterize the Poincaré sphere [Polarization Priors](@ref).

As of v0.8 `NamedDist` has been moved to HypercubeTransforms.jl. If you depended on it please load that package instead.

Documentation for [VLBIImagePriors](https://github.com/ptiede/VLBIImagePriors.jl).

!!! warn
    As of 0.9 VLBIImagePriors requires you to load Enzyme explicitly for many AD operations since 
    rules will generically call into Enzyme as needed.

    
```@index
```

```@autodocs
Modules = [VLBIImagePriors]
```
