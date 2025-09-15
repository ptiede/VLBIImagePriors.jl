# VLBIImagePriors

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ptiede.github.io/VLBIImagePriors.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ptiede.github.io/VLBIImagePriors.jl/dev)
[![Build Status](https://github.com/ptiede/VLBIImagePriors.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ptiede/VLBIImagePriors.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ptiede/VLBIImagePriors.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ptiede/VLBIImagePriors.jl)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)


## The Problem

VLBI imaging is a diffcult problem because the data corresponds to a largely incomplete description of the image due to the fact that an ideal VLBI interferometer measures the Fourier transform of an image and not the image itself.

To fix this problem additional information must be used to inform what a reasonable image looks like. In Bayesian modeling language this is essentially picking an image priors. This package has a number of different image priors that can be used with VLBI imaging software like [`Comrade.jl`](https://github.com/ptiede/Comrade.jl).

We also include a number of useful priors when doing polarized imaging, such as uniform priors on the n-sphere which are useful when parameterizing the [`Poincare sphere`](https://en.wikipedia.org/wiki/Polarization_(waves)#Poincar%C3%A9_sphere).
