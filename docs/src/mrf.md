# Markov Random Field Priors

VLBI imaging is an ill-posed problem. Namely there is an infinite number of images that reproduce the observations, especially since the true image often has many more degrees of freedom than what we actually observe. To counteract this, we require the additional information in the form of a prior. For VLBI imaging we are limited to the resolution of the telescope which is set by the distance from the two furthest sites. Any additional information we gain about the image structure is then given by some additional assumption of belief we have about the source.

A common spatial modeling prior used in other fields such as geostatistics are *Markov Random Fields*. These consist of priors that encode correlations between neighbors using a Markovian structure. This Markovian dependency induces sparsity in the precision matrix of Gaussian distribution. This spartsity is key to making Markov Random fields scale to large amounts of data and is the reason things like Kalman filters scale linearly in the number of data points while standard Gaussian processes scale cubically. 

## 

## Relation to RML Regularizers

In *regularized maximum likelihood* (RML) imaging these additional assumptions are encoded into regularizers that enforce things like smoothness, sparsity, similarity to some fiducial image structure. The cost function for regularized imaging is

```math
    J_λ(I) = \sum_d χ_d²(I) + \sum_r \alpha_r R(I), 
```
where ``R`` are the regularizers and ``\alpha_r`` are the regularizer hyperparameters. The problem with RML imaging is that the values of the hyperparameters is often unknown. Therefore, to find their values people often use heuristics and surveys over different values to find what *looks good*. However, this can often lead to biases and imaging artifacts. 

However, probabilistic or statistical imaging provides a formalism simultaneously solve for the image and the regularizer or *prior* hyperparameters. For this we will assume that our prior `p` with hyperparameters ``\alpha`` are of the form

```math
    p(x | \alpha) = N(\alpha) f(x^T Q x),
```
where`Q` is the scale matrix that sets the variation and correlation scale of the problem, `f` is a function, and `N` is the normalization. We then seek a set of distributions for which the inner-product ``x^t Q x`` and `N` are easy to compute. To accomplish this we will look at Markov Random Fields and specifically the PDE

```math
    (\kappa^2 - \Delta)^n \varphi = \mathcal{W}(x)
```


