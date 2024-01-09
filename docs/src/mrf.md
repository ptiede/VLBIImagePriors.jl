# Markov Random Field Priors

VLBI imaging is an ill-posed problem. Namely there is an infinite number of images that reproduce the observations, especially since the true image often has many more degrees of freedom than what we actually observe. To counteract this, we require the additional information in the form of a prior. For VLBI imaging we are limited to the resolution of the telescope which is set by the distance from the two furthest sites. Any additional information we gain about the image structure is then given by some additional assumption of belief we have about the source.

A common spatial modeling prior used in other fields such as geostatistics are *Markov Random Fields*. These consist of priors that encode correlations between neighbors using a Markovian structure. This Markovian dependency induces sparsity in the precision matrix of Gaussian distribution. This spartsity is key to making Markov Random fields scale to large amounts of data and is the reason things like Kalman filters scale linearly in the number of data points while standard Gaussian processes scale cubically. 

## 

## Relation to RML Regularizers

In *regularized maximum likelihood* (RML) imaging these additional assumptions are encoded into regularizers that enforce things like smoothness, sparsity, similarity to some fiducial image structure, etc. 


