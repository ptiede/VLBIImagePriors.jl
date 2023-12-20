var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = VLBIImagePriors","category":"page"},{"location":"#VLBIImagePriors","page":"Home","title":"VLBIImagePriors","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for VLBIImagePriors.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [VLBIImagePriors]","category":"page"},{"location":"#VLBIImagePriors.AdditiveLR","page":"Home","title":"VLBIImagePriors.AdditiveLR","text":"AdditiveLR <: LogRatioTransform\n\nDefines the additive log-ratio transform. The clr transformation moves from the simplex Sⁿ → R^n-1 and is given by\n\nalr(x) = [log(x₁/xₙ) ... log(xₙ/xₙ)],\n\nwhere g(x) = (∏xᵢ)ⁿ⁻¹ is the geometric mean. The inverse transformation is given by\n\nalr⁻¹(x) = exp.(x)./(1 + sum(x[1:n-1])).\n\n\n\n\n\n","category":"type"},{"location":"#VLBIImagePriors.AngleTransform","page":"Home","title":"VLBIImagePriors.AngleTransform","text":"AngleTransform\n\nA transformation that moves two vector x and y to an angle θ.  Note that is x and y are normally distributed then the resulting distribution in θ is uniform on the circle.\n\n\n\n\n\n","category":"type"},{"location":"#VLBIImagePriors.CenterImage-Tuple{AxisKeys.KeyedArray{T, N, <:Any, G} where {T, N, G<:ComradeBase.AbstractDims}}","page":"Home","title":"VLBIImagePriors.CenterImage","text":"CenterImage(X, Y)\nCenterImage(img::IntensityMap)\nCenterImage(grid::ComradeBase.AbstractDims)\n\nConstructs a projections operator that will take an arbritrary image and return a transformed version whose centroid is at the origin.\n\nArguments\n\nX, Y: the iterators for the image pixel locations\nimg: A IntensityMap whose grid is the same as the image you want to center\ngrid: The grid that the image is defined on.\n\nExample\n\njulia> using ComradeBase: centroid, imagepixels\njulia> grid = imagepixels(μas2rad(100.0), μas2rad(100.0), 256, 256)\njulia> K = CenterImage(grid)\njulia> img = IntensityMap(rand(256, 256), grid)\njulia> centroid(img)\n(1.34534e-10, 5.23423e-11)\njulia> cimg = K(img)\njulia> centroid(cimg)\n(1.231e-14, -2.43e-14)\n\nNote\n\nThis center image works using a linear projection operator. This means that is does not necessarily preserve the image total flux and the positive definiteness of the image. In practise we find that the deviation from the original flux, and the amount of negative flux is small.\n\nExplanation\n\nThe centroid is constructed by\n\nX ⋅ I = 0\nY ⋅ I = 0\n\nwhere I is the flattened image of length N, and X and Y are the image pixel locations. This can be simplified into the matrix equation\n\nXY ⋅ I = 0\n\nwhere XY is the 2×N matrix whose first for is given by X and second is Y. The space of centered images is then given by the null space of XY. Given this we can form a matrix C which is the kernel matrix of the nullspace whose columns are the orthogonal basis vectors of the null space of XY. Using this we can construct a centered image projection opertor by\n\nK = CC'.\n\nOur centered image is then given by I₀ = KI.\n\n\n\n\n\n","category":"method"},{"location":"#VLBIImagePriors.CenteredLR","page":"Home","title":"VLBIImagePriors.CenteredLR","text":"CenteredLR <: LogRatioTransform\n\nDefines the centered log-ratio transform. The clr transformation moves from the simplex Sⁿ → Rⁿ and is given by\n\nclr(x) = [log(x₁/g(x)) ... log(xₙ/g(x))]\n\nwhere g(x) = (∏xᵢ)ⁿ⁻¹ is the geometric mean. The inverse transformation is given by the softmax function and is only defined on a subset of the domain otherwise it is not injective\n\nclr⁻¹(x) = exp.(x)./sum(exp, x).\n\nNotes\n\nAs mentioned above this transformation is bijective on the entire codomain of the function. However, unlike the additive log-ratio transform it does not treat any pixel as being special.\n\n\n\n\n\n","category":"type"},{"location":"#VLBIImagePriors.CenteredRegularizer","page":"Home","title":"VLBIImagePriors.CenteredRegularizer","text":"CenteredRegularizer(x, y, σ, p)\n\nRegularizes a general image prior p such that the center of light is close the the origin of the imag. After regularization the log density of the prior is modified to\n\n    log p(I) to log p(I) - frac(x_C^2 + y_C^2)^22sigma^2 N_x N_y\n\nwhere N_x and N_y are the number of pixels in the x and y direction of the image, and x_C y_C are the center of light of the image I.\n\n\n\n\n\n","category":"type"},{"location":"#VLBIImagePriors.ConditionalMarkov-Tuple{Type{<:Union{Distributions.Exponential, Distributions.Normal, Distributions.TDist}}, Vararg{Any}}","page":"Home","title":"VLBIImagePriors.ConditionalMarkov","text":"ConditionalMarkov(D, args...)\n\nCreates a Conditional Markov measure, that behaves as a Julia functional. The functional returns a probability measure defined by the arguments passed to the functional.\n\nArguments\n\nD: The base distribution or measure of the random field. Currently Normal and TDist      are valid random fields\nargs: Additional arguments used to construct the Markov random field cache.         See MarkovRandomFieldCache for more information.\n\nExample\n\njulia> grid = imagepixels(10.0, 10.0, 64, 64)\njulia> ℓ = ConditionalMarkov(Normal, grid)\njulia> d = ℓ(16) # This is now a distribution\njulia> rand(d)\n\n\n\n\n\n","category":"method"},{"location":"#VLBIImagePriors.DiagonalVonMises","page":"Home","title":"VLBIImagePriors.DiagonalVonMises","text":"DiagonalVonMises(μ::Real, κ::Real)\nDiagonalVonMises(μ::AbstractVector{<:Real}, κ::AbstractVector{<:Real})\n\nConstructs a Von Mises distribution, with mean μ and concentraion parameter κ. If μ and κ are vectors then this constructs a independent multivariate Von Mises distribution.\n\nNotes\n\nThis is a custom implementation since the version in Distributions.jl has certain properties that do not play well (having an support only between [-π+μ, π+μ]) with usual VLBI problems. Additionally this distribution has a special overloaded product_distribution method so that concatenating multiple DiagonalVonMises together preserves the type. This is helpful for Zygote autodiff.\n\n\n\n\n\n","category":"type"},{"location":"#VLBIImagePriors.ExpMarkovRandomField","page":"Home","title":"VLBIImagePriors.ExpMarkovRandomField","text":"struct ExpMarkovRandomField{T<:Number, C} <: VLBIImagePriors.MarkovRandomField\n\nA image prior based off of the first-order zero mean unit variance Exponential Markov random field. This prior is similar to the combination of total variation TSV and L₁ norm and is given by\n\n√TV(I)² + inv(ρ²)L₁(I)²) + lognorm(ρ)\n\nwhere ρ is the correlation length the random field and lognorm(ρ) is the log-normalization of the random field. This normalization is needed to jointly infer I and the hyperparameters ρ.\n\nFields\n\nρ\nThe correlation length of the random field.\n\ncache\nThe Markov Random Field cache used to define the specific Markov random field class used.\n\nExample\n\njulia> ρ = 10.0\njulia> d = ExpMarkovRandomField(ρ, (32, 32))\njulia> cache = MarkovRandomFieldCache(Float64, (32, 32)) # now instead construct the cache\njulia> d2 = ExpMarkovRandomField(ρ, cache)\njulia> invcov(d) ≈ invcov(d2)\ntrue\n\n\n\n\n\n","category":"type"},{"location":"#VLBIImagePriors.ExpMarkovRandomField-Tuple{Number, AbstractMatrix}","page":"Home","title":"VLBIImagePriors.ExpMarkovRandomField","text":"ExpMarkovRandomField(ρ, img::AbstractArray)\n\nConstructs a first order zero-mean Exponential Markov random field with dimensions size(img), correlation ρ and unit covariance.\n\n\n\n\n\n","category":"method"},{"location":"#VLBIImagePriors.ExpMarkovRandomField-Tuple{Number, MarkovRandomFieldCache}","page":"Home","title":"VLBIImagePriors.ExpMarkovRandomField","text":"ExpMarkovRandomField(ρ, cache::MarkovRandomFieldCache)\n\nConstructs a first order zero-mean and unit variance Exponential Markov random field using the precomputed cache cache.\n\n\n\n\n\n","category":"method"},{"location":"#VLBIImagePriors.ExpMarkovRandomField-Tuple{Number, Tuple{Int64, Int64}}","page":"Home","title":"VLBIImagePriors.ExpMarkovRandomField","text":"ExpMarkovRandomField(ρ, dims)\n\nConstructs a first order zero-mean unit variance Exponential Markov random field with dimensions dims, correlation ρ.\n\n\n\n\n\n","category":"method"},{"location":"#VLBIImagePriors.GaussMarkovRandomField","page":"Home","title":"VLBIImagePriors.GaussMarkovRandomField","text":"struct GaussMarkovRandomField{T<:Number, C} <: VLBIImagePriors.MarkovRandomField\n\nA image prior based off of the first-order zero mean unit variance Gaussian Markov random field. This prior is similar to the combination of total squared variation TSV and L₂ norm, and is given by\n\nTSV(I) + (ρ²π²)L₂(I) + lognorm(ρ)\n\nwhere ρ is the correlation length the random field and lognorm(ρ) is the log-normalization of the random field. This normalization is needed to jointly infer I and the hyperparameters ρ.\n\nFields\n\nρ\nThe correlation length of the random field.\n\ncache\nThe Markov Random Field cache used to define the specific Markov random field class used.\n\nExample\n\njulia> ρ = 10.0\njulia> d = GaussMarkovRandomField(ρ, (32, 32))\njulia> cache = MarkovRandomFieldCache(Float64, (32, 32)) # now instead construct the cache\njulia> d2 = GaussMarkovRandomField(ρ, cache)\njulia> invcov(d) ≈ invcov(d2)\ntrue\n\n\n\n\n\n","category":"type"},{"location":"#VLBIImagePriors.GaussMarkovRandomField-Tuple{Number, AbstractMatrix}","page":"Home","title":"VLBIImagePriors.GaussMarkovRandomField","text":"GaussMarkovRandomField(ρ, img::AbstractArray)\n\nConstructs a first order zero-mean Gaussian Markov random field with dimensions size(img), correlation ρ and unit covariance.\n\n\n\n\n\n","category":"method"},{"location":"#VLBIImagePriors.GaussMarkovRandomField-Tuple{Number, MarkovRandomFieldCache}","page":"Home","title":"VLBIImagePriors.GaussMarkovRandomField","text":"GaussMarkovRandomField(ρ, cache::MarkovRandomFieldCache)\n\nConstructs a first order zero-mean and unit variance Gaussian Markov random field using the precomputed cache cache.\n\n\n\n\n\n","category":"method"},{"location":"#VLBIImagePriors.GaussMarkovRandomField-Tuple{Number, Tuple{Int64, Int64}}","page":"Home","title":"VLBIImagePriors.GaussMarkovRandomField","text":"GaussMarkovRandomField(ρ, dims)\n\nConstructs a first order zero-mean unit variance Gaussian Markov random field with dimensions dims, correlation ρ.\n\n\n\n\n\n","category":"method"},{"location":"#VLBIImagePriors.ImageDirichlet","page":"Home","title":"VLBIImagePriors.ImageDirichlet","text":"ImageDirichlet(α::AbstractMatrix)\nImageDirichlet(α::Real, ny, nx)\n\nA Dirichlet distribution defined on a matrix. Samples from this produce matrices whose elements sum to unity. This is a useful image prior when you want to separately constrain the flux. The  α parameter defines the usual Dirichlet concentration amount.\n\nNotes\n\nMuch of this code was taken from Distributions.jl and it's Dirichlet distribution. However, some changes were made to make it faster. Additionally, we use define a custom rrule to speed up derivatives.\n\n\n\n\n\n","category":"type"},{"location":"#VLBIImagePriors.ImageSimplex","page":"Home","title":"VLBIImagePriors.ImageSimplex","text":"ImageSimplex(ny,nx)\n\nThis defines a transformation from ℝⁿ⁻¹ to the n probability simplex defined on an matrix with dimension ny×nx. This is a more natural transformation for rasterized images, which are most naturally represented as a matrix.\n\nNotes\n\nMuch of this code was inspired by TransformVariables. However, we have specified custom rrules using Enzyme as a backend. This allowed the simplex transform to be used with Zygote and we achieved an order of magnitude speedup when computing the pullback of the simplex transform.\n\n\n\n\n\n","category":"type"},{"location":"#VLBIImagePriors.ImageSphericalUniform","page":"Home","title":"VLBIImagePriors.ImageSphericalUniform","text":"ImageSphericalUniform(nx, ny)\n\nConstruct a distribution where each image pixel is a 3-sphere uniform variable. This is useful for polarization where the stokes parameters are parameterized on the 3-sphere.\n\nCurrently we use a struct of vectors memory layout. That is the image is described by three matrices (X,Y,Z) grouped together as a tuple, where each matrix is one direction on the sphere, and we require norm((X,Y,Z)) == 1.\n\n\n\n\n\n","category":"type"},{"location":"#VLBIImagePriors.ImageUniform","page":"Home","title":"VLBIImagePriors.ImageUniform","text":"ImageUniform(a::Real, b::Real, nx, ny)\n\nA uniform distribution in image pixels where a/b are the lower/upper bound for the interval. This then concatenates ny×nx uniform distributions together.\n\n\n\n\n\n","category":"type"},{"location":"#VLBIImagePriors.MarkovRandomFieldCache","page":"Home","title":"VLBIImagePriors.MarkovRandomFieldCache","text":"struct MarkovRandomFieldCache{A, TD, M}\n\nA cache for a Markov random field.\n\nFields\n\nΛ\nIntrinsic Gaussian Random Field pseudo-precison matrix. This is similar to the TSV regularizer\n\nD\nGaussian Random Field diagonal precision matrix. This is similar to the L2-regularizer\n\nλQ\nThe eigenvalues of the Λ matrix which is needed to compute the log-normalization constant of the GMRF.\n\nExamples\n\njulia> mean = zeros(2,2) # define a zero mean\njulia> cache = GRMFCache(mean)\njulia> prior_map(x) = GaussMarkovRandomField(mean, x[1], x[2], cache)\njulia> d = HierarchicalPrior(prior_map, product_distribution([Uniform(-5.0,5.0), Uniform(0.0, 10.0)]))\njulia> x0 = rand(d)\n\n\n\n\n\n","category":"type"},{"location":"#VLBIImagePriors.MarkovRandomFieldCache-Tuple{ComradeBase.AbstractDims}","page":"Home","title":"VLBIImagePriors.MarkovRandomFieldCache","text":"MarkovRandomFieldCache(grid::AbstractDims)\n\nCreate a GMRF cache out of a Comrade model as the mean image.\n\nArguments\n\nm: Comrade model used for the mean image.\ngrid: The grid of points you want to create the model image on.\n\nKeyword arguments\n\ntransform = identity: A transform to apply to the image when creating the mean image. See the examples\n\nExample\n\njulia> m = MarkovRandomFieldCache(imagepixels(10.0, 10.0, 64, 64))\n\n\n\n\n\n","category":"method"},{"location":"#VLBIImagePriors.MarkovRandomFieldCache-Tuple{Type{<:Number}, Tuple{Int64, Int64}}","page":"Home","title":"VLBIImagePriors.MarkovRandomFieldCache","text":"MarkovRandomFieldCache(mean::AbstractMatrix)\n\nContructs the MarkovRandomFieldCache cache.\n\n\n\n\n\n","category":"method"},{"location":"#VLBIImagePriors.NamedDist-Union{Tuple{NamedTuple{N}}, Tuple{N}} where N","page":"Home","title":"VLBIImagePriors.NamedDist","text":"NamedDist(d::NamedTuple{N})\nNamedDist(;dists...)\n\nA Distribution with names N. This is useful to construct a set of random variables with a set of names attached to them.\n\njulia> d = NamedDist((a=Normal(), b = Uniform(), c = MvNormal(randn(2), rand(2))))\njulia> x = rand(d)\n(a = 0.13789342, b = 0.2347895, c = [2.023984392, -3.09023840923])\njulia> logpdf(d, x)\n\nNote that NamedDist values passed to NamedDist can also be abstract collections of distributions as well\n\njulia> d = NamedDist(a = Normal(),\n                     b = MvNormal(ones(2)),\n                     c = (Uniform(), InverseGamma())\n                     d = (a = Normal(), Beta)\n                    )\n\nHow this is done internally is considered an implementation detail and is not part of the public interface.\n\n\n\n\n\n","category":"method"},{"location":"#VLBIImagePriors.SphericalUnitVector","page":"Home","title":"VLBIImagePriors.SphericalUnitVector","text":"SphericalUnitVector{N}()\n\nA transformation from a set of N+1 vectors to the N sphere. The set of N+1 vectors are inherently assumed to be N+1 a distributed according to a unit multivariate Normal distribution.\n\nNotes\n\nFor more information about this transformation see the Stan manual. In the future this may be depricated when  is merged.\n\n\n\n\n\n","category":"type"},{"location":"#VLBIImagePriors.TDistMarkovRandomField","page":"Home","title":"VLBIImagePriors.TDistMarkovRandomField","text":"struct TDistMarkovRandomField{T<:Number, C} <: VLBIImagePriors.MarkovRandomField\n\nA image prior based off of the first-order Multivariate T distribution Markov random field.\n\nFields\n\nρ\nThe correlation length of the random field.\n\nν\nThe student T \"degrees of freedom parameter which ≥ 1 for a proper prior\n\ncache\nThe Markov Random Field cache used to define the specific Markov random field class used.\n\nExamples\n\njulia> ρ, ν = 16.0, 1.0\njulia> d = TDistMarkovRandomField(ρ, ν, (32, 32))\njulia> cache = MarkovRandomFieldCache(Float64, (32, 32)) # now instead construct the cache\njulia> d2 = TDistMarkovRandomField(ρ, ν, cache)\njulia> invcov(d) ≈ invcov(d2)\ntrue\n\n\n\n\n\n","category":"type"},{"location":"#VLBIImagePriors.TDistMarkovRandomField-Tuple{Number, Number, AbstractMatrix}","page":"Home","title":"VLBIImagePriors.TDistMarkovRandomField","text":"TDistMarkovRandomField(ρ, ν, img::AbstractArray)\n\nConstructs a first order TDist Markov random field with zero mean if it exists, correlation ρ and degrees of freedom ν.\n\n\n\n\n\n","category":"method"},{"location":"#VLBIImagePriors.TDistMarkovRandomField-Tuple{Number, Number, MarkovRandomFieldCache}","page":"Home","title":"VLBIImagePriors.TDistMarkovRandomField","text":"TDistMarkovRandomField(ρ, ν, cache::MarkovRandomFieldCache)\n\nConstructs a first order TDist Markov random field with zero mean ,correlation ρ, degrees of freedom ν, and the precomputed MarkovRandomFieldCache cache.\n\n\n\n\n\n","category":"method"},{"location":"#VLBIImagePriors.TDistMarkovRandomField-Tuple{Number, Number, Tuple{Int64, Int64}}","page":"Home","title":"VLBIImagePriors.TDistMarkovRandomField","text":"TDistMarkovRandomField(ρ, ν, dims)\n\nConstructs a first order TDist Markov random field with zero mean ,correlation ρ, degrees of freedom ν, with dimension dims.\n\n\n\n\n\n","category":"method"},{"location":"#VLBIImagePriors.WrappedUniform","page":"Home","title":"VLBIImagePriors.WrappedUniform","text":"WrappedUniform(period)\n\nConstructs a potentially multivariate uniform distribution that is wrapped a given period. That is\n\nd = WrappedUniform(period)\nlogpdf(d, x) ≈ logpdf(d, x+period)\n\nfor any x.\n\nIf period is a vector this creates a multivariate independent wrapped uniform distribution.\n\n\n\n\n\n","category":"type"},{"location":"#VLBIImagePriors.alr!-Tuple{Any, Any}","page":"Home","title":"VLBIImagePriors.alr!","text":"alr!(x, y)\n\nCompute the inverse alr transform. That is x lives in ℜⁿ and y, lives in Δⁿ\n\n\n\n\n\n","category":"method"},{"location":"#VLBIImagePriors.alrinv!-Tuple{Any, Any}","page":"Home","title":"VLBIImagePriors.alrinv!","text":"alrinv!(x, y)\n\nComputes the additive logit transform inplace. This converts from ℜⁿ → Δⁿ where Δⁿ is the n-simplex\n\nNotes\n\nThis function is mainly to transform the GaussMarkovRandomField to live on the simplex. In order to preserve the nice properties of the GRMF namely the log det we only use y[begin:end-1] elements and the last one is not included in the transform. This shouldn't be a problem since the additional parameter is just a dummy in that case and the Gaussian prior should ensure it is easy to sample from.\n\n\n\n\n\n","category":"method"},{"location":"#VLBIImagePriors.clr!-Tuple{Any, Any}","page":"Home","title":"VLBIImagePriors.clr!","text":"clr!(x, y)\n\nCompute the inverse alr transform. That is x lives in ℜⁿ and y, lives in Δⁿ\n\n\n\n\n\n","category":"method"},{"location":"#VLBIImagePriors.clrinv!-Tuple{Any, Any}","page":"Home","title":"VLBIImagePriors.clrinv!","text":"clrinv!(x, y)\n\nComputes the additive logit transform inplace. This converts from ℜⁿ → Δⁿ where Δⁿ is the n-simplex\n\nNotes\n\nThis function is mainly to transform the GaussMarkovRandomField to live on the simplex.\n\n\n\n\n\n","category":"method"},{"location":"#VLBIImagePriors.to_real-Tuple{VLBIImagePriors.LogRatioTransform, Any}","page":"Home","title":"VLBIImagePriors.to_real","text":"to_real(t::LogRatioTransform, y)\n\nTransform the value u that lives on the simplex to a value in the real vector space. See subtypes(LogRatioTransform) for a list of possible transformations.\n\nThe inverse of this transform is given by to_simplex(t, x).\n\nExample\n\n```julia julia> y = randn(100) julia> y .= y./sum(y) julia> toreal(CenteredLR(), y) julia> toreal(AdditiveLR(), y)\n\n\n\n\n\n","category":"method"},{"location":"#VLBIImagePriors.to_simplex!-Tuple{AdditiveLR, Any, Any}","page":"Home","title":"VLBIImagePriors.to_simplex!","text":"to_simplex!(t::LogRatioTransform, y, x)\n\nTransform the vector x assumed to be a real valued array to the simplex using the log-ratio transform t and stores the value in y.\n\nThe inverse of this transform is given by to_real!(t, x, y) where y is a vector that sums to unity, i.e. it lives on the simplex.\n\nExample\n\njulia> x = randn(100)\njulia> to_simplex(CenteredLR(), x)\njulia> to_simplex(AdditiveLR(), x)\n\n\n\n\n\n\n\n","category":"method"},{"location":"#VLBIImagePriors.to_simplex-Tuple{VLBIImagePriors.LogRatioTransform, Any}","page":"Home","title":"VLBIImagePriors.to_simplex","text":"to_simplex(t::LogRatioTransform, x)\n\nTransform the vector x assumed to be a real valued array to the simplex using the log-ratio transform t. See subtypes(LogRatioTransform) for a list of possible transformations.\n\nThe inverse of this transform is given by to_real(t, y) where y is a vector that sums to unity, i.e. it lives on the simplex.\n\nExample\n\njulia> x = randn(100)\njulia> to_simplex(CenteredLR(), x)\njulia> to_simplex(AdditiveLR(), x)\n\n\n\n\n\n\n\n","category":"method"}]
}
