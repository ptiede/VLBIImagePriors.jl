using Pkg; Pkg.activate(@__DIR__)
using VLBIImagePriors
using AdvancedHMC
using Zygote
import VLBIImagePriors as RP
import TransformVariables as TV



function make_ℓ(d)
    t = RP.ImageSimplex(length(d.y), length(d.x))
    dIm = ImageDirichlet(1.0, length(d.y), length(d.x))
    ℓ = Base.Fix1(lpdf, d)
    let t=t, dIm=dIm, ℓ=ℓ, d=d
        y->begin
        Im, lj = TV.transform_and_logjac(t, y)
        logdensityof(dIm, Im) + lj + lpdf(d, Im)
    end
    end
end

nx = 20
ny = 20
x = range(-10.0, 10.0, length=nx)
y = range(-10.0, 10.0, length=ny)
d = CenteredImage(x,y,0.05, ImageDirichlet(1.0, ny, nx))


d = ImageDirichlet(1.0, ny, nx)
ℓ =  logdensityof(d)


t = VLBIImagePriors.ImageSimplex(ny, nx)

ℓt(z) = ℓ(TV.transform(t, z))

z = rand(TV.dimension(t))

@time ℓt'(z)

import ReverseDiff as RD
ftape = RD.GradientTape(ℓt, z)
cftape = RD.compile(ftape)

out = similar(z)
@time RD.gradient!(out, cftape, z)

metric = DiagEuclideanMetric(length(z))
hamiltonian = Hamiltonian(metric, ℓ, Zygote)

# Define a leapfrog solver, with initial step size chosen heuristically
initial_ϵ = find_good_stepsize(hamiltonian, z)
integrator = Leapfrog(initial_ϵ)

# Define an HMC sampler, with the following components
#   - multinomial sampling scheme,
#   - generalised No-U-Turn criteria, and
#   - windowed adaption for step-size and diagonal mass matrix
proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

samples, stats = sample(hamiltonian, proposal, z, 2000, adaptor, 1000; progress=true)

using Plots
chain = TV.transform.(Ref(RP.ImageSimplex(20,20)), samples)
heatmap(chain[50])
