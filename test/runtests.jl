using VLBIImagePriors
using ChainRulesCore
using ChainRulesTestUtils
using Distributions
using FiniteDifferences
using Zygote
import TransformVariables as TV
using HypercubeTransform
using Test

@testset "VLBIImagePriors.jl" begin

    npix = 10
    d1 = Dirichlet(npix^2, 1.0)
    d2 = ImageDirichlet(1.0, npix, npix)

    t1 = asflat(d1)
    t2 = asflat(d2)

    ndim = dimension(t1)
    y0 = fill(0.1, ndim)

    x1,l1 = transform_and_logjac(t1, y0)
    x2,l2 = transform_and_logjac(t2, y0)

    @testset "ImageSimplex" begin
        @test x1 ≈ reshape(x2, :)
        @test l1 ≈ l2

        @test inverse(t2, x2) ≈ y0
        test_rrule(VLBIImagePriors.simplex_fwd,
                  TV.NoLogJac()⊢NoTangent(),
                  VLBIImagePriors.ImageSimplex(10,10)⊢NoTangent(),
                  randn(99))
        test_rrule(VLBIImagePriors.simplex_fwd,
                   TV.LogJac()⊢NoTangent(),
                   VLBIImagePriors.ImageSimplex(10,10)⊢NoTangent(),
                   randn(99))
    end

    @testset "ImageDirichlet" begin
        @test dimension(t1) == dimension(t2)
        @test length(d1) == prod(size(d2))

        @test logdensityof(d1, x1) ≈ logdensityof(d2, x2)
        ℓ2 = logdensityof(d2)

        function ℓpt(x)
            y, lj = transform_and_logjac(t2, x)
            return ℓ2(y) + lj
        end

        ℓpt(y0)
        ℓpt'(y0)
        s = central_fdm(5,1)
        g1 = first(grad(s, ℓpt, y0))
        g2 = first(Zygote.gradient(ℓpt, y0))

        @test g1 ≈ g2
    end

    @testset "ImageUniform" begin
        d1 = ImageUniform(0.0, 2.0, 2, 3)
        d2 = reshape(product_distribution([Uniform(0.0, 2.0) for _ in 1:(2*3)]), 2, 3)

        x0 = rand(d1)
        @test logdensityof(d1, x0) ≈ logdensityof(d2, x0)
        @test mean(d1) ≈ mean(d2)
        @test size(d1) == size(d2)

        t = asflat(d1)
        p0 = inverse(t, x0)

        @test transform(t, p0) ≈ x0
        test_rrule(Distributions._logpdf, d1⊢NoTangent(), x0, atol=1e-8)

    end

    @testset "DiagonalVonMises" begin
        d1 = DiagonalVonMises([0.5, 0.1], [0.5, 0.2])
        d2 = product_distribution(VonMises.(d1.μ, d1.κ))

        x = rand(d1)

        @test length(d1) == 2
        @test logdensityof(d1, x) ≈ logdensityof(d2, x)

        test_rrule(VLBIImagePriors._vonlogpdf, d1.μ, d1.κ, x)
        test_rrule(VLBIImagePriors._vonmisesnorm, d1.μ, d1.κ)

        t = asflat(d1)
        px = inverse(t, x)
        @test transform(t, px) ≈ x

        # test_rrule(TV.transform_with, TV.LogJac()⊢NoTangent(), t⊢NoTangent(), px, 1⊢NoTangent())
        function f(x)
            y, lj = transform_and_logjac(t, x)
            return logdensityof(d1, y) + lj
        end
        gz = Zygote.gradient(f, px)
        m = central_fdm(5, 1)
        gfd = FiniteDifferences.grad(m, f, px)
        @test first(gz) ≈ first(gfd)
    end

    @testset "ImageSphericalUniform" begin
        d = ImageSphericalUniform(2, 3)
        xx = rand(d)
        Distributions.rand!(d, xx)
        norms = map(hypot, xx...)
        @test norms ≈ fill(1.0, size(d))
        t = asflat(d)
        px = inverse(t, xx)
        @test prod(transform(t, px) .≈ xx)
        @test logdensityof(d, xx) ≈ -6*log(4π)

        function f(x)
            y, lj = transform_and_logjac(t, x)
            return logdensityof(d, y) + lj
        end
        gz = Zygote.gradient(f, px)
        m = central_fdm(5, 1)
        gfd = FiniteDifferences.grad(m, f, px)
        @test first(gz) ≈ first(gfd)

    end





end
