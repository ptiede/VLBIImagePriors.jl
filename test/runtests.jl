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



end
