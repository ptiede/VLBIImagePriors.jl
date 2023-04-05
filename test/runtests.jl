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
    d3 = ImageDirichlet(rand(10,10).+0.1)

    t1 = asflat(d1)
    t2 = asflat(d2)
    t3 = asflat(d3)

    @test t2 === t3

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

        x3 = rand(d3)

        y3 = inverse(t3, x3)
        ℓ3 = logdensityof(d3)
        function ℓpt3(x)
            y, lj = transform_and_logjac(t3, x)
            return ℓ3(y) + lj
        end

        ℓpt(y3)
        s = central_fdm(5,1)
        g1 = first(grad(s, ℓpt3, y3))
        g2 = first(Zygote.gradient(ℓpt3, y3))

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

        ℓ = logdensityof(d1)
        function ℓpt(x)
            y, lj = transform_and_logjac(t, x)
            return ℓ(y) + lj
        end

        s = central_fdm(5,1)
        g1 = first(grad(s, ℓpt, p0))
        g2 = first(Zygote.gradient(ℓpt, p0))
        @test first(g1) ≈ first(g2)

    end

    @testset "DiagonalVonMises" begin
        d0 = DiagonalVonMises(0.0, 0.5)
        d1 = DiagonalVonMises([0.5, 0.1], [inv(0.1), inv(π^2)])
        d2 = product_distribution(VonMises.(d1.μ, d1.κ))

        @test product_distribution([d0,d0]) isa DiagonalVonMises

        x = rand(d1)

        @test length(d1) == 2
        @test logdensityof(d1, x) ≈ logdensityof(d2, x)
        @test logdensityof(d1, x) ≈ logdensityof(d1, x .+ 2π)

        test_rrule(VLBIImagePriors._vonlogpdf, d1.μ, d1.κ, x)
        test_rrule(VLBIImagePriors._vonmisesnorm, d1.μ, d1.κ)

        t = asflat(d1)
        px = inverse(t, x)
        x2 = transform(t, px)

        @test sin.(x2) ≈ sin.(x)
        @test cos.(x2) ≈ cos.(x)

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

    @testset "WrappedUniform" begin
        periods = rand(5)
        d1 = WrappedUniform(periods)
        x = 2 .* rand(5)
        @test logdensityof(d1, x) ≈ logdensityof(d1, x .+ periods)

        xx = rand(d1)
        @test length(d1) == length(periods)

        d2 = product_distribution([d1, d1])
        @test d2 isa WrappedUniform
        @test length(d2) == length(periods)*2

        t = asflat(d1)
        px = inverse(t, xx)
        @test sin.(transform(t, px)) ≈ sin.(xx)
        @test cos.(transform(t, px)) ≈ cos.(xx)

        test_rrule(Distributions.logpdf, d1, xx, atol=1e-8)

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

    @testset "CenteredImage" begin
        d = ImageDirichlet(1.0, 4, 4)
        @test_throws AssertionError CenteredImage(1:2, 1:2, 1.0, d)
        x = y =  range(-2, 2, length=4)
        dc = CenteredImage(x, y, 1.0, d)

        @test Distributions.insupport(dc, rand(4,4)) == Distributions.insupport(d, rand(4,4))

        test_rrule(VLBIImagePriors.lcol, dc⊢NoTangent(), rand(4,4))

        xx = rand(d)
        @test logpdf(d, xx) + VLBIImagePriors.lcol(dc, xx) ≈ logdensityof(dc, xx)

        @test asflat(dc) === asflat(d)

    end

    @testset "GMRF" begin
        @testset "Tall" begin
            mimg = rand(10, 8)
            d1 = GaussMarkovRF(mimg, 1.0, 2.0)
            c = GMRFCache(mimg)
            d2 = GaussMarkovRF(mimg, 1.0, 2.0, c)

            x = rand(d1)
            @test logpdf(d1, x) ≈ logpdf(d2, x)
            Q = invcov(d1)
            b = Q*reshape(mimg, :)
            dd = MvNormalCanon(b, Array(Q))

            @test logpdf(d1, x) ≈ logpdf(dd, reshape(x, :))

            @test cov(d1) ≈ cov(dd)
            @test mean(d1) ≈ reshape(mean(dd), size(mimg))
        end

        @testset "Wide" begin
            mimg = rand(8, 10)
            d1 = GaussMarkovRF(mimg, 1.0, 2.0)
            c = GMRFCache(mimg)
            d2 = GaussMarkovRF(mimg, 1.0, 2.0, c)

            x = rand(d1)
            @test logpdf(d1, x) ≈ logpdf(d2, x)
            Q = invcov(d1)
            b = Q*reshape(mimg, :)
            dd = MvNormalCanon(b, Array(Q))

            @test logpdf(d1, x) ≈ logpdf(dd, reshape(x, :))

            @test cov(d1) ≈ cov(dd)
            @test mean(d1) ≈ reshape(mean(dd), size(mimg))
        end

        @testset "Equal" begin
            mimg = rand(8, 10)
            d1 = GaussMarkovRF(mimg, 1.0, 2.0)
            c = GMRFCache(mimg)
            d2 = GaussMarkovRF(mimg, 1.0, 2.0, c)

            x = rand(d1)
            @test logpdf(d1, x) ≈ logpdf(d2, x)
            Q = invcov(d1)
            b = Q*reshape(mimg, :)
            dd = MvNormalCanon(b, Array(Q))

            @test logpdf(d1, x) ≈ logpdf(dd, reshape(x, :))

            @test cov(d1) ≈ cov(dd)
            @test mean(d1) ≈ reshape(mean(dd), size(mimg))
        end


        @testset "rrules" begin
            test_rrule(VLBIImagePriors.igrmf_1n, rand(64,64))
        end


    end

    @testset "Hierarchical Prior" begin
        f(x) = Normal(x[1], x[2])
        dhyper = product_distribution([Normal(0.0, 1.0), LogNormal(-1.0, 1.0)])
        dHp = HierarchicalPrior(f, dhyper)
        x0 = rand(dHp)
        fd = f(x0.hyperparams)
        @test logpdf(dHp, x0) ≈ logpdf(fd, x0.params) + logpdf(dhyper, x0.hyperparams)

        x0s = rand(dHp, 1_00)
        x0s = rand(dHp, (10,10))

    end

    # @testset "SpecialRules" begin
    #     t = asflat(ImageUniform(10, 10))
    #     y = rand(TV.dimension(t))
    #     yoff = rand(TV.dimension(t) + 5)
    #     test_rrule(TV.transform_with, TV.LogJac()⊢NoTangent(), t⊢NoTangent(), y, 1)
    #     test_rrule(TV.transform_with, TV.LogJac()⊢NoTangent(), t⊢NoTangent(), yoff, 6)
    # end





end
