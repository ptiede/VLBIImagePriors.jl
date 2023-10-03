using VLBIImagePriors
using ChainRulesCore
using ChainRulesTestUtils
using Distributions
using FiniteDifferences
using Zygote
import TransformVariables as TV
using HypercubeTransform
using Test
using ComradeBase

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
        x = 2. * rand(5)
        @test logdensityof(d1, x) ≈ logdensityof(d1, x .+ periods)
        du = WrappedUniform(2π, 5)
        xx = rand(d1)
        @test length(d1) == length(periods)
        @test length(rand(d1)) == length(rand(du))
        d2 = product_distribution([d1, d1])
        @test d2 isa WrappedUniform
        @test length(d2) == length(periods)*2

        t = asflat(d1)
        px = inverse(t, xx)
        @test sin.(transform(t, px)) ≈ sin.(xx)
        @test cos.(transform(t, px)) ≈ cos.(xx)

        test_rrule(Distributions.logpdf, d1, xx, atol=1e-8)

    end

    @testset "SphericalUniform" begin
        t = SphericalUnitVector{3}()
        @inferred TV.transform(t, randn(dimension(t)))
        f = let t = t
            x->sum(abs2, TV.transform(t, x))
        end
        px = randn(dimension(t))
        gz = Zygote.gradient(f, px)
        m = central_fdm(5, 1)
        gfd = FiniteDifferences.grad(m, f, px)
        @test isapprox(first(gz), first(gfd), atol=1e-6)
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
        @test isapprox(first(gz), first(gfd), atol=1e-6)
    end

    @testset "CenteredRegularizer" begin
        d = ImageDirichlet(1.0, 4, 4)
        @test_throws AssertionError CenteredRegularizer(1:2, 1:2, 1.0, d)
        x = y =  range(-2, 2, length=4)
        dc = CenteredRegularizer(x, y, 1.0, d)

        @test Distributions.insupport(dc, rand(4,4)) == Distributions.insupport(d, rand(4,4))

        test_rrule(VLBIImagePriors.lcol, dc⊢NoTangent(), rand(4,4))

        xx = rand(d)
        @test logpdf(d, xx) + VLBIImagePriors.lcol(dc, xx) ≈ logdensityof(dc, xx)

        @test asflat(dc) === asflat(d)

    end
    @testset "CenterImage" begin
        grid = imagepixels(10.0, 10.0, 48, 48)
        K = CenterImage(grid)
        img0 = IntensityMap(rand(48, 48), grid)
        K2 = CenterImage(img0)
        cimg = K(img0)
        @test K(img0) ≈ K2(img0)
        c0 = centroid(IntensityMap(cimg, grid))
        @test isapprox(c0[1], 0.0, atol=1e-6)
        @test isapprox(c0[2], 0.0, atol=1e-6)
        test_rrule(VLBIImagePriors.center_kernel, K.kernel⊢NoTangent(), rand(48, 48))
    end

    @testset "GMRF" begin
        function moment_test(d, nsamples=100_000, atol=5e-2)
            c = cov(d)
            s = reduce(hcat, reshape.(rand(d, nsamples), :))
            cs = cov(s; dims=2)
            ms = reshape(mean(s; dims=2), size(d))
            @test isapprox(c, cs; atol)
            @test isapprox(mean(d), ms; atol)
        end

        @testset "Tall" begin
            mimg = rand(10, 8)
            d1 = GaussMarkovRandomField(mimg, 3.0, 0.2)
            c = MarkovRandomFieldCache(mimg)
            d2 = GaussMarkovRandomField(mimg, 3.0, 0.2, c)

            moment_test(d1)

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
            d1 = GaussMarkovRandomField(mimg, 3.0, 0.2)
            c = MarkovRandomFieldCache(mimg)
            d2 = GaussMarkovRandomField(mimg, 3.0, 0.2, c)

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
            mimg = rand(10, 10)
            d1 = GaussMarkovRandomField(mimg, 3.0, 0.2)
            c = MarkovRandomFieldCache(mimg)
            d2 = GaussMarkovRandomField(mimg, 3.0, 0.2, c)
            trf, d = standardize(c, Normal)

            p = trf(rand(d), mimg, 1.0, 0.1, 0.0)
            trf(rand(d), mimg, 1.0, 0.1, 1.0)
            dimg = mean(map(_->trf(rand(d), 1.0, 1.0, 0.0), 1:1_000_000))
            isapprox(dimg, mimg, atol=1e-2)
            @test size(p) == size(rand(d2))
            logdensityof(d, rand(d))

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

    @testset "TDistMRF" begin
        @testset "Tall" begin
            mimg = rand(10, 8)
            d1 = TDistMarkovRandomField(mimg, 3.0, 2.0, 1.0)
            c = MarkovRandomFieldCache(mimg)
            d2 = TDistMarkovRandomField(mimg, 3.0, 2.0, 1.0, c)

            x = rand(d1)
            @test logpdf(d1, x) ≈ logpdf(d2, x)
            Q = invcov(d1)
        end

        @testset "Wide" begin
            mimg = rand(8, 10)
            d1 = TDistMarkovRandomField(mimg, 3.0, 2.0, 5.0)
            c = MarkovRandomFieldCache(mimg)
            d2 = TDistMarkovRandomField(mimg, 3.0, 2.0, 5.0, c)

            x = rand(d1)
            @test logpdf(d1, x) ≈ logpdf(d2, x)
            Q = invcov(d1)
        end

        @testset "Equal" begin
            mimg = rand(10, 10)
            d1 = TDistMarkovRandomField(mimg, 3.0, 2.0, 100.0)
            c = MarkovRandomFieldCache(mimg)
            d2 = TDistMarkovRandomField(mimg, 3.0, 2.0, 100.0, c)

            x = rand(d1)
            @test logpdf(d1, x) ≈ logpdf(d2, x)
        end

    end



    @testset "Hierarchical Prior" begin
        f(x) = Normal(x[1], exp(x[2]))
        dhyper = product_distribution([Normal(0.0, 1.0), Normal(-1.0, 1.0)])
        dHp = HierarchicalPrior(f, dhyper)
        x0 = rand(dHp)
        fd = f(x0.hyperparams)
        @test logpdf(dHp, x0) ≈ logpdf(fd, x0.params) + logpdf(dhyper, x0.hyperparams)

        x0s = rand(dHp, 1_00)
        x0s = rand(dHp, (10,10))
        asflat(dHp)
    end

    @testset "NamedDist" begin
        d1 = NamedDist((a=Normal(), b = Uniform(), c = MvNormal(ones(2))))
        @test propertynames(d1) == (:a, :b, :c)
        @test d1.a == Normal()
        x1 = rand(d1)
        @test rand(d1, 2) isa Vector{<:NamedTuple}
        @test size(rand(d1, 2)) == (2,)
        rand(d1, 20, 21)
        @test logpdf(d1, x1) ≈ logpdf(d1.a, x1.a) + logpdf(d1.b, x1.b) + logpdf(d1.c, x1.c)

        dists = getfield(d1, :dists)
        xt = (b = 0.5, a = 1.0, c = [-0.5, 0.6])
        @test logpdf(d1, xt) ≈ logpdf(d1.a, xt.a) + logpdf(d1.b, xt.b) + logpdf(d1.c, xt.c)

        d2 = NamedDist(a=(Uniform(), Normal()), b = Beta(), c = [Uniform(), Uniform()], d = (a=Normal(), b = ImageUniform(2, 2)))
        @inferred logdensityof(d2, rand(d2))
        p0 = (a=(0.5, 0.5), b = 0.5, c = [0.25, 0.75], d = (a = 0.1, b = fill(0.1, 2, 2)))
        @test typeof(p0) == typeof(rand(d2))
        tf = asflat(d2)
        # tc = ascube(d2)
        @inferred TV.transform(tf, randn(dimension(tf)))
        # @inferred TV.transform(tc, rand(dimension(tc)))

    end

    @testset "Log Ratio Transform" begin
        x = randn(10,10)
        ycl = to_simplex(CenteredLR(), x)
        yal = to_simplex(AdditiveLR(), x)

        @test length(ycl) == length(x)
        @test length(yal) == length(x)
        @test sum(ycl) ≈ 1
        @test sum(yal) ≈ 1

        test_rrule(to_simplex, AdditiveLR(), x)
        test_rrule(to_simplex, CenteredLR(), x)

        y = rand(10, 10) .+ 0.5
        far(x) = sum(abs2, to_real(AdditiveLR(), x/sum(x)))
        s = central_fdm(5,1)
        gf_ar = first(grad(s, far, y))
        gz_ar = first(Zygote.gradient(far, y))
        @test isapprox(first(gf_ar), first(gz_ar), atol=1e-6)

        fcr(x) = sum(abs2, to_real(CenteredLR(), x/sum(x)))
        gf_cr = first(grad(s, fcr, y))
        gz_cr = first(Zygote.gradient(fcr, y))
        @test isapprox(first(gf_cr), first(gz_cr), atol=1e-6)


        # test_rrule(to_real, AdditiveLR(), yal)
        # test_rrule(to_real, CenteredLR(), ycl)

        x0al = to_real(AdditiveLR(), yal)
        x0cl = to_real(CenteredLR(), ycl)
        @test x0al[1:end-1] ≈ x[1:end-1]
        @test x0cl .- x0cl[1] ≈ x .- x[1]
    end

    # @testset "SpecialRules" begin
    #     t = asflat(ImageUniform(10, 10))
    #     y = rand(TV.dimension(t))
    #     yoff = rand(TV.dimension(t) + 5)
    #     test_rrule(TV.transform_with, TV.LogJac()⊢NoTangent(), t⊢NoTangent(), y, 1)
    #     test_rrule(TV.transform_with, TV.LogJac()⊢NoTangent(), t⊢NoTangent(), yoff, 6)
    # end





end
