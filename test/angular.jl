@testset "Angular Distributions" begin
    @testset "DiagonalVonMises" begin
        d0 = DiagonalVonMises(0.0, 0.5)

        t0 = asflat(d0)
        @test transform(t0, inverse(t0, π/4)) ≈ π/4

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

        d0 = WrappedUniform(2π)
        @test 0 ≤ rand(d0) ≤ 2π
        @test logpdf(d0, 0.0) == logpdf(d0, 1.0)
        @test asflat(d0) isa VLBIImagePriors.AngleTransform
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

end
