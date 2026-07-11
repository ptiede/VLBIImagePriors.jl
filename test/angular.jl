@testset "Angular Distributions" begin
    @testset "DiagonalVonMises" begin
        d0 = DiagonalVonMises(0.0, 0.5)

        t0 = transport_to(d0, TVFlat())
        @test latent_pfwd(t0, latent_pback(t0, π / 4)) ≈ π / 4

        x = rand(d0)
        @test x isa Float64


        d1 = DiagonalVonMises([0.5, 0.1], [inv(0.1), inv(π^2)])
        d2 = product_distribution(VonMises.(d1.μ, d1.κ))

        ds = DiagonalVonMises(0.5, 0.1)
        d32 = DiagonalVonMises(0.5f0, 0.1f0)
        @test logpdf(d32, 3.0f-1) isa Float32

        dv = DiagonalVonMises([0.5], [0.1])
        x = rand(ds)
        @test logpdf(ds, x) ≈ logpdf(dv, [x])
        @test insupport(ds, x)

        @test product_distribution([d0, d0]) isa DiagonalVonMises

        x = rand(d1)
        @test x isa Vector{Float64}

        @test length(d1) == 2
        @test logdensityof(d1, x) ≈ logdensityof(d2, x)
        @test logdensityof(d1, x) ≈ logdensityof(d1, x .+ 2π)

        t = transport_to(d1, TVFlat())
        px = latent_pback(t, x)
        x2 = latent_pfwd(t, px)

        @test sin.(x2) ≈ sin.(x)
        @test cos.(x2) ≈ cos.(x)

        function f(x)
            _, ld = latent_pfwd_and_logdensity(t, x)
            return ld
        end
        @test isapprox(enzyme_grad(f, px), fdm_grad(f, px); atol = 1.0e-6)
    end

    @testset "WrappedUniform" begin
        periods = rand(5)
        d1 = WrappedUniform(periods)
        x = 2.0 * rand(5)
        @test logdensityof(d1, x) ≈ logdensityof(d1, x .+ periods)
        du = WrappedUniform(2π, 5)
        xx = rand(d1)
        @test length(d1) == length(periods)
        @test length(rand(d1)) == length(rand(du))
        d2 = product_distribution([d1, d1])
        @test d2 isa WrappedUniform
        @test length(d2) == length(periods) * 2

        t = transport_to(d1, TVFlat())
        px = latent_pback(t, xx)
        @test sin.(latent_pfwd(t, px)) ≈ sin.(xx)
        @test cos.(latent_pfwd(t, px)) ≈ cos.(xx)

        d0 = WrappedUniform(2π)
        @test 0 ≤ rand(d0) ≤ 2π
        @test logpdf(d0, 0.0) == logpdf(d0, 1.0)
        @test transport_node(d0, TVFlat()) === angle_transform()
        @test insupport(d0, 0.0)
    end

    @testset "SphericalUniform" begin
        t = spherical_unit_vector(3)
        @inferred TV.transform(t, randn(dimension(t)))
        f = let t = t
            x -> sum(abs2, TV.transform(t, x))
        end
        px = randn(dimension(t))
        @test isapprox(enzyme_grad(f, px), fdm_grad(f, px); atol = 1.0e-6)
    end

end
