@testset "Reactant-friendly distributions" begin

    @testset "argument validation" begin
        @test_throws ArgumentError VLBIGaussian(0.0, -1.0)
        @test_throws ArgumentError VLBIGaussian(0.0, 0.0)
        @test_throws ArgumentError VLBIGaussian(0.0, [1.0, -1.0])
        @test_throws ArgumentError VLBIExponential(-1.0)
        @test_throws ArgumentError VLBIUniform(2.0, 1.0)
        @test_throws ArgumentError VLBIUniform(2.0, 2.0)
        @test_throws ArgumentError VLBIUniform([1.0, 2.0], [3.0, 1.5])
        @test_throws ArgumentError VLBIInverseGamma(-1.0, 2.0)
        @test_throws ArgumentError VLBIInverseGamma(2.0, -1.0)
        @test_throws ArgumentError VLBITDist(-1.0)
        @test_throws ArgumentError VLBITDist(5.0, 0.0, -1.0)

        # Public AffineDistribution constructor enforces shape match.
        @test_throws ArgumentError AffineDistribution(zeros(3, 3), 1.0, StdNormal((4, 4)))
        @test_throws ArgumentError AffineDistribution(0.0, ones(3, 3), StdNormal((4, 4)))
    end

    @testset "Distributions.params returns user-visible parameters" begin
        @test params(VLBIGaussian(0.3, 1.2)) == (0.3, 1.2)
        @test params(VLBIExponential(2.5)) == (2.5,)
        @test params(VLBIUniform(-1.0, 3.0)) == (-1.0, 3.0)
        @test params(VLBIInverseGamma(3.0, 2.0)) == (3.0, 2.0)
        @test params(VLBITDist(5.0, 0.3, 1.2)) == (5.0, 0.3, 1.2)
    end

    @testset "mean / var work for array AffineDistribution" begin
        d = VLBIGaussian(2.0, 1.5, (3, 4))
        @test mean(d) == fill(2.0, 3, 4)
        @test var(d) ≈ fill(1.5^2, 3, 4)

        μ = randn(2, 3)
        σ = abs.(randn(2, 3)) .+ 0.1
        dpe = VLBIGaussian(μ, σ)
        @test mean(dpe) ≈ μ
        @test var(dpe) ≈ σ .^ 2
    end

    @testset "Base.show for AffineDistribution" begin
        io = IOBuffer()
        show(io, VLBIGaussian(0.0, 1.0, (3, 4)))
        s = String(take!(io))
        @test occursin("StdNormal", s)
        @test occursin("size=(3, 4)", s)
    end


    @testset "unnormed_logpdf + lognorm == logpdf (and lognorm is data-independent)" begin
        scalar_cases = [
            VLBIGaussian(0.5, 1.3),
            VLBIExponential(2.0),
            VLBIUniform(-1.0, 3.0),
            VLBIInverseGamma(3.0, 2.0),
            VLBITDist(5.0),
            VLBITDist(5.0, 0.3, 1.2),
        ]
        for d in scalar_cases
            ln = lognorm(d)
            for x in [rand(d) for _ in 1:5]
                @test logpdf(d, x) ≈ unnormed_logpdf(d, x) + ln
            end
        end

        # Array forms — including the expensive per-element InverseGamma case
        array_cases = [
            VLBIGaussian(0.0, 1.0, (3, 4)),
            VLBIGaussian(randn(3, 4), abs.(randn(3, 4)) .+ 0.1),
            VLBIExponential(abs.(randn(3, 4)) .+ 0.1),
            VLBIUniform(-1.0, 1.0, (3, 4)),
            VLBIInverseGamma(2.0, 1.0, (3, 4)),
            VLBIInverseGamma(abs.(randn(3, 4)) .+ 1.5, abs.(randn(3, 4)) .+ 0.5),
            VLBITDist(5.0, 0.0, 1.0, (3, 4)),
            VLBITDist(abs.(randn(3, 4)) .+ 2.0, zeros(3, 4), ones(3, 4)),
            StdNormal((3, 4)),
            StdInverseGamma(abs.(randn(3, 4)) .+ 1.5),
            StdTDist(abs.(randn(3, 4)) .+ 2.0),
        ]
        for d in array_cases
            ln = lognorm(d)
            x = rand(d)
            @test logpdf(d, x) ≈ unnormed_logpdf(d, x) + ln

            # `lognorm` must not depend on `x`: changing x must not change ln.
            x2 = rand(d)
            @test lognorm(d) === ln || lognorm(d) ≈ ln
            @test logpdf(d, x2) ≈ unnormed_logpdf(d, x2) + ln
        end
    end

    @testset "scalar logpdf / mean / var matches Distributions.jl" begin
        cases = [
            (VLBIGaussian(0.3, 1.2), Normal(0.3, 1.2)),
            (VLBIExponential(2.5), Distributions.Exponential(2.5)),
            (VLBIUniform(-1.0, 3.0), Uniform(-1.0, 3.0)),
            (VLBIInverseGamma(3.0, 2.0), InverseGamma(3.0, 2.0)),
            (VLBITDist(5.0), TDist(5.0)),
        ]
        for (d, ref) in cases
            for x in [rand(ref) for _ in 1:50]
                @test logpdf(d, x) ≈ logpdf(ref, x) atol = 1.0e-10
            end
            @test mean(d) ≈ mean(ref)
            @test var(d) ≈ var(ref)
        end
    end

    @testset "scalar cdf / quantile match Distributions.jl" begin
        cases = [
            (VLBIGaussian(0.3, 1.2), Normal(0.3, 1.2)),
            (VLBIExponential(2.5), Distributions.Exponential(2.5)),
            (VLBIUniform(-1.0, 3.0), Uniform(-1.0, 3.0)),
            (VLBIInverseGamma(3.0, 2.0), InverseGamma(3.0, 2.0)),
            (VLBITDist(5.0), TDist(5.0)),
        ]
        ps = collect(0.05:0.1:0.95)
        for (d, ref) in cases
            for p in ps
                @test cdf(d, quantile(ref, p)) ≈ p atol = 1.0e-8
                @test quantile(d, p) ≈ quantile(ref, p) atol = 1.0e-8
            end
        end
    end

    @testset "scalar asflat round-trip" begin
        for d in (
                VLBIGaussian(0.0, 1.0), VLBIExponential(2.0), VLBIUniform(-1.0, 1.0),
                VLBIInverseGamma(2.0, 1.0), VLBITDist(5.0), VLBITDist(5.0, 0.3, 1.2),
            )
            t = asflat(d)
            x = rand(d)
            p = HypercubeTransform.inverse(t, x)
            @test HypercubeTransform.transform(t, p) ≈ x
        end
    end

    @testset "shared-params shape (VLBIGaussian(μ, σ, dims))" begin
        d = VLBIGaussian(0.5, 1.3, (3, 4))
        @test size(d) == (3, 4)
        x = randn(3, 4)
        @test logpdf(d, x) ≈ sum(logpdf.(Normal(0.5, 1.3), x))
        y = rand(d)
        @test size(y) == (3, 4)

        t = asflat(d)
        p = HypercubeTransform.inverse(t, y)
        @test HypercubeTransform.transform(t, p) ≈ y
    end

    @testset "per-element parameters (same family across grid)" begin
        μ = randn(3, 4)
        σ = abs.(randn(3, 4)) .+ 0.1
        d = VLBIGaussian(μ, σ)
        x = μ .+ σ .* randn(3, 4)
        @test logpdf(d, x) ≈ sum(logpdf.(Normal.(μ, σ), x))

        t = asflat(d)
        p = HypercubeTransform.inverse(t, x)
        @test HypercubeTransform.transform(t, p) ≈ x

        # InverseGamma per-element
        α = abs.(randn(2, 3)) .+ 1.5
        θ = abs.(randn(2, 3)) .+ 0.5
        ig = VLBIInverseGamma(α, θ)
        y = abs.(randn(2, 3)) .+ 0.1
        @test logpdf(ig, y) ≈ sum(logpdf.(InverseGamma.(α, θ), y))

        # Uniform per-element
        a = randn(2, 3)
        b = a .+ 1 .+ rand(2, 3)
        ud = VLBIUniform(a, b)
        xu = a .+ 0.5 .* (b .- a)
        @test logpdf(ud, xu) ≈ sum(logpdf.(Uniform.(a, b), xu))

        # Exponential per-element
        θe = abs.(randn(2, 3)) .+ 0.2
        ed = VLBIExponential(θe)
        ye = abs.(randn(2, 3)) .+ 0.1
        @test logpdf(ed, ye) ≈ sum(logpdf.(Distributions.Exponential.(θe), ye))

        # TDist per-element
        νg = abs.(randn(2, 3)) .+ 2.0
        td = VLBITDist(νg, zeros(2, 3), ones(2, 3))
        yt = randn(2, 3)
        @test logpdf(td, yt) ≈ sum(logpdf.(TDist.(νg), yt))
    end

    @testset "mixed scalar / array params" begin
        σ = abs.(randn(3, 3)) .+ 0.1
        d = VLBIGaussian(0.0, σ)
        x = σ .* randn(3, 3)
        @test logpdf(d, x) ≈ sum(logpdf.(Normal.(0.0, σ), x))

        μ = randn(3, 3)
        d2 = VLBIGaussian(μ, 1.5)
        x2 = μ .+ 1.5 .* randn(3, 3)
        @test logpdf(d2, x2) ≈ sum(logpdf.(Normal.(μ, 1.5), x2))
    end

    @testset "Std bases" begin
        for b in (
                StdNormal((3, 4)), StdExponential((3, 4)), StdUniform((3, 4)),
                StdInverseGamma(2.0, (3, 4)), StdTDist(5.0, (3, 4)),
            )
            z = rand(b)
            @test size(z) == (3, 4)
            @test isfinite(logpdf(b, z))
            t = asflat(b)
            p = HypercubeTransform.inverse(t, z)
            @test HypercubeTransform.transform(t, p) ≈ z
        end

        # scalar bases
        for b in (
                StdNormal{Float64, 0}(()), StdExponential(), StdUniform(),
                StdInverseGamma(2.0), StdTDist(5.0),
            )
            z = rand(b)
            @test isfinite(logpdf(b, z))
            t = asflat(b)
            @test t isa TV.AbstractTransform
        end
    end

    @testset "HierarchicalPrior with VLBIGaussian + VLBIExponential" begin
        h = HierarchicalPrior(ρ -> VLBIGaussian(0.0, ρ, (3, 4)), VLBIExponential(1.0))
        x = rand(h)
        @test x isa NamedTuple
        @test isfinite(logpdf(h, x))

        t = asflat(h)
        y = randn(TV.dimension(t))
        xback = HypercubeTransform.transform(t, y)
        @test keys(xback) == (:params, :hyperparams)
        @test size(xback.params) == (3, 4)
    end

    @testset "gradient cross-check (Enzyme vs FiniteDifferences) on per-element Gaussian" begin
        μ = randn(2, 3)
        σ = abs.(randn(2, 3)) .+ 0.2
        d = VLBIGaussian(μ, σ)
        x = μ .+ σ .* randn(2, 3)

        f = let d = d
            x -> logpdf(d, x)
        end

        s = central_fdm(5, 1)
        g_fd = first(FiniteDifferences.grad(s, f, x))
        g_en = Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(f), x)[1]
        @test isapprox(g_fd, g_en; atol = 1.0e-6)
    end

end
