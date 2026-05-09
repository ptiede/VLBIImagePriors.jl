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

    @testset "constructor variants with dims::Int..." begin
        # Each VLBI* family supports dims as either a tuple or splatted ints.
        for (d_int, d_tup) in (
                (VLBIGaussian(0.0, 1.0, 3, 4), VLBIGaussian(0.0, 1.0, (3, 4))),
                (VLBIExponential(2.0, 3, 4), VLBIExponential(2.0, (3, 4))),
                (VLBIUniform(-1.0, 1.0, 3, 4), VLBIUniform(-1.0, 1.0, (3, 4))),
                (VLBIInverseGamma(2.0, 1.0, 3, 4), VLBIInverseGamma(2.0, 1.0, (3, 4))),
                (VLBITDist(5.0, 0.0, 1.0, 3, 4), VLBITDist(5.0, 0.0, 1.0, (3, 4))),
            )
            @test size(d_int) == size(d_tup) == (3, 4)
        end

        @test size(StdNormal(3, 4)) == (3, 4)
        @test size(StdExponential(3, 4)) == (3, 4)
        @test size(StdUniform(3, 4)) == (3, 4)
        @test size(StdInverseGamma(2.0, 3, 4)) == (3, 4)
        @test size(StdTDist(5.0, 3, 4)) == (3, 4)
    end

    @testset "mixed scalar/array constructors for InverseGamma and TDist" begin
        αg = abs.(randn(2, 3)) .+ 1.5
        θg = abs.(randn(2, 3)) .+ 0.5

        # InverseGamma: scalar-α + array-θ
        d1 = VLBIInverseGamma(2.0, θg)
        @test size(d1) == size(θg)
        x1 = abs.(randn(2, 3)) .+ 0.1
        @test logpdf(d1, x1) ≈ sum(logpdf.(InverseGamma.(2.0, θg), x1))

        # InverseGamma: array-α + scalar-θ
        d2 = VLBIInverseGamma(αg, 1.5)
        @test size(d2) == size(αg)
        @test logpdf(d2, x1) ≈ sum(logpdf.(InverseGamma.(αg, 1.5), x1))

        # TDist: array-ν + scalar-μ + scalar-σ
        νg = abs.(randn(2, 3)) .+ 3.0
        d3 = VLBITDist(νg, 0.0, 1.0)
        @test size(d3) == size(νg)
        y3 = randn(2, 3)
        @test logpdf(d3, y3) ≈ sum(logpdf.(TDist.(νg), y3))

        # TDist: scalar-ν + array-μ + array-σ
        μg = randn(2, 3)
        σg = abs.(randn(2, 3)) .+ 0.2
        d4 = VLBITDist(5.0, μg, σg)
        @test size(d4) == size(μg)
        y4 = μg .+ σg .* randn(2, 3)
        # Compare against per-element TDist scaled/shifted by hand
        ref4 = sum(@. logpdf(TDist(5.0), (y4 - μg) / σg) - log(σg))
        @test logpdf(d4, y4) ≈ ref4
    end

    @testset "mixed scalar/array constructors for Uniform" begin
        b_arr = [1.0, 2.0, 3.0]
        d1 = VLBIUniform(0.0, b_arr)
        @test size(d1) == (3,)
        x1 = [0.5, 1.0, 2.5]
        @test logpdf(d1, x1) ≈ sum(logpdf.(Uniform.(0.0, b_arr), x1))

        a_arr = [0.0, -1.0, -2.0]
        d2 = VLBIUniform(a_arr, 5.0)
        @test size(d2) == (3,)
        x2 = [1.0, 0.0, -1.0]
        @test logpdf(d2, x2) ≈ sum(logpdf.(Uniform.(a_arr, 5.0), x2))
    end

    @testset "more argument validation" begin
        @test_throws ArgumentError VLBIGaussian([0.0, 0.0], [1.0, 1.0, 1.0])
        @test_throws ArgumentError VLBIInverseGamma([1.0, 2.0], [1.0, 2.0, 3.0])
        @test_throws ArgumentError VLBIInverseGamma([-1.0, 2.0], [1.0, 2.0])
        @test_throws ArgumentError VLBIInverseGamma([1.0, 2.0], [-1.0, 2.0])
        @test_throws ArgumentError VLBIExponential([1.0, -1.0])
        @test_throws ArgumentError VLBITDist([5.0, -1.0], 0.0, 1.0)
        @test_throws ArgumentError VLBITDist([5.0, 1.0], [0.0, 0.0], [1.0, -1.0])
        @test_throws ArgumentError VLBITDist([5.0, 1.0], [0.0, 0.0], [1.0, 1.0, 1.0])
        @test_throws ArgumentError VLBIUniform(2.0, [1.0, 3.0])
        @test_throws ArgumentError VLBIUniform([0.0, 5.0], 4.0)
    end

    @testset "insupport for VLBI* (scalar and array)" begin
        # Scalar
        @test insupport(VLBIGaussian(0.0, 1.0), 0.5)
        @test insupport(VLBIExponential(1.0), 0.5)
        @test !insupport(VLBIExponential(1.0), -0.1)
        @test insupport(VLBIUniform(-1.0, 1.0), 0.0)
        @test !insupport(VLBIUniform(-1.0, 1.0), 2.0)
        @test insupport(VLBIInverseGamma(2.0, 1.0), 0.5)
        @test !insupport(VLBIInverseGamma(2.0, 1.0), -0.5)
        @test insupport(VLBITDist(5.0), 100.0)

        # Array
        de = VLBIExponential(1.0, (2, 3))
        @test insupport(de, ones(2, 3))
        @test !insupport(de, -ones(2, 3))
        @test !insupport(de, ones(2, 4))  # wrong size

        du = VLBIUniform(-1.0, 1.0, (2, 3))
        @test insupport(du, zeros(2, 3))
        @test !insupport(du, fill(2.0, 2, 3))

        dig = VLBIInverseGamma(2.0, 1.0, (2, 3))
        @test insupport(dig, ones(2, 3))
        @test !insupport(dig, -ones(2, 3))
    end

    @testset "Dists.std for AffineDistribution" begin
        d = VLBIGaussian(0.0, 1.5)
        @test std(d) ≈ 1.5
        d2 = VLBIGaussian(0.0, 1.5, (2, 3))
        @test std(d2) ≈ fill(1.5, 2, 3)
    end

    @testset "Base.show for AffineDistribution covers all base types" begin
        # Verify show emits the expected base-type name and handles array params.
        for d in (
                VLBIExponential(2.0, (2, 2)),
                VLBIUniform(0.0, 1.0, (2, 2)),
                VLBIInverseGamma(3.0, 1.0, (2, 2)),
                VLBITDist(5.0, 0.0, 1.0, (2, 2)),
            )
            io = IOBuffer()
            show(io, d)
            s = String(take!(io))
            @test occursin("AffineDistribution(", s)
            @test occursin("size=", s)
        end

        # Scalar (N = 0) — no size annotation.
        io = IOBuffer()
        show(io, VLBIGaussian(0.0, 1.0))
        @test !occursin("size=", String(take!(io)))

        # Array params — eltype/shape summary path.
        io = IOBuffer()
        μ = randn(2, 3)
        show(io, VLBIGaussian(μ, 1.0))
        s = String(take!(io))
        @test occursin("Float64", s)
    end

    @testset "params for non-Normal AffineDistribution bases" begin
        @test params(VLBIExponential(2.5, (2, 2))) == (2.5,)
        # VLBIUniform stores (loc, loc + scale) ≈ (a, b)
        a, b = params(VLBIUniform(-1.0, 3.0, (2, 2)))
        @test a == -1.0
        @test b == 3.0
        @test params(VLBIInverseGamma(2.0, 1.5, (2, 2))) == (2.0, 1.5)
        @test params(VLBITDist(5.0, 0.3, 1.2, (2, 2))) == (5.0, 0.3, 1.2)
    end

    @testset "mean / var match Distributions for non-Normal arrays" begin
        de = VLBIExponential(2.0, (2, 3))
        @test mean(de) ≈ fill(2.0, 2, 3)
        @test var(de) ≈ fill(4.0, 2, 3)

        du = VLBIUniform(-1.0, 3.0, (2, 3))
        @test mean(du) ≈ fill(1.0, 2, 3)
        @test var(du) ≈ fill((3.0 - (-1.0))^2 / 12, 2, 3)

        dig = VLBIInverseGamma(3.0, 2.0, (2, 3))
        @test mean(dig) ≈ fill(2.0 / (3.0 - 1.0), 2, 3)
        @test var(dig) ≈ fill(2.0^2 / ((3.0 - 1.0)^2 * (3.0 - 2.0)), 2, 3)

        # TDist: mean defined for ν > 1, var for ν > 2
        dt = VLBITDist(5.0, 0.0, 1.0, (2, 3))
        @test mean(dt) ≈ zeros(2, 3)
        @test var(dt) ≈ fill(5.0 / (5.0 - 2.0), 2, 3)
    end

    @testset "Std bases — scalar logpdf / cdf / quantile" begin
        # StdInverseGamma scalar
        sig = StdInverseGamma(3.0)
        @test logpdf(sig, 0.5) ≈ logpdf(InverseGamma(3.0, 1.0), 0.5)
        for p in (0.1, 0.5, 0.9)
            q = quantile(sig, p)
            @test cdf(sig, q) ≈ p atol = 1.0e-7
        end
        # mean/var for scalar
        @test mean(sig) ≈ 1 / (3.0 - 1.0)
        @test var(sig) ≈ 1 / ((3.0 - 1.0)^2 * (3.0 - 2.0))
        # mean/var when undefined → Inf
        @test mean(StdInverseGamma(0.5)) == Inf
        @test var(StdInverseGamma(1.5)) == Inf

        # StdTDist scalar
        st = StdTDist(5.0)
        @test logpdf(st, 0.3) ≈ logpdf(TDist(5.0), 0.3)
        for p in (0.05, 0.25, 0.5, 0.75, 0.95)
            q = quantile(st, p)
            @test cdf(st, q) ≈ p atol = 1.0e-7
        end
        @test mean(st) == 0.0
        @test var(st) ≈ 5.0 / (5.0 - 2.0)
        @test isnan(mean(StdTDist(0.5)))
        @test var(StdTDist(1.5)) == Inf

        # StdExponential scalar
        se = StdExponential()
        @test logpdf(se, 1.0) ≈ -1.0
        @test cdf(se, 1.0) ≈ 1 - exp(-1.0)
        @test quantile(se, 0.5) ≈ -log(0.5)
        @test mean(se) ≈ 1.0
        @test var(se) ≈ 1.0

        # StdUniform scalar
        su = StdUniform()
        @test logpdf(su, 0.5) ≈ 0.0
        @test logpdf(su, 1.5) == -Inf
        @test cdf(su, 0.3) ≈ 0.3
        @test quantile(su, 0.7) ≈ 0.7
        @test mean(su) ≈ 0.5
        @test var(su) ≈ 1 / 12
    end

    @testset "Std bases — array mean/var/insupport" begin
        # StdInverseGamma — both scalar-α and array-α arrays.
        sig_s = StdInverseGamma(3.0, (2, 3))
        @test mean(sig_s) ≈ fill(1 / (3.0 - 1.0), 2, 3)
        @test var(sig_s) ≈ fill(1 / ((3.0 - 1.0)^2 * (3.0 - 2.0)), 2, 3)

        αa = abs.(randn(2, 3)) .+ 2.5
        sig_a = StdInverseGamma(αa)
        @test mean(sig_a) ≈ @. 1 / (αa - 1)
        @test var(sig_a) ≈ @. 1 / ((αa - 1)^2 * (αa - 2))
        @test insupport(sig_a, abs.(randn(2, 3)) .+ 0.1)
        @test !insupport(sig_a, -ones(2, 3))
        @test !insupport(sig_a, ones(2, 4))

        # StdTDist — both scalar-ν and array-ν arrays.
        st_s = StdTDist(5.0, (2, 3))
        @test mean(st_s) ≈ zeros(2, 3)
        @test var(st_s) ≈ fill(5.0 / 3.0, 2, 3)

        νa = abs.(randn(2, 3)) .+ 4.0
        st_a = StdTDist(νa)
        @test mean(st_a) ≈ zeros(2, 3)
        @test var(st_a) ≈ @. νa / (νa - 2)
        @test insupport(st_a, randn(2, 3))
        @test !insupport(st_a, randn(2, 4))

        # StdExponential / StdUniform — array mean/var
        @test mean(StdExponential((2, 3))) ≈ ones(2, 3)
        @test var(StdExponential((2, 3))) ≈ ones(2, 3)
        @test mean(StdUniform((2, 3))) ≈ fill(0.5, 2, 3)
        @test var(StdUniform((2, 3))) ≈ fill(1 / 12, 2, 3)

        # StdNormal — mean/var/cov
        sn = StdNormal((2, 3))
        @test mean(sn) ≈ zeros(2, 3)
        @test var(sn) ≈ ones(2, 3)
        @test size(cov(sn)) == (6, 6)
    end

    @testset "Std bases — sampling for array params" begin
        αa = abs.(randn(2, 3)) .+ 2.0
        sig_a = StdInverseGamma(αa)
        z = rand(sig_a)
        @test size(z) == (2, 3)
        @test all(>(0), z)

        νa = abs.(randn(2, 3)) .+ 3.0
        st_a = StdTDist(νa)
        z2 = rand(st_a)
        @test size(z2) == (2, 3)
    end

    @testset "Std bases — array logpdf size mismatch errors" begin
        @test_throws Exception logpdf(StdNormal((2, 3)), zeros(2, 4))
        @test_throws Exception logpdf(StdExponential((2, 3)), zeros(2, 4))
        @test_throws Exception logpdf(StdUniform((2, 3)), zeros(2, 4))
        @test_throws Exception logpdf(StdInverseGamma(2.0, (2, 3)), ones(2, 4))
        @test_throws Exception logpdf(StdTDist(5.0, (2, 3)), zeros(2, 4))

        # AffineDistribution scalar bases
        @test_throws Exception logpdf(VLBIGaussian(0.0, 1.0, (2, 3)), zeros(2, 4))
        @test_throws Exception logpdf(VLBIExponential(1.0, (2, 3)), zeros(2, 4))
    end

    @testset "StdNormal ascube round-trip" begin
        sn = StdNormal((6,))
        c = HypercubeTransform.ascube(sn)
        @test HypercubeTransform.dimension(c) == 6
        u = rand(HypercubeTransform.dimension(c))
        x = HypercubeTransform.transform(c, u)
        u_back = HypercubeTransform.inverse(c, x)
        @test u_back ≈ u
    end

    @testset "AffineDistribution direct construction from Std bases" begin
        # Construct directly from the public AffineDistribution(loc, scale, base)
        # API for each Std base — exercises the dispatch tables outside the
        # VLBI* helper families.
        d = AffineDistribution(0.5, 2.0, StdExponential((2, 3)))
        x = rand(d)
        @test size(x) == (2, 3)
        @test logpdf(d, x) ≈ unnormed_logpdf(d, x) + lognorm(d)
        @test all(>=(0.5), x)

        d2 = AffineDistribution(-1.0, 2.0, StdUniform((2, 3)))
        x2 = rand(d2)
        @test all(xi -> -1.0 <= xi <= 1.0, x2)
        @test logpdf(d2, x2) ≈ unnormed_logpdf(d2, x2) + lognorm(d2)

        # Per-element loc/scale combined with a scalar-param Std base.
        loc = randn(2, 3)
        scale = abs.(randn(2, 3)) .+ 0.5
        d3 = AffineDistribution(loc, scale, StdNormal((2, 3)))
        @test mean(d3) ≈ loc
        @test var(d3) ≈ scale .^ 2
    end

end
