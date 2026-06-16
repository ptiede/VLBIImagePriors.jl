@testset "Reactant-friendly distributions" begin

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
            t = transport_to(d, TVFlat())
            x = rand(d)
            p = latent_pback(t, x)
            @test latent_pfwd(t, p) ≈ x
        end
    end

    @testset "shared-params shape (VLBIGaussian(μ, σ, dims))" begin
        d = VLBIGaussian(0.5, 1.3, (3, 4))
        @test size(d) == (3, 4)
        x = randn(3, 4)
        @test logpdf(d, x) ≈ sum(logpdf.(Normal(0.5, 1.3), x))
        y = rand(d)
        @test size(y) == (3, 4)

        t = transport_to(d, TVFlat())
        p = latent_pback(t, y)
        @test latent_pfwd(t, p) ≈ y
    end

    @testset "per-element parameters (same family across grid)" begin
        μ = randn(3, 4)
        σ = abs.(randn(3, 4)) .+ 0.1
        d = VLBIGaussian(μ, σ)
        x = μ .+ σ .* randn(3, 4)
        @test logpdf(d, x) ≈ sum(logpdf.(Normal.(μ, σ), x))

        t = transport_to(d, TVFlat())
        p = latent_pback(t, x)
        @test latent_pfwd(t, p) ≈ x

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
            t = transport_to(b, TVFlat())
            p = latent_pback(t, z)
            @test latent_pfwd(t, p) ≈ z
        end

        # scalar bases
        for b in (
                StdNormal{Float64, 0}(()), StdExponential(), StdUniform(),
                StdInverseGamma(2.0), StdTDist(5.0),
            )
            z = rand(b)
            @test isfinite(logpdf(b, z))
            t = transport_node(b, TVFlat())
            @test t isa TV.AbstractTransform
        end
    end

    @testset "HierarchicalPrior with VLBIGaussian + VLBIExponential" begin
        h = HierarchicalPrior(ρ -> VLBIGaussian(0.0, ρ, (3, 4)), VLBIExponential(1.0))
        x = rand(h)
        @test x isa NamedTuple
        @test isfinite(logpdf(h, x))

        t = transport_to(h, TVFlat())
        y = randn(dimension(t))
        xback = latent_pfwd(t, y)
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

        g_fd = fdm_grad(f, x)
        g_en = enzyme_grad(f, x)
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
            @test occursin("PushforwardDistribution(", s)
            @test occursin("size=", s)
        end

        # Scalar (N = 0) — no size annotation.
        io = IOBuffer()
        show(io, VLBIGaussian(0.0, 1.0))
        @test !occursin("size=", String(take!(io)))

        # Array params — base/shape summary path.
        io = IOBuffer()
        μ = randn(2, 3)
        show(io, VLBIGaussian(μ, 1.0))
        s = String(take!(io))
        @test occursin("StdNormal", s)
        @test occursin("size=(2, 3)", s)
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

    @testset "StdNormal ascube round-trip" begin
        sn = StdNormal((6,))
        c = transport_to(sn, StdUniform())
        @test dimension(c) == 6
        u = rand(dimension(c))
        x = latent_pfwd(c, u)
        u_back = latent_pback(c, x)
        @test u_back ≈ u
    end

    @testset "StdUniform routing: 0-dim → scalar value, N>=1 → array value" begin
        # 0-dim (univariate) distributions latent_pfwd a length-1 latent to a
        # scalar value; N>=1 distributions latent_pfwd to an array of the
        # distribution's shape.
        scalar_cases = [
            VLBIGaussian(0.0, 1.0),
            VLBIUniform(0.0, 1.0),
            VLBIExponential(2.0),
            VLBIInverseGamma(2.0, 1.0),
            VLBITDist(5.0),
            StdNormal(),
            StdExponential(),
            StdUniform(),
            StdInverseGamma(2.5),
            StdTDist(5.0),
            VLBITruncated(VLBIGaussian(0.0, 1.0), -1.0, 2.0),
            VLBITruncated(VLBIExponential(1.0), nothing, 3.0),
            VLBITruncated(VLBIUniform(-2.0, 2.0), -1.0, 1.5),
            VLBITruncated(VLBIGaussian(0.0, 1.0), 0.5, nothing),
        ]
        for d in scalar_cases
            c = transport_to(d, StdUniform())
            @test dimension(c) == 1
            u = rand(1)
            x = latent_pfwd(c, u)
            @test x isa Number
            @test latent_pback(c, x) ≈ u
        end

        # Matrixvariate cases — used to fall off HC's `ascube` dispatch
        # table because HC's Union only catches multivariate.
        matrix_cases = [
            StdExponential((2, 3)),
            VLBIGaussian(2.0, 1.5, (3, 4)),
            VLBIInverseGamma(abs.(randn(2, 3)) .+ 1.5, abs.(randn(2, 3)) .+ 0.5),
            VLBITDist(abs.(randn(2, 3)) .+ 2.0, zeros(2, 3), ones(2, 3)),
        ]
        for d in matrix_cases
            c = transport_to(d, StdUniform())
            u = rand(dimension(c))
            x = latent_pfwd(c, u)
            @test size(x) == size(d)
            @test latent_pback(c, x) ≈ u
        end
    end

    @testset "ascube N>=1 round-trip for Std bases and AffineDistributions" begin
        # N>=1 distributions go through ArrayHC. Includes both multivariate
        # (N=1) which HC's stock dispatch handles, and matrixvariate (N>=2)
        # which needs our explicit override (see std_normal.jl comment).
        cases = Any[
            StdExponential((6,)),
            StdUniform((4,)),
            StdInverseGamma(2.5, (3,)),
            StdInverseGamma(abs.(randn(4)) .+ 1.5),
            StdTDist(5.0, (3,)),
            StdTDist(abs.(randn(4)) .+ 2.0),
            VLBIGaussian(2.0, 1.5, (3, 4)),
            VLBIGaussian(randn(2, 3), abs.(randn(2, 3)) .+ 0.1),
            VLBIExponential(2.0, (3,)),
            VLBIExponential(abs.(randn(2, 3)) .+ 0.1),
            VLBIUniform(-1.0, 1.0, (3,)),
            VLBIUniform(randn(2, 3), randn(2, 3) .+ 5.0),
            VLBIInverseGamma(2.0, 1.0, (3,)),
            VLBIInverseGamma(abs.(randn(2, 3)) .+ 1.5, abs.(randn(2, 3)) .+ 0.5),
            VLBITDist(5.0, 0.0, 1.0, (3,)),
            VLBITDist(abs.(randn(2, 3)) .+ 2.0, zeros(2, 3), ones(2, 3)),
        ]
        for d in cases
            c = transport_to(d, StdUniform())
            u = rand(dimension(c))
            x = latent_pfwd(c, u)
            u_back = latent_pback(c, x)
            @test u_back ≈ u
        end
    end

    @testset "array asflat round-trip for all VLBI* families" begin
        # Covers the array asflat dispatches in affine.jl (StdNormal, StdTDist,
        # StdExponential, StdInverseGamma, StdUniform — both shared-param and
        # per-element constructors).
        cases = [
            VLBIGaussian(0.0, 1.0, (2, 3)),
            VLBIGaussian(randn(2, 3), abs.(randn(2, 3)) .+ 0.1),
            VLBIExponential(2.0, (2, 3)),
            VLBIExponential(abs.(randn(2, 3)) .+ 0.1),
            VLBIUniform(-1.0, 1.0, (2, 3)),
            VLBIUniform(randn(2, 3), randn(2, 3) .+ 5.0),
            VLBIInverseGamma(2.0, 1.0, (2, 3)),
            VLBIInverseGamma(abs.(randn(2, 3)) .+ 1.5, abs.(randn(2, 3)) .+ 0.5),
            VLBITDist(5.0, 0.0, 1.0, (2, 3)),
            VLBITDist(abs.(randn(2, 3)) .+ 2.0, zeros(2, 3), ones(2, 3)),
        ]
        for d in cases
            t = transport_to(d, TVFlat())
            @test dimension(t) == length(d)
            x = rand(d)
            p = latent_pback(t, x)
            @test latent_pfwd(t, p) ≈ x
        end
    end

    @testset "Std base array asflat round-trip" begin
        for b in (
                StdNormal((2, 3)),
                StdExponential((2, 3)),
                StdUniform((2, 3)),
                StdInverseGamma(2.0, (2, 3)),
                StdInverseGamma(abs.(randn(2, 3)) .+ 1.5),
                StdTDist(5.0, (2, 3)),
                StdTDist(abs.(randn(2, 3)) .+ 2.0),
            )
            t = transport_to(b, TVFlat())
            @test dimension(t) == length(b)
            z = rand(b)
            p = latent_pback(t, z)
            @test latent_pfwd(t, p) ≈ z
        end
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

    @testset "product_distribution lifts scalar VLBI* into 1D AffineDistribution" begin
        # Per family: an array of N scalar VLBI*(...) folds into one 1D
        # AffineDistribution(<:StdX, 1) of length N with per-element params.

        # Gaussian
        ds_g = [VLBIGaussian(Float64(i), 0.5 + 0.1 * i) for i in 1:4]
        p_g = product_distribution(ds_g)
        @test p_g isa PushforwardDistribution{<:Any, <:StdNormal, 1}
        @test length(p_g) == 4
        x = randn(4)
        @test logpdf(p_g, x) ≈ sum(logpdf(d, x[i]) for (i, d) in enumerate(ds_g))

        # Exponential
        ds_e = [VLBIExponential(0.5 + 0.2 * i) for i in 1:3]
        p_e = product_distribution(ds_e)
        @test p_e isa PushforwardDistribution{<:Any, <:StdExponential, 1}
        @test length(p_e) == 3
        y = abs.(randn(3)) .+ 0.1
        @test logpdf(p_e, y) ≈ sum(logpdf(d, y[i]) for (i, d) in enumerate(ds_e))

        # Uniform
        ds_u = [VLBIUniform(-Float64(i), Float64(i + 1)) for i in 1:3]
        p_u = product_distribution(ds_u)
        @test p_u isa PushforwardDistribution{<:Any, <:StdUniform, 1}
        @test length(p_u) == 3
        xu = [0.0, 0.5, 1.0]
        @test logpdf(p_u, xu) ≈ sum(logpdf(d, xu[i]) for (i, d) in enumerate(ds_u))

        # InverseGamma
        ds_ig = [VLBIInverseGamma(2.0 + 0.5 * i, 1.0 + 0.1 * i) for i in 1:3]
        p_ig = product_distribution(ds_ig)
        @test p_ig isa PushforwardDistribution{<:Any, <:StdInverseGamma, 1}
        @test length(p_ig) == 3
        xig = abs.(randn(3)) .+ 0.5
        @test logpdf(p_ig, xig) ≈ sum(logpdf(d, xig[i]) for (i, d) in enumerate(ds_ig))

        # TDist
        ds_t = [VLBITDist(3.0 + Float64(i), Float64(i), 1.0 + 0.1 * i) for i in 1:3]
        p_t = product_distribution(ds_t)
        @test p_t isa PushforwardDistribution{<:Any, <:StdTDist, 1}
        @test length(p_t) == 3
        xt = randn(3)
        @test logpdf(p_t, xt) ≈ sum(logpdf(d, xt[i]) for (i, d) in enumerate(ds_t))
    end

    @testset "VLBITruncated" begin
        # Truncated normal — compare to Distributions.truncated for cross-check.
        d_base = VLBIGaussian(0.0, 1.0)
        lower, upper = -1.5, 2.0
        d = VLBITruncated(d_base, lower, upper)
        ref = truncated(Normal(0.0, 1.0), lower, upper)

        # logpdf inside support
        for x in (-1.0, 0.0, 0.7, 1.5)
            @test logpdf(d, x) ≈ logpdf(ref, x) atol = 1.0e-10
            @test logpdf(d, x) ≈ unnormed_logpdf(d, x) + lognorm(d)
        end

        # Outside support → -Inf
        @test logpdf(d, lower - 0.1) == -Inf
        @test logpdf(d, upper + 0.1) == -Inf

        # cdf / quantile inverses
        for p in (0.05, 0.25, 0.5, 0.75, 0.95)
            q = quantile(d, p)
            @test lower <= q <= upper
            @test cdf(d, q) ≈ p atol = 1.0e-8
            @test quantile(d, p) ≈ quantile(ref, p) atol = 1.0e-8
            @test cdf(d, q) ≈ cdf(ref, q) atol = 1.0e-8
        end

        # cdf clamping at the boundaries
        @test cdf(d, lower - 1.0) == 0
        @test cdf(d, upper + 1.0) == 1

        # Sampling stays inside the truncation interval
        rng = Random.MersenneTwister(42)
        samples = [rand(rng, d) for _ in 1:5000]
        @test all(s -> lower <= s <= upper, samples)
        # And matches Distributions.truncated within sampling noise
        @test isapprox(mean(samples), mean(ref); atol = 0.05)

        # `lognorm` is data-independent — recompute, must match
        ln1 = lognorm(d)
        for _ in 1:5
            @test lognorm(d) === ln1 || lognorm(d) ≈ ln1
        end

        # insupport
        @test insupport(d, 0.0)
        @test !insupport(d, lower - 0.5)
        @test !insupport(d, upper + 0.5)

        # flat: bounded interval transform
        t = transport_node(d, TVFlat())
        @test t isa TV.AbstractTransform
        for _ in 1:5
            x = rand(rng, d)
            p = latent_pback(t, x)
            @test latent_pfwd(t, p) ≈ x
        end

        # Truncated exponential — non-symmetric base
        d2 = VLBITruncated(VLBIExponential(2.0), 0.5, 5.0)
        ref2 = truncated(Distributions.Exponential(2.0), 0.5, 5.0)
        for x in (0.6, 1.0, 2.0, 4.0)
            @test logpdf(d2, x) ≈ logpdf(ref2, x) atol = 1.0e-10
        end

        # Keyword API — matches Distributions.truncated
        @test logpdf(VLBITruncated(d_base; lower = -1.0, upper = 1.0), 0.0) ≈
            logpdf(truncated(Normal(0, 1); lower = -1.0, upper = 1.0), 0.0)

        # One-sided: left-truncated only (X >= lower)
        dl = VLBITruncated(d_base; lower = 0.0)
        refl = truncated(Normal(0.0, 1.0); lower = 0.0)
        for x in (0.1, 0.5, 1.5, 3.0)
            @test logpdf(dl, x) ≈ logpdf(refl, x) atol = 1.0e-10
        end
        @test logpdf(dl, -0.5) == -Inf
        @test cdf(dl, 0.0) ≈ 0
        @test cdf(dl, 1.0e6) ≈ 1
        # Sampling stays >= lower
        @test all(s -> s >= 0.0, [rand(rng, dl) for _ in 1:1000])

        # One-sided: right-truncated only (X <= upper)
        dr = VLBITruncated(d_base; upper = 0.0)
        refr = truncated(Normal(0.0, 1.0); upper = 0.0)
        for x in (-3.0, -1.0, -0.5, -0.1)
            @test logpdf(dr, x) ≈ logpdf(refr, x) atol = 1.0e-10
        end
        @test logpdf(dr, 0.5) == -Inf
        @test cdf(dr, -1.0e6) ≈ 0
        @test cdf(dr, 0.0) ≈ 1
        @test all(s -> s <= 0.0, [rand(rng, dr) for _ in 1:1000])

        # regression: support endpoints intersect the truncation bounds with the BASE
        # support, and `asflat` is built from that support. A one-sided truncation of a
        # bounded base (`VLBITruncated(VLBIExponential(0.1); upper=1)`) once produced an
        # asflat mapping ℝ → (-∞, 1): the flat space then contained a reachable
        # logpdf = -Inf region, yielding a negative-background optimum and a frozen NUTS
        # chain in production.
        @testset "one-sided truncation keeps the base support" begin
            dexp = VLBITruncated(VLBIExponential(0.1); upper = 1.0)
            refexp = truncated(Distributions.Exponential(0.1); upper = 1.0)
            @test minimum(dexp) == minimum(refexp) == 0.0
            @test maximum(dexp) == maximum(refexp) == 1.0
            # an explicit bound equal to the base bound is honored exactly
            # (a bound *outside* the base support is unconstructable here: the branchless
            # VLBI* cdf is only valid on the support, so the constructor DomainErrors)
            @test minimum(VLBITruncated(VLBIExponential(0.1), 0.0, 1.0)) == 0.0
            # insupport derives from the same endpoints (single source of truth)
            @test insupport(dexp, 0.0) && insupport(dexp, 1.0)
            @test !insupport(dexp, -eps()) && !insupport(dexp, nextfloat(1.0))
            # scalar AffineDistribution bases report their support (and flip with scale < 0)
            @test minimum(VLBIExponential(0.1)) == 0.0
            @test maximum(VLBIExponential(0.1)) == Inf
            @test minimum(VLBIUniform(-2.0, 3.0)) == -2.0
            @test maximum(VLBIUniform(-2.0, 3.0)) == 3.0
            # the flat transform respects the support for any latent value
            t = asflat(dexp)
            for y in (-30.0, 0.0, 30.0)
                x = TV.transform(t, y)
                @test 0.0 <= x <= 1.0
                @test isfinite(logpdf(dexp, max(x, eps())))
            end
        end
    end

    @testset "AffineDistribution with Matrix scale (linear operator)" begin
        # `AffineDistribution(μ, A, StdNormal((K,)))` with A::Matrix is the
        # MvNormal(μ, A * A') reparameterisation via Cholesky factor.
        rng = Random.MersenneTwister(0x00c0ffee)
        K = 4
        μ = randn(rng, K)
        Σ = let M = randn(rng, K, K)
            M * M' + I  # positive definite
        end
        L = cholesky(Σ).L
        A = Matrix(L)

        d = AffineDistribution(μ, A, StdNormal((K,)))
        ref = MvNormal(μ, Σ)

        # logpdf round-trip
        for _ in 1:20
            x = μ .+ A * randn(rng, K)
            @test logpdf(d, x) ≈ logpdf(ref, x) atol = 1.0e-8
            @test logpdf(d, x) ≈ unnormed_logpdf(d, x) + lognorm(d)
        end

        # Moments
        @test mean(d) ≈ mean(ref)
        @test cov(d) ≈ cov(ref)
        @test var(d) ≈ diag(Σ)

        # Sampling — empirical mean/cov over a large sample. Use elementwise
        # tolerance: `isapprox` on matrices uses the Frobenius norm, which
        # accumulates K^2 element-level errors and trips much sooner than
        # any individual entry would.
        n = 200_000
        samples = reduce(hcat, [rand(rng, d) for _ in 1:n])
        @test maximum(abs, vec(mean(samples; dims = 2)) .- μ) < 0.05
        emp_cov = cov(samples; dims = 2)
        @test maximum(abs, emp_cov .- Σ) < 0.2

        # insupport — Normal base accepts everything in ℝᴷ
        @test insupport(d, randn(rng, K))
        @test !insupport(d, randn(rng, K + 1))   # wrong length

        # asflat: 1D unconstrained, dimension == K
        t = transport_to(d, TVFlat())
        @test dimension(t) == K
        x = rand(rng, d)
        p = latent_pback(t, x)
        @test latent_pfwd(t, p) ≈ x
    end

    @testset "VLBIBeta" begin
        rng = Random.MersenneTwister(0xbe7a)

        @testset "interface (length / eltype / insupport)" begin
            ds = VLBIBeta(2.0, 5.0)
            @test length(ds) == 1
            @test eltype(ds) == Float64

            α = [2.0, 3.0, 1.5]
            β = [5.0, 2.0, 4.0]
            dv = VLBIBeta(α, β)
            @test length(dv) == 3
            @test eltype(dv) == Float64

            @test insupport(dv, [0.2, 0.7, 0.4])
            @test !insupport(dv, [0.2, 1.3, 0.4])    # > 1
            @test !insupport(dv, [-0.1, 0.7, 0.4])   # < 0
        end

        @testset "logpdf matches Distributions.Beta" begin
            # scalar params (against the scalar logpdf method)
            ds = VLBIBeta(2.0, 5.0)
            rb = Beta(2.0, 5.0)
            for x in 0.05:0.1:0.95
                @test logpdf(ds, x) ≈ logpdf(rb, x) atol = 1.0e-10
            end
            # out of support → -Inf
            @test logpdf(ds, -0.1) == -Inf
            @test logpdf(ds, 1.1) == -Inf

            # vector params == sum of independent marginals
            α = [2.0, 3.0, 1.5]
            β = [5.0, 2.0, 4.0]
            dv = VLBIBeta(α, β)
            for _ in 1:10
                x = rand(rng, 3)
                @test logpdf(dv, x) ≈ sum(logpdf.(Beta.(α, β), x)) atol = 1.0e-10
            end
            # any pixel out of support drags the whole density to -Inf
            @test logpdf(dv, [0.2, 1.5, 0.4]) == -Inf
        end

        @testset "unnormed_logpdf + lognorm == logpdf" begin
            α = [2.0, 3.0, 1.5]
            β = [5.0, 2.0, 4.0]
            dv = VLBIBeta(α, β)
            ln = lognorm(dv)
            for _ in 1:5
                x = rand(rng, 3)
                @test logpdf(dv, x) ≈ unnormed_logpdf(dv, x) + ln
            end
            # scalar form
            ds = VLBIBeta(2.0, 5.0)
            @test logpdf(ds, 0.3) ≈ unnormed_logpdf(ds, 0.3) + lognorm(ds)
        end

        @testset "sampler moments match Distributions.Beta" begin
            n = 200_000
            # scalar params: rand returns a length-1 vector
            ds = VLBIBeta(2.0, 5.0)
            rb = Beta(2.0, 5.0)
            ss = reduce(vcat, [rand(rng, ds) for _ in 1:n])
            @test all(0 .<= ss .<= 1)
            @test isapprox(mean(ss), mean(rb); atol = 5.0e-3)
            @test isapprox(var(ss), var(rb); atol = 5.0e-3)

            # vector params: per-component moments
            α = [2.0, 3.0, 1.5]
            β = [5.0, 2.0, 4.0]
            dv = VLBIBeta(α, β)
            mat = reduce(hcat, [rand(rng, dv) for _ in 1:n])
            @test all(0 .<= mat .<= 1)
            for k in 1:3
                rbk = Beta(α[k], β[k])
                @test isapprox(mean(mat[k, :]), mean(rbk); atol = 5.0e-3)
                @test isapprox(var(mat[k, :]), var(rbk); atol = 5.0e-3)
            end
        end
    end

end
