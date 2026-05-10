using Reactant, ComradeBase, VLBIImagePriors, Distributions
using Test

@testset "Reactant Ext" begin
    @testset "GMRF" begin
        d = GMRF(10.0, (6, 6))
        x = rand(d)
        xr = Reactant.to_rarray(x)

        @test @jit(logpdf(d, xr)) ≈ logpdf(d, x)

        cm = ConditionalMarkov(GMRF, (8, 8))
        f(cm, ρ, x) = logpdf(cm(ρ), x)
        @test @jit(f(cm, ConcreteRNumber(10.0), xr)) ≈ f(cm, 10.0, x)
    end

    @testset "SRF" begin
        g = imagepixels(10.0, 10.0, 16, 16)
        gr = @jit(identity(g))
        pl = StationaryRandomFieldPlan(g)
        plr = StationaryRandomFieldPlan(gr)
        rf = StationaryRandomField(MaternPS(10.0, 1.0), pl)
        d = std_dist(rf)
        x = rand(d)
        xr = Reactant.to_rarray(x)

        @test @jit(genfield(rf, xr)) ≈ genfield(rf, x)

        f2(pl, xr, ρs) = genfield(StationaryRandomField(MarkovPS(ρs), pl), xr)
        ρs = (10.0, 10.0)
        ρsr = ConcreteRNumber.(ρs)
        @test @jit(f2(plr, xr, ρsr)) ≈ f2(pl, x, ρs)
    end

    @testset "Transforms" begin
        d1 = DiagonalVonMises([0.5, 0.1], [inv(0.1), inv(π^2)])
        t = asflat(d1)

        x = randn(dimension(t))
        xr = Reactant.to_rarray(x)
        @test @jit(transform(t, xr)) ≈ transform(t, x)

        ds = ImageSphericalUniform(4, 4)
        ts = asflat(ds)
        xs = randn(dimension(ts))
        xrs = Reactant.to_rarray(xs)
        # Broken in Reactant currently
        # @test @jit(transform(ts, xrs)) ≈ transform(ts, xs)

    end

    @testset "Affine distributions" begin
        # Scalar with traced parameters and traced input
        f_g(μ, σ, x) = logpdf(VLBIGaussian(μ, σ), x)
        @test @jit(f_g(ConcreteRNumber(0.3), ConcreteRNumber(1.2), ConcreteRNumber(0.5))) ≈
            f_g(0.3, 1.2, 0.5)

        f_e(θ, x) = logpdf(VLBIExponential(θ), x)
        @test @jit(f_e(ConcreteRNumber(2.5), ConcreteRNumber(1.0))) ≈ f_e(2.5, 1.0)

        f_u(a, b, x) = logpdf(VLBIUniform(a, b), x)
        @test @jit(f_u(ConcreteRNumber(-1.0), ConcreteRNumber(3.0), ConcreteRNumber(0.5))) ≈
            f_u(-1.0, 3.0, 0.5)

        f_ig(α, θ, x) = logpdf(VLBIInverseGamma(α, θ), x)
        @test @jit(f_ig(ConcreteRNumber(3.0), ConcreteRNumber(2.0), ConcreteRNumber(1.5))) ≈
            f_ig(3.0, 2.0, 1.5)

        f_t(ν, μ, σ, x) = logpdf(VLBITDist(ν, μ, σ), x)
        @test @jit(
            f_t(
                ConcreteRNumber(5.0), ConcreteRNumber(0.3),
                ConcreteRNumber(1.2), ConcreteRNumber(0.7)
            )
        ) ≈ f_t(5.0, 0.3, 1.2, 0.7)

        # Array form: shared scalar parameters, traced input matrix
        d = VLBIGaussian(0.0, 1.0, (4, 4))
        x = randn(4, 4)
        xr = Reactant.to_rarray(x)
        @test @jit(logpdf(d, xr)) ≈ logpdf(d, x)

        # Per-element params with traced parameter arrays and traced input
        μ = randn(4, 4)
        σ = abs.(randn(4, 4)) .+ 0.1
        μr = Reactant.to_rarray(μ)
        σr = Reactant.to_rarray(σ)
        f_pe(μ, σ, x) = logpdf(VLBIGaussian(μ, σ), x)
        @test @jit(f_pe(μr, σr, xr)) ≈ f_pe(μ, σ, x)

        # Per-element TDist with traced ν, μ, σ
        νg = abs.(randn(4, 4)) .+ 2.0
        νr = Reactant.to_rarray(νg)
        f_tpe(ν, μ, σ, x) = logpdf(VLBITDist(ν, μ, σ), x)
        @test @jit(f_tpe(νr, μr, σr, xr)) ≈ f_tpe(νg, μ, σ, x)

        # `unnormed_logpdf` + `lognorm` split traces under Reactant (the
        # caching pathway — `lognorm` should be precomputable from traced
        # parameters and `unnormed_logpdf` evaluable on traced inputs).
        f_split(μ, σ, x) =
            unnormed_logpdf(VLBIGaussian(μ, σ), x) + lognorm(VLBIGaussian(μ, σ))
        @test @jit(f_split(μr, σr, xr)) ≈ f_split(μ, σ, x)

        αg = abs.(randn(4, 4)) .+ 1.5
        θg = abs.(randn(4, 4)) .+ 0.5
        αr = Reactant.to_rarray(αg)
        θr = Reactant.to_rarray(θg)
        yg = abs.(randn(4, 4)) .+ 0.1
        yr = Reactant.to_rarray(yg)
        f_ig_split(α, θ, y) =
            unnormed_logpdf(VLBIInverseGamma(α, θ), y) +
            lognorm(VLBIInverseGamma(α, θ))
        @test @jit(f_ig_split(αr, θr, yr)) ≈ f_ig_split(αg, θg, yg)
    end

    @testset "Matrix-scale AffineDistribution (MvNormal reparam)" begin
        # `AffineDistribution(μ, A, StdNormal((K,)))` ≡ `MvNormal(μ, A * A')`.
        # Traced params + traced input: `\\` (linear solve) and `logabsdet`
        # have to lower under Reactant for this to work.
        K = 3
        A = randn(K, K) + I  # well-conditioned, not symmetric
        μ = randn(K)
        x = μ .+ A * randn(K)
        Ar = Reactant.to_rarray(A)
        μr = Reactant.to_rarray(μ)
        xr = Reactant.to_rarray(x)
        f_mat(μ, A, x) = logpdf(AffineDistribution(μ, A, StdNormal((length(μ),))), x)
        @test @jit(f_mat(μr, Ar, xr)) ≈ f_mat(μ, A, x)
    end

    @testset "VLBITruncated" begin
        # Two-sided truncated normal with traced parameters + traced input.
        f_t2(μ, σ, lo, hi, x) = logpdf(VLBITruncated(VLBIGaussian(μ, σ), lo, hi), x)
        @test @jit(
            f_t2(
                ConcreteRNumber(0.0), ConcreteRNumber(1.0),
                ConcreteRNumber(-1.5), ConcreteRNumber(2.0),
                ConcreteRNumber(0.5)
            )
        ) ≈ f_t2(0.0, 1.0, -1.5, 2.0, 0.5)

        # Out-of-support → -Inf even when tracing
        @test @jit(
            f_t2(
                ConcreteRNumber(0.0), ConcreteRNumber(1.0),
                ConcreteRNumber(-1.5), ConcreteRNumber(2.0),
                ConcreteRNumber(3.0)
            )
        ) == -Inf

        # Left-truncated only
        f_lt(μ, σ, lo, x) = logpdf(VLBITruncated(VLBIGaussian(μ, σ); lower = lo), x)
        @test @jit(
            f_lt(
                ConcreteRNumber(0.0), ConcreteRNumber(1.0),
                ConcreteRNumber(0.0), ConcreteRNumber(0.5)
            )
        ) ≈ f_lt(0.0, 1.0, 0.0, 0.5)

        # Right-truncated only
        f_rt(μ, σ, hi, x) = logpdf(VLBITruncated(VLBIGaussian(μ, σ); upper = hi), x)
        @test @jit(
            f_rt(
                ConcreteRNumber(0.0), ConcreteRNumber(1.0),
                ConcreteRNumber(0.0), ConcreteRNumber(-0.5)
            )
        ) ≈ f_rt(0.0, 1.0, 0.0, -0.5)
    end

    @testset "product_distribution lift" begin
        # An array of scalar VLBIGaussians folds into one 1D
        # AffineDistribution; the lift uses `[d.loc for d in dists]` which
        # has to allow comprehensions over a vector of structs holding
        # traced numbers.
        function f_pd(μs, σs, x)
            ds = [VLBIGaussian(μs[i], σs[i]) for i in eachindex(μs)]
            return logpdf(product_distribution(ds), x)
        end
        μs = randn(4)
        σs = abs.(randn(4)) .+ 0.1
        x = μs .+ σs .* randn(4)
        μsr = Reactant.to_rarray(μs)
        σsr = Reactant.to_rarray(σs)
        xr = Reactant.to_rarray(x)
        @test_skip @jit(f_pd(μsr, σsr, xr)) ≈ f_pd(μs, σs, x)
    end

end
