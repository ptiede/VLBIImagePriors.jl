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
    end

end
