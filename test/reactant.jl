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


end
