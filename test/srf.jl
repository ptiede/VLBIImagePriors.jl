using Serialization
using Statistics

@testset "StationaryRandomFieldPlan" begin
    g = imagepixels(10.0, 10.0, 8, 8)
    d = StationaryRandomFieldPlan(g)
    @test size(d) == (8, 8)
    @test eltype(d) == Float64
    @test length(d) == 64

    d2 = StationaryRandomFieldPlan(Float64, size(g))
    @test d2.kx == d.kx
    @test d2.ky == d.ky

    StationaryRandomFieldPlan(Float64, size(g), executor = :dummy)

    show(d)
    serialize("temp_srf.jls", d)
    d3 = deserialize("temp_srf.jls")
    rm("temp_srf.jls")
    @test d3.kx == d.kx
    @test d3.ky == d.ky
    @test d3.executor === d.executor
end


function testps(ps, plan; nsamples = 10_000)
    d = std_dist(plan)

    @inferred genfield(StationaryRandomField(ps, plan), rand(d))

    x = map(1:nsamples) do _
        z = rand(d)
        return genfield(StationaryRandomField(ps, plan), z)
    end
    m = mean(x)
    @test all(x -> isapprox(x, 0.0, atol = 8.0e-2), m)
    s = var(x)
    return @test all(x -> 0.5 < x < 2.0, s)
end

@testset "Power Spectra" begin
    g = imagepixels(10.0, 10.0, 64, 64)
    pl = StationaryRandomFieldPlan(g)
    gth = imagepixels(10.0, 10.0, 64, 64; executor = ThreadsEx())
    plth = StationaryRandomFieldPlan(gth)
    @testset "MaternPS" begin
        ps = MaternPS(10.0, 1.0)
        testps(ps, pl)
        testps(ps, plth)
    end

    @testset "SqExp" begin
        ps = SqExpPS(10.0)
        testps(ps, pl)
    end

    @testset "RationalQuadPS" begin
        ps = RationalQuadPS(10.0, 1.0)
        testps(ps, pl)
    end

    @testset "MarkovPS" begin
        @testset "Order 1" begin
            ps = MarkovPS((10.0,))
            testps(ps, pl)
        end

        @testset "Order 2" begin
            ps = MarkovPS((0.0, 10.0))
            testps(ps, pl)
        end

        @testset "Order 2" begin
            ps = MarkovPS((0.0, 10.0))
            testps(ps, pl)
        end

        @testset "Order 3" begin
            ps = MarkovPS((0.0, 0.0, 10.0))
            testps(ps, pl)
        end

        @testset "multi order 3" begin
            ps = MarkovPS((1.0, 10.0, 10.0))
            testps(ps, pl)
        end
    end

    @testset "ScaledPS" begin
        base = MaternPS(10.0, 1.0)
        ps = ScaledPS(base, 10.0, 1.0)
        testps(ps, pl)

        z = rand(std_dist(pl))
        f1 = genfield(StationaryRandomField(base, pl), z)
        f2 = genfield(StationaryRandomField(ScaledPS(base, 1.0), pl), z)
        @test f1 ≈ f2
    end
end


using VLBIImagePriors: StdNormal
@testset "StdNormal" begin
    g = imagepixels(10.0, 10.0, 64, 64)
    pl = StationaryRandomFieldPlan(g)
    rf = StationaryRandomField(MaternPS(10.0, 1.0), pl)
    d = std_dist(rf)

    d2 = StdNormal(size(g))
    @test length(d) == length(d2)
    @test eltype(d) == eltype(d2)
    @test mean(d) ≈ mean(d2)
    @test cov(d) ≈ cov(d2)
    asflat(d)

    @test Distributions.insupport(d, rand(d))

    z = rand(d)
    dd = Distributions.MvNormal(ones(length(d)))
    @test logpdf(d, z) ≈ logpdf(d2, z)
    @test logpdf(d, z) ≈ logpdf(dd, reshape(z, :))


    x = rand(d, 10_000)
    @test isapprox(mean(x), zeros(size(g)), atol = 5.0e-2, norm = maximum)
    @test isapprox(var(x), ones(size(g)), atol = 0.1, norm = maximum)

    dN = StdNormal((64, 64, 10, 3))
    z = rand(dN)
    @test logpdf(dN, z) ≈ logpdf(Distributions.MvNormal(zeros(length(dN)), I), reshape(z, :))
end


@testset "matern" begin
    trf_wide, dstd_w = matern((8, 10))
    trf_tall, dstd_t = matern((10, 8))
    trf_square, dstd_s = matern((10, 10))

    @test std_dist(trf_square) == dstd_s

    g = imagepixels(10.0, 10.0, 8, 10)
    t, d = matern(g)
    @test t.plan.kx == trf_wide.plan.kx
    @test t.plan.ky == trf_wide.plan.ky

    zw = rand(dstd_w)
    zt = rand(dstd_t)
    zs = rand(dstd_s)

    @test trf_wide(zw, (3.0, 3.0), 0.0, 1.0) ≈ trf_wide(zw, 3.0, 1.0)
    @test trf_tall(zt, (3.0, 3.0), 0.0, 1.0) ≈ trf_tall(zt, 3.0, 1.0)
    @test trf_square(zs, (3.0, 3.0), 0.0, 1.0) ≈ trf_square(zs, 3.0, 1.0)
end
