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

    show(d)
    serialize("temp_srf.jls", d)
    d3 = deserialize("temp_srf.jls")
    rm("temp_srf.jls")
    @test d3.kx == d.kx
    @test d3.ky == d.ky
    @test d3.executor === d.executor
end


function testps(ps, plan; nsamples=10_000)
    d = std_dist(plan)

    @inferred genfield(StationaryRandomField(ps, plan), rand(d))

    x = map(1:nsamples) do _
        z = rand(d)
        return genfield(StationaryRandomField(ps, plan), z)
    end
    m = mean(x)
    @test all(x->isapprox(x, 0.0, atol=5e-2), m)
    s = var(x)
    @test all(x-> 0.5 < x < 2.0, s)
end

@testset "Power Spectra" begin
    g = imagepixels(10.0, 10.0, 64, 64)
    pl = StationaryRandomFieldPlan(g)
    gth = imagepixels(10.0, 10.0, 64, 64; executor=ThreadsEx())
    plth = StationaryRandomFieldPlan(gth)
    @testset "MaternPS" begin
        ps = MaternPS(10.0, 1.0)
        testps(ps, pl)
        testps(ps, plth)
    end

    @testset "SqExp" begin
        ps = MaternPS(10.0, 1.0)
        testps(ps, pl)
    end

    @testset "RationalQuadPS" begin
        ps = MaternPS(10.0, 1.0)
        testps(ps, pl)
    end

    @testset "MarkovPS" begin
        @testset "Order 1" begin
            ps = MarkovPS((10.0, ))
            testps(ps, pl)
        end

        @testset "Order 2" begin
            ps = MarkovPS((0.0, 10.0, ))
            testps(ps, pl)
        end

        @testset "Order 2" begin
            ps = MarkovPS((0.0, 10.0, ))
            testps(ps, pl)
        end

        @testset "Order 3" begin
            ps = MarkovPS((0.0, 00.0, 10.0))
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
    end
end



@testset "matern" begin
    trf_wide, dstd_w = matern((8, 10))
    trf_tall, dstd_t = matern((10, 8))
    trf_square, dstd_s = matern((10, 10))

    zw = rand(dstd_w)
    zt = rand(dstd_t)
    zs = rand(dstd_s)

    @test trf_wide(zw, (3.0, 3.0), 0.0, 1.0) ≈ trf_wide(zw, 3.0, 1.0)
    @test trf_tall(zt, (3.0, 3.0), 0.0, 1.0) ≈ trf_tall(zt, 3.0, 1.0)
    @test trf_square(zs, (3.0, 3.0), 0.0, 1.0) ≈ trf_square(zs, 3.0, 1.0)
end

