@testset "CenteredRegularizer" begin
    d = ImageDirichlet(1.0, 4, 4)
    @test_throws AssertionError CenteredRegularizer(1:2, 1:2, 1.0, d)
    x = y =  range(-2, 2, length=4)
    dc = CenteredRegularizer(x, y, 1.0, d)

    @test Distributions.insupport(dc, rand(4,4)) == Distributions.insupport(d, rand(4,4))

    test_rrule(VLBIImagePriors.lcol, dc⊢NoTangent(), rand(4,4))

    xx = rand(d)
    @test logpdf(d, xx) + VLBIImagePriors.lcol(dc, xx) ≈ logdensityof(dc, xx)

    @test asflat(dc) === asflat(d)

end


@testset "CenterImage" begin
    @testset "Equal" begin
        grid = imagepixels(10.0, 10.0, 48, 48)
        K = CenterImage(grid)
        img0 = IntensityMap(rand(48, 48), grid)
        K2 = CenterImage(img0)
        cimg = K(img0)
        @test K(img0) ≈ K2(img0)
        c0 = centroid(IntensityMap(cimg, grid))
        @test isapprox(c0[1], 0.0, atol=1e-6)
        @test isapprox(c0[2], 0.0, atol=1e-6)
        test_rrule(VLBIImagePriors.center_kernel, K.kernel⊢NoTangent(), rand(48, 48))
    end

    @testset "x Offset" begin
        grid = imagepixels(10.0, 10.0, 48, 48, 10.0)
        K = CenterImage(grid)
        img0 = IntensityMap(rand(48, 48), grid)
        K2 = CenterImage(img0)
        cimg = K(img0)
        @test K(img0) ≈ K2(img0)
        c0 = centroid(IntensityMap(cimg, grid))
        @test isapprox(c0[1], 0.0, atol=1e-6)
        @test isapprox(c0[2], 0.0, atol=1e-6)
    end

    @testset "y Offset" begin
        grid = imagepixels(10.0, 10.0, 48, 48, 0.0, 10.0)
        K = CenterImage(grid)
        img0 = IntensityMap(rand(48, 48), grid)
        K2 = CenterImage(img0)
        cimg = K(img0)
        @test K(img0) ≈ K2(img0)
        c0 = centroid(IntensityMap(cimg, grid))
        @test isapprox(c0[1], 0.0, atol=1e-6)
        @test isapprox(c0[2], 0.0, atol=1e-6)
    end

    @testset "Equal dims diff" begin
        grid = imagepixels(10.0, 10.0, 48, 24, 0.0, 0.0)
        K = CenterImage(grid)
        img0 = IntensityMap(rand(48, 24), grid)
        K2 = CenterImage(img0)
        cimg = K(img0)
        @test K(img0) ≈ K2(img0)
        c0 = centroid(IntensityMap(cimg, grid))
        @test isapprox(c0[1], 0.0, atol=1e-6)
        @test isapprox(c0[2], 0.0, atol=1e-6)
    end

    @testset "x offset dims diff" begin
        grid = imagepixels(10.0, 10.0, 48, 24, 10.0, 0.0)
        K = CenterImage(grid)
        img0 = IntensityMap(rand(48, 24), grid)
        K2 = CenterImage(img0)
        cimg = K(img0)
        @test K(img0) ≈ K2(img0)
        c0 = centroid(IntensityMap(cimg, grid))
        @test isapprox(c0[1], 0.0, atol=1e-6)
        @test isapprox(c0[2], 0.0, atol=1e-6)
    end

    @testset "y offset dims diff" begin
        grid = imagepixels(10.0, 10.0, 48, 24, 10.0, 0.0)
        K = CenterImage(grid)
        img0 = IntensityMap(rand(48, 24), grid)
        K2 = CenterImage(img0)
        cimg = K(img0)
        @test K(img0) ≈ K2(img0)
        c0 = centroid(IntensityMap(cimg, grid))
        @test isapprox(c0[1], 0.0, atol=1e-6)
        @test isapprox(c0[2], 0.0, atol=1e-6)
    end



end
