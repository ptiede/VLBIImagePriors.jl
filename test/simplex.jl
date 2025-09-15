@testset "Simplex" begin

    npix = 10
    d1 = Dirichlet(npix^2, 1.0)
    d2 = ImageDirichlet(1.0, npix, npix)
    d3 = ImageDirichlet(rand(10, 10) .+ 0.1)

    t1 = asflat(d1)
    t2 = asflat(d2)
    t3 = asflat(d3)

    @test t2 === t3

    ndim = dimension(t1)
    y0 = fill(0.1, ndim)

    x1, l1 = transform_and_logjac(t1, y0)
    x2, l2 = transform_and_logjac(t2, y0)


    @testset "ImageSimplex" begin
        @test x1 ≈ reshape(x2, :)
        @test l1 ≈ l2

        @test inverse(t2, x2) ≈ y0
        test_rrule(
            VLBIImagePriors.simplex_fwd,
            TV.NoLogJac() ⊢ NoTangent(),
            VLBIImagePriors.ImageSimplex(10, 10) ⊢ NoTangent(),
            randn(99)
        )
        test_rrule(
            VLBIImagePriors.simplex_fwd,
            TV.LogJac() ⊢ NoTangent(),
            VLBIImagePriors.ImageSimplex(10, 10) ⊢ NoTangent(),
            randn(99)
        )
    end


end

@testset "Log Ratio Transform" begin
    x = randn(10, 10)
    ycl = to_simplex(CenteredLR(), x)
    yal = to_simplex(AdditiveLR(), x)

    @test length(ycl) == length(x)
    @test length(yal) == length(x)
    @test sum(ycl) ≈ 1
    @test sum(yal) ≈ 1

    test_rrule(to_simplex, AdditiveLR(), x)
    test_rrule(to_simplex, CenteredLR(), x)

    y = rand(10, 10) .+ 0.5
    far(x) = sum(abs2, to_real(AdditiveLR(), x / sum(x)))
    s = central_fdm(5, 1)
    gf_ar = first(grad(s, far, y))
    gz_ar = first(Zygote.gradient(far, y))
    @test isapprox(first(gf_ar), first(gz_ar), atol = 1.0e-6)

    fcr(x) = sum(abs2, to_real(CenteredLR(), x / sum(x)))
    gf_cr = first(grad(s, fcr, y))
    gz_cr = first(Zygote.gradient(fcr, y))
    @test isapprox(first(gf_cr), first(gz_cr), atol = 1.0e-6)


    # test_rrule(to_real, AdditiveLR(), yal)
    # test_rrule(to_real, CenteredLR(), ycl)

    x0al = to_real(AdditiveLR(), yal)
    x0cl = to_real(CenteredLR(), ycl)
    @test x0al[1:(end - 1)] ≈ x[1:(end - 1)]
    @test x0cl .- x0cl[1] ≈ x .- x[1]
end
