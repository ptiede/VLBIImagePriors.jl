@testset "ImageUniform" begin
    d1 = ImageUniform(0.0, 2.0, 2, 3)
    d2 = reshape(product_distribution([Uniform(0.0, 2.0) for _ in 1:(2 * 3)]), 2, 3)

    x0 = rand(d1)
    @test logdensityof(d1, x0) ≈ logdensityof(d2, x0)
    @test mean(d1) ≈ mean(d2)
    @test size(d1) == size(d2)

    t = asflat(d1)
    p0 = inverse(t, x0)

    @test transform(t, p0) ≈ x0
    test_rrule(Distributions._logpdf, d1 ⊢ NoTangent(), fill(0.5, size(x0)), atol = 1.0e-8)

    ℓ = logdensityof(d1)
    function ℓpt(x)
        y, lj = transform_and_logjac(t, x)
        return ℓ(y) + lj
    end

    s = central_fdm(5, 1)
    g1 = first(grad(s, ℓpt, p0))
    g2 = first(Zygote.gradient(ℓpt, p0))
    @test first(g1) ≈ first(g2)

end

@testset "ImageSphericalUniform" begin
    d = ImageSphericalUniform(2, 3)
    xx = rand(d)
    Distributions.rand!(d, xx)
    norms = map(hypot, xx...)
    @test norms ≈ fill(1.0, size(d))
    t = asflat(d)
    px = inverse(t, xx)
    @test prod(transform(t, px) .≈ xx)
    @test logdensityof(d, xx) ≈ -6 * log(4π)

    function f(x)
        y, lj = transform_and_logjac(t, x)
        return logdensityof(d, y) + lj
    end
    gz = Zygote.gradient(f, px)
    m = central_fdm(5, 1)
    gfd = FiniteDifferences.grad(m, f, px)
    @test isapprox(first(gz), first(gfd), atol = 1.0e-6)
end

@testset "ImageDirichlet" begin
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


    @test dimension(t1) == dimension(t2)
    @test length(d1) == prod(size(d2))

    @test logdensityof(d1, x1) ≈ logdensityof(d2, x2)
    ℓ2 = logdensityof(d2)

    function ℓpt(x)
        y, lj = transform_and_logjac(t2, x)
        return ℓ2(y) + lj
    end

    ℓpt(y0)
    ℓpt'(y0)
    s = central_fdm(5, 1)
    g1 = first(grad(s, ℓpt, y0))
    g2 = first(Zygote.gradient(ℓpt, y0))

    x3 = rand(d3)

    y3 = inverse(t3, x3)
    ℓ3 = logdensityof(d3)
    function ℓpt3(x)
        y, lj = transform_and_logjac(t3, x)
        return ℓ3(y) + lj
    end

    ℓpt(y3)
    s = central_fdm(5, 1)
    g1 = first(grad(s, ℓpt3, y3))
    g2 = first(Zygote.gradient(ℓpt3, y3))

    @test g1 ≈ g2
end
