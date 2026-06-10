@testset "ImageUniform" begin
    d1 = ImageUniform(0.0, 2.0, 2, 3)
    d2 = reshape(product_distribution([Uniform(0.0, 2.0) for _ in 1:(2 * 3)]), 2, 3)

    x0 = rand(d1)
    @test logdensityof(d1, x0) ≈ logdensityof(d2, x0)
    @test mean(d1) ≈ mean(d2)
    @test size(d1) == size(d2)

    t = transport_to(d1, TVFlat())
    p0 = pullback(t, x0)

    @test transport(t, p0) ≈ x0

    function ℓpt(x)
        _, ld = transport_and_logdensity(t, x)
        return ld
    end

    @test isapprox(enzyme_grad(ℓpt, p0), fdm_grad(ℓpt, p0); atol = 1.0e-5)

end

@testset "ImageSphericalUniform" begin
    d = ImageSphericalUniform(2, 3)
    xx = rand(d)
    Distributions.rand!(d, xx)
    norms = map(hypot, xx...)
    @test norms ≈ fill(1.0, size(d))
    t = transport_to(d, TVFlat())
    px = pullback(t, xx)
    @test prod(transport(t, px) .≈ xx)
    @test logdensityof(d, xx) ≈ -6 * log(4π)

    function f(x)
        _, ld = transport_and_logdensity(t, x)
        return ld
    end
    @test isapprox(enzyme_grad(f, px), fdm_grad(f, px); atol = 1.0e-6)
end

@testset "ImageDirichlet" begin
    npix = 10
    d1 = Dirichlet(npix^2, 1.0)
    d2 = ImageDirichlet(1.0, npix, npix)
    d3 = ImageDirichlet(rand(10, 10) .+ 0.1)

    t1 = transport_to(d1, TVFlat())
    t2 = transport_to(d2, TVFlat())
    t3 = transport_to(d3, TVFlat())

    @test all(x -> x[1] ≈ x[2], zip(mean(d1), mean(d2)))

    ndim = dimension(t1)
    y0 = fill(0.1, ndim)

    x1, l1 = transport_and_logdensity(t1, y0)
    x2, l2 = transport_and_logdensity(t2, y0)


    @test dimension(t1) == dimension(t2)
    @test length(d1) == prod(size(d2))

    @test logdensityof(d1, x1) ≈ logdensityof(d2, x2)
    function ℓpt(x)
        _, ld = transport_and_logdensity(t2, x)
        return ld
    end

    @test isapprox(enzyme_grad(ℓpt, y0), fdm_grad(ℓpt, y0); atol = 1.0e-5)

    # Gradient cross-check at a moderate latent point. (Using `pullback` of a
    # random Dirichlet draw can land on an extreme latent where the stick-breaking
    # `logistic` saturates and the central-difference reference underflows, even
    # though Enzyme returns the correct analytic value.)
    Random.seed!(1234)
    y3 = randn(dimension(t3)) ./ 2
    function ℓpt3(x)
        _, ld = transport_and_logdensity(t3, x)
        return ld
    end

    @test isapprox(enzyme_grad(ℓpt3, y3), fdm_grad(ℓpt3, y3); atol = 1.0e-5)
end
