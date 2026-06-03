@testset "Simplex" begin

    npix = 10
    d1 = Dirichlet(npix^2, 1.0)
    d2 = ImageDirichlet(1.0, npix, npix)
    d3 = ImageDirichlet(rand(10, 10) .+ 0.1)

    t1 = transport_to(d1, StdFlat())
    t2 = transport_to(d2, StdFlat())
    t3 = transport_to(d3, StdFlat())

    # the flat transform is alpha-independent (same simplex for both priors)
    @test transport_node(d2, StdFlat()) === transport_node(d3, StdFlat())

    ndim = dimension(t1)
    y0 = fill(0.1, ndim)

    x1, l1 = transport_and_logjac(t1, y0)
    x2, l2 = transport_and_logjac(t2, y0)


    @testset "ImageSimplex" begin
        @test x1 ≈ reshape(x2, :)
        @test l1 ≈ l2

        @test pullback(t2, x2) ≈ y0
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

    y = rand(10, 10) .+ 0.5
    far(x) = sum(abs2, to_real(AdditiveLR(), x / sum(x)))
    @test isapprox(enzyme_grad(far, y), fdm_grad(far, y); atol = 1.0e-6)

    fcr(x) = sum(abs2, to_real(CenteredLR(), x / sum(x)))
    @test isapprox(enzyme_grad(fcr, y), fdm_grad(fcr, y); atol = 1.0e-6)

    x0al = @jit to_real(AdditiveLR(), yal)
    x0cl = @jit to_real(CenteredLR(), ycl)
    @test x0al[1:(end - 1)] ≈ x[1:(end - 1)]
    @test x0cl .- x0cl[1] ≈ x .- x[1]
end
