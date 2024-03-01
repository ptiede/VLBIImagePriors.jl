using FiniteDifferences
FiniteDifferences.to_vec(p::NamedTuple) = mapreduce(values, vcat, p)

@testset "NamedDist" begin
    d1 = NamedDist((a=Normal(), b = Uniform(), c = MvNormal(ones(2))))
    @test propertynames(d1) == (:a, :b, :c)
    @test d1.a == Normal()
    x1 = rand(d1)
    @test rand(d1, 2) isa Vector{<:NamedTuple}
    @test size(rand(d1, 2)) == (2,)
    rand(d1, 20, 21)
    @test logpdf(d1, x1) ≈ logpdf(d1.a, x1.a) + logpdf(d1.b, x1.b) + logpdf(d1.c, x1.c)

    dists = getfield(d1, :dists)
    xt = (b = 0.5, a = 1.0, c = [-0.5, 0.6])
    @test logpdf(d1, xt) ≈ logpdf(d1.a, xt.a) + logpdf(d1.b, xt.b) + logpdf(d1.c, xt.c)

    d2 = NamedDist(a=(Uniform(), Normal()), b = Beta(0.5, 0.5), c = [Uniform(), Uniform()], d = (a=Normal(), b = ImageUniform(2, 2)))
    @inferred logdensityof(d2, rand(d2))
    p0 = (a=(0.5, 0.5), b = 0.5, c = [0.25, 0.75], d = (a = 0.1, b = fill(0.1, 2, 2)))
    @test typeof(p0) == typeof(rand(d2))
    tf = asflat(d2)
    # tc = ascube(d2)
    @inferred TV.transform(tf, randn(dimension(tf)))
    # @inferred TV.transform(tc, rand(dimension(tc)))
    show(d1)
    show(d2)

    function foo(x)
        y, lj = TV.transform_and_logjac(tf, x)
        return logpdf(d2, y) + lj
    end

    x = 0.01*randn(dimension(tf))
    gz, = Zygote.gradient(foo, x)
    fdm = central_fdm(5, 1)
    gfd, = grad(fdm, foo, x)
    @test gz ≈ gfd

end

@testset "ComponentDist" begin
    dnt = NamedDist((a=Normal(), b = Uniform(), c = MvNormal(ones(2))))
    dcm = VLBIImagePriors.ComponentDist((a=Normal(), b = Uniform(), c = MvNormal(ones(2))))
    @test propertynames(dcm) == (:a, :b, :c)
    @test dcm.a == Normal()
    x1 = rand(dcm)
    @test rand(dcm) isa ComponentArray
    @test logpdf(dcm, x1) ≈ logpdf(dcm.a, x1.a) + logpdf(dcm.b, x1.b) + logpdf(dcm.c, x1.c)

    dists = getfield(dcm, :dists)
    xt = ComponentArray((b = 0.5, a = 1.0, c = [-0.5, 0.6]))
    @test logpdf(dcm, xt) ≈ logpdf(dcm.a, xt.a) + logpdf(dcm.b, xt.b) + logpdf(dcm.c, xt.c)
    @test logpdf(dcm, xt) ≈ logpdf(dnt, NamedTuple(xt))
    d2 = NamedDist(a=(Uniform(), Normal()), b = Beta(), c = [Uniform(), Uniform()], d = (a=Normal(), b = ImageUniform(2, 2)))
    @inferred logdensityof(d2, rand(d2))
    p0 = (a=(0.5, 0.5), b = 0.5, c = [0.25, 0.75], d = (a = 0.1, b = fill(0.1, 2, 2)))
    @test typeof(p0) == typeof(rand(d2))
    tf = asflat(d2)
    # tc = ascube(d2)
    @inferred TV.transform(tf, randn(dimension(tf)))
    # @inferred TV.transform(tc, rand(dimension(tc)))
    show(dcm)
    show(d2)

end
