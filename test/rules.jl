using EnzymeTestUtils
using Enzyme
using FFTW
using FiniteDifferences

@testset "FFTW Rules" begin
    # Unfortunately EnzymeTestUtils does not support inplace matrix derivatives
    # so we will do some manual testing here
    function f(p, X)
        FFTW.unsafe_execute!(p, X, X)
        return sum(abs2, X)
    end

    fdm = central_fdm(5, 1)

    @testset "Complex inplace transform" begin
        X = rand(ComplexF64, 8, 8)
        p = plan_fft!(copy(X); flags = FFTW.MEASURE)
        dX = zero(X)
        autodiff(Reverse, f, Const(p), Duplicated(copy(X), dX))
        dXfd, = grad(fdm, x -> f(p, copy(x)), copy(X)) # have to copy because of inplace
        @test dX ≈ dXfd
    end


    @testset "R2R inplace transform" begin
        X = rand(Float64, 8, 8)
        p = FFTW.plan_r2r!(copy(X), FFTW.RODFT00; flags = FFTW.MEASURE)
        dX = zero(X)
        autodiff(Reverse, f, Const(p), Duplicated(copy(X), dX))
        dXfd, = grad(fdm, x -> f(p, copy(x)), copy(X)) # have to copy because of inplace
        @test dX ≈ dXfd
    end
end
