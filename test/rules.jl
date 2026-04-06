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

        # test Forward via directional derivative consistency
        V = randn(ComplexF64, size(X))
        dϕ_fwd, = autodiff(Forward, f, Duplicated, Const(p), Duplicated(copy(X), copy(V)))
        dϕ_rev = sum(real.(conj.(dX) .* V))
        @test dϕ_fwd ≈ dϕ_rev

        # Complex inplace transform with BatchDuplicated
        @testset "Complex inplace transform with BatchDuplicated" begin
            batch_size = 3
            
            dXs = Tuple(collect(randn(ComplexF64, 8, 8) for _ in 1:batch_size))
            dϕ_fwd = autodiff(Forward, f, BatchDuplicated, Const(p), BatchDuplicated(copy(X), deepcopy(dXs)))
            # Check consistency with directional derivatives
            for (i, V) in enumerate(dXs)
                dϕ_rev = sum(real.(conj.(dX) .* V))
                @test dϕ_fwd[1][i] ≈ dϕ_rev
            end
        end
    end


    @testset "R2R inplace transform" begin
        X = rand(Float64, 8, 8)
        p = FFTW.plan_r2r!(copy(X), FFTW.RODFT00; flags = FFTW.MEASURE)
        dX = zero(X)
        autodiff(Reverse, f, Const(p), Duplicated(copy(X), dX))
        dXfd, = grad(fdm, x -> f(p, copy(x)), copy(X)) # have to copy because of inplace
        @test dX ≈ dXfd

        # test Forward via directional derivative consistency
        V = randn(Float64, size(X))
        dϕ_fwd, = autodiff(Forward, f, Duplicated, Const(p), Duplicated(copy(X), copy(V)))
        dϕ_rev = sum(dX .* V)
        @test dϕ_fwd ≈ dϕ_rev

        @testset "R2R inplace transform with BatchDuplicated" begin
            batch_size = 3

            dXs = Tuple(collect(randn(Float64, 8, 8) for _ in 1:batch_size))
            dϕ_fwd = autodiff(Forward, f, BatchDuplicated, Const(p), BatchDuplicated(copy(X), deepcopy(dXs)))

            for (i, V) in enumerate(dXs)
                dϕ_rev = sum(dX .* V)
                @test dϕ_fwd[1][i] ≈ dϕ_rev
            end
        end
    end
end
