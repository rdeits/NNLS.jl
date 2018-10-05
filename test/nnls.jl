@testset "bigfloat" begin
    srand(5)
    for i in 1:100
        m = rand(1:10)
        n = rand(1:10)
        A = randn(m, n)
        b = randn(m)
        x1 = nnls(A, b)
        x2 = nnls(BigFloat.(A), BigFloat.(b))
        @test x1 ≈ x2
    end
end

function h1_reference!(u::DenseVector)
    mode = 1
    lpivot = 1
    l1 = 2
    m = length(u)
    iue = 1
    up = Ref{Cdouble}()
    c = Vector{Cdouble}()
    ice = 1
    icv = 1
    ncv = 0
    ccall((:h12_, libnnls), Void,
        (Ref{Cint}, Ref{Cint}, Ref{Cint}, Ref{Cint},
         Ref{Cdouble}, Ref{Cint}, Ref{Cdouble},
         Ref{Cdouble}, Ref{Cint}, Ref{Cint}, Ref{Cint}),
        mode, lpivot, l1, m,
        u, iue, up,
        c, ice, icv, ncv)
    return up[]
end

function h2_reference!{T}(u::DenseVector{T}, up::T, c::DenseVector{T})
    mode = 2
    lpivot = 1
    l1 = 2
    m = length(u)
    @assert length(c) == m
    iue = 1
    ice = 1
    icv = m
    ncv = 1
    ccall((:h12_, libnnls), Void,
        (Ref{Cint}, Ref{Cint}, Ref{Cint}, Ref{Cint},
         Ref{Cdouble}, Ref{Cint}, Ref{Cdouble},
         Ref{Cdouble}, Ref{Cint}, Ref{Cint}, Ref{Cint}),
        mode, lpivot, l1, m,
        u, iue, up,
        c, ice, icv, ncv)
end

function g1_reference(a, b)
    c = Ref{Float64}()
    s = Ref{Float64}()
    sig = Ref{Float64}()
    ccall((:g1_, libnnls), Void,
        (Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}),
        a, b, c, s, sig)
    return c[], s[], sig[]
end

function nnls_reference!(work::NNLSWorkspace{Cdouble, Cint})
    A = work.QA
    b = work.Qb
    m, n = size(A)
    @assert length(work.x) == n
    @assert length(work.w) == n
    mda = m
    mode = Ref{Cint}()
    rnorm = Ref{Cdouble}()
    ccall((:nnls_, libnnls), Void,
          (Ref{Cdouble}, Ref{Cint}, Ref{Cint}, Ref{Cint}, # A, mda, m, n
           Ref{Cdouble}, # b
           Ref{Cdouble}, # x
           Ref{Cdouble}, # rnorm
           Ref{Cdouble}, # w
           Ref{Cdouble}, # zz
           Ref{Cint},    # idx
           Ref{Cint}),    # mode
          A, mda, m, n,
          b,
          work.x,
          rnorm,
          work.w,
          work.zz,
          work.idx,
          mode)
    work.rnorm = rnorm[]
    work.mode = mode[]
    if work.mode[] == 2
        error("nnls.f exited with dimension error")
    end
end

@testset "construct_householder!" begin
    srand(1)
    for i in 1:100000
        u = randn(rand(3:10))

        u1 = copy(u)
        up1 = NNLS.construct_householder!(u1, 0.0)

        u2 = copy(u)
        up2 = h1_reference!(u2)
        @test up1 == up2
        @test u1 == u2
    end
end

@testset "apply_householder!" begin
    srand(2)
    for i in 1:10000
        u = randn(rand(3:10))
        c = randn(length(u))

        u1 = copy(u)
        c1 = copy(c)
        up1 = NNLS.construct_householder!(u1, 0.0)
        NNLS.apply_householder!(u1, up1, c1)

        u2 = copy(u)
        c2 = copy(c)
        up2 = h1_reference!(u2)
        h2_reference!(u2, up2, c2)

        @test up1 == up2
        @test u1 == u2
        @test c1 == c2

        u3 = copy(u)
        c3 = copy(c)
        @test @wrappedallocs(NNLS.construct_householder!(u3, 0.0)) == 0
        up3 = up1
        @test @wrappedallocs(NNLS.apply_householder!(u3, up3, c3)) == 0
    end
end

@testset "orthogonal_rotmat" begin
    srand(3)
    for i in 1:1000
        a = randn()
        b = randn()
        c, s, sig = NNLS.orthogonal_rotmat(a, b)
        @test [c s; -s c] * [a, b] ≈ [sig, 0]
        @test NNLS.orthogonal_rotmat(a, b) == g1_reference(a, b)
        @test @wrappedallocs(NNLS.orthogonal_rotmat(a, b)) == 0
    end
end

@testset "nnls vs fortran reference" begin
    srand(4)
    for i in 1:5000
        m = rand(20:100)
        n = rand(20:100)
        A = randn(m, n)
        b = randn(m)

        work1 = NNLSWorkspace(A, b)
        solve!(work1)

        work2 = NNLSWorkspace(A, b, Cint)
        nnls_reference!(work2)

        @test work1.x == work2.x
        @test work1.QA == work2.QA
        @test work1.Qb == work2.Qb
        @test work1.w == work2.w
        @test work1.zz == work2.zz
        @test work1.idx == work2.idx
        @test work1.rnorm == work2.rnorm
        @test work1.mode == work2.mode
    end
end

@testset "nnls allocations" begin
    srand(101)
    for i in 1:50
        m = rand(20:100)
        n = rand(20:100)
        A = randn(m, n)
        b = randn(m)
        work = NNLSWorkspace(A, b)
        @test @wrappedallocs(solve!(work)) == 0
    end
end

@testset "nnls workspace reuse" begin
    srand(200)
    m = 10
    n = 20
    work = NNLSWorkspace(m, n)
    solve!(work, randn(m, n), randn(m))
    for i in 1:100
        A = randn(m, n)
        b = randn(m)
        @test @wrappedallocs(solve!(work, A, b)) == 0
        @test work.x == pyopt[:nnls](A, b)[1]
    end

    m = 20
    n = 10
    for i in 1:100
        A = randn(m, n)
        b = randn(m)
        solve!(work, A, b)
        @test work.x == pyopt[:nnls](A, b)[1]
    end
end

@testset "non-Int Integer workspace" begin
    m = 10
    n = 20
    A = randn(m, n)
    b = randn(m)
    work = NNLSWorkspace(A, b, Int32)
    # Compile
    solve!(work)

    A = randn(m, n)
    b = randn(m)
    work = NNLSWorkspace(A, b, Int32)
    @test @wrappedallocs(solve!(work)) <= 0
end

# @testset "nnls vs NonNegLeastSquares" begin
#     srand(5)
#     for i in 1:1000
#         m = rand(20:60)
#         n = rand(20:60)
#         A = randn(m, n)
#         b = randn(m)

#         @test nnls(A, b) ≈ NonNegLeastSquares.nnls(A, b)
#     end
# end

@testset "nnls vs scipy" begin
    srand(5)
    for i in 1:10000
        m = rand(1:60)
        n = rand(1:60)
        A = randn(m, n)
        b = randn(m)
        x1 = nnls(A, b)
        x2, residual2 = pyopt[:nnls](A, b)
        @test x1 == x2
    end
end
