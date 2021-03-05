using FixedPointNumbers: Fixed

function rand_qp_data(n, q)
    Q = randn(n, n)
    Q = Q * Q' .+ 1e-6 .* I
    c = randn(n)
    G = randn(q, n)
    g = randn(q)
    QP(Q, c, G, g)
end

function rand_infeasible_qp_data(n, q)
    qp = rand_qp_data(n, q - 1)
    Q, c, G, g = qp.Q, qp.c, qp.G, qp.g
    index = rand(1 : q - 1)
    Gi = G[index, :]'
    G = [G; -Gi]
    g = [g; -g[index] - 100]
    QP(Q, c, G, g)
end

# More straightforward, less efficient implementation of the paper
function quadprog_bemporad_simple(qp::QP)
    Q, c, G, g = qp.Q, qp.c, qp.G, qp.g
    n = size(Q, 1)
    T = Float64
    L = cholesky(Hermitian(Q, :U))
    M = G / L.U
    e = (L \ c)
    d = g + G * e
    γ = 1
    A = [-M'; -d']
    b = [zeros(n); γ]
    y = nnls(A, b)
    r = A * y - b
    status = sum(abs, r) < 1e-7 ? :Infeasible : :Optimal
    z = - inv(Q) * (c + 1 / (γ + d ⋅ y) * G' * y)

    @assert isapprox(L.U' * L.U, Q; rtol = 1e-4)
    @assert isapprox(G * inv(L.U), M; rtol = 1e-4)
    @assert isapprox(g + G * (Q \ c), d; rtol = 1e-4)

    status, z
end

# Solve quadratic program with ECOS (use trick from http://www.seas.ucla.edu/~vandenbe/publications/socp.pdf)
function quadprog_ecos(qp::QP)
    Q, c, G, g = qp.Q, qp.c, qp.G, qp.g
    n = length(c)
    q = length(g)
    m = Model(solver = ECOSSolver(verbose = false))
    @variable m z[1 : n]
    constr = @constraint m G * z .<= g
    @variable m slack >= 0
    P = sqrt(Q)
    @constraint m norm(P * z + P \ c) <= slack
    @objective m Min slack
    status = solve(m, suppress_warnings = true)
    z = status == :Optimal ? getvalue(z) : fill(NaN, n)
    λ = status == :Optimal ? getdual(constr) : fill(NaN, q)
    status, z, λ
end

# Solve a quadratic program using the NNLS solver from JuMP
function qp_jump(qp::QP)
    Q, c, G, g = qp.Q, qp.c, qp.G, qp.g
    n = length(c)
    q = length(g)
    m = Model(solver=NNLS.NNLSSolver())
    @variable m z[1:n]
    constr = @constraint m G * z .<= g
    @static if VERSION < v"0.6-"
        @objective m Min 0.5 * (z' * Q * z)[1] + (c' * z)[1]
    else
        @objective m Min 0.5 * (z' * Q * z) + c' * z
    end
    status = solve(m, suppress_warnings = true)
    z = status == :Optimal ? getvalue(z) : fill(NaN, n)
    λ = status == :Optimal ? getdual(constr) : fill(NaN, q)
    status, z, λ
end


function qp_test(qp::QP)
    status_scs, z_scs, λ_scs = quadprog_ecos(qp)
    status_basic, z_basic = quadprog_bemporad_simple(qp)
    status_nnlsqp, z_nnlsqp, λ_nnlsqp = qp_jump(qp)

    work = QPWorkspace(qp)
    z, λ = solve!(work)

    work_big = QPWorkspace(QP{BigFloat}(qp))
    z_big, λ_big = solve!(work_big)

    norminf = x -> norm(x, Inf)
    @test status_scs == status_basic
    @test status_scs == work.status
    @test status_scs == status_nnlsqp
    @test status_scs == work_big.status

    if work.status == :Optimal
        @test check_optimality_conditions(qp, z, λ) <= 2e-5
        @test isapprox(z_basic, z; norm = norminf, atol = 1e-2)
        @test isapprox(z_scs, z; norm = norminf, atol = 5e-2)
        @test isapprox(λ_scs, λ; norm = norminf, atol = 5e-2)
        @test isapprox(z_scs, z_nnlsqp; norm = norminf, atol = 5e-2)
        @test isapprox(λ_scs, λ_nnlsqp; norm = norminf, atol = 5e-2)
        @test isapprox(z_scs, z_big; norm = norminf, atol = 5e-2)
        @test isapprox(λ_scs, λ_big; norm = norminf, atol = 5e-2)
    end
end

@testset "qp" begin
    Random.seed!(1)
    n, q = 100, 50
    for i = 1 : 100
        qp = rand_qp_data(n, q)
        qp_test(qp)
    end
    for j = 1 : 100
        qp = rand_infeasible_qp_data(n, q)
        qp_test(qp)
    end
    qp = rand_qp_data(n, q)
    QPWorkspace(qp)
end

@testset "qp precision" begin
    Random.seed!(2)
    n, q = 10, 5

    # Generate a random QP with the default scalar type of Float64 (equivalent to C double)
    qp_float64 = rand_qp_data(n, q)
    @test typeof(qp_float64) == QP{Float64}

    # Solve the high-precision QP and check its solution:
    z_float64, λ_float64 = solve!(QPWorkspace(qp_float64))
    @test check_optimality_conditions(qp_float64, z_float64, λ_float64) <= 1e-12

    # Generate a copy of the QP using a lower precision scalar type:
    qp_float32 = convert(QP{Float32}, qp_float64)

    # Solve the low-precision QP and check its solution:
    z_float32, λ_float32 = solve!(QPWorkspace(qp_float32))
    @test check_optimality_conditions(qp_float32, z_float32, λ_float32) <= 1e-4

    # Test whether the low-precision QP solution also satisfies the optimality
    # conditions of the high-precision QP:
    @test check_optimality_conditions(qp_float64, z_float32, λ_float32) <= 1e-4

    # Also solve the QP at very high precision, just for fun. This uses
    # Julia's built-in BigFloat:
    # https://docs.julialang.org/en/stable/stdlib/numbers/#BigFloats-1
    setprecision(BigFloat, 1000) do
        qp_big = convert(QP{BigFloat}, qp_float64)
        z_big, λ_big = solve!(QPWorkspace(qp_big))
        @test check_optimality_conditions(qp_big, z_big, λ_big) <= 1e-299
    end

    # Solve a QP using fixed point numbers from the FixedPointNumbers.jl package:
    # https://github.com/JuliaMath/FixedPointNumbers.jl
    F = Fixed{Int64, 32} # fixed-point number using 64 total bits, with 32 bits used for the fraction
    qp_fixed_64 = convert(QP{F}, qp_float64)
    z_fixed_64, λ_fixed_64 = solve!(QPWorkspace(qp_fixed_64))
    @test check_optimality_conditions(qp_fixed_64, z_fixed_64, λ_fixed_64) <= 1e-8
end


