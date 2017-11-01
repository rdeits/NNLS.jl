
function rand_qp_data(n, q)
    Q = randn(n, n)
    Q = Q * Q'
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
    L = cholfact(Q, :U)
    M = G / L[:U]
    e = (L \ c)
    d = g + G * e
    γ = 1
    A = [-M'; -d']
    b = [zeros(n); γ]
    y = nnls(A, b)
    r = A * y - b
    status = sum(abs, r) < 1e-7 ? :Infeasible : :Optimal
    z = - inv(Q) * (c + 1 / (γ + d ⋅ y) * G' * y)

    @assert isapprox(L[:U]' * L[:U], Q; rtol = 1e-4)
    @assert isapprox(G * inv(L[:U]), M; rtol = 1e-4)
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
    P = sqrtm(Q)
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


function qp_test(work, qp::QP)
    status_scs, z_scs, λ_scs = quadprog_ecos(qp)
    status_basic, z_basic = quadprog_bemporad_simple(qp)
    status_nnlsqp, z_nnlsqp, λ_nnlsqp = qp_jump(qp)
    load!(work, qp)
    z, λ = solve!(work)

    norminf = x -> norm(x, Inf)
    @test status_scs == status_basic
    @test status_scs == work.status
    @test status_scs == status_basic

    if work.status == :Optimal
        @test check_optimality_conditions(qp, z, λ) <= 1e-5
        @test isapprox(z_basic, z; norm = norminf, atol = 1e-2)
        @test isapprox(z_scs, z; norm = norminf, atol = 5e-2)
        @test isapprox(λ_scs, λ; norm = norminf, atol = 5e-2)
        @test isapprox(z_scs, z_nnlsqp; norm = norminf, atol = 5e-2)
        @test isapprox(λ_scs, λ_nnlsqp; norm = norminf, atol = 5e-2)
    end
end

@testset "qp" begin
    srand(1)
    n, q = 100, 50
    work = QPWorkspace(q, n)
    for i = 1 : 100
        qp = rand_qp_data(n, q)
        qp_test(work, qp)
    end
    for j = 1 : 100
        qp = rand_infeasible_qp_data(n, q)
        qp_test(work, qp)
    end
    qp = rand_qp_data(n, q)
    QPWorkspace(qp)
end

@testset "qp precision" begin
    srand(2)
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
end


