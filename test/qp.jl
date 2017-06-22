
function rand_qp_data(n, q)
    Q = randn(n, n)
    Q = Q * Q'
    c = randn(n)
    G = randn(q, n)
    g = randn(q)
    Q, c, G, g
end

function rand_infeasible_qp_data(n, q)
    Q, c, G, g = rand_qp_data(n, q - 1)
    index = rand(1 : q - 1)
    Gi = G[index, :]'
    G = [G; -Gi]
    g = [g; -g[index] - 100]
    Q, c, G, g
end

# More straightforward, less efficient implementation of the paper
function quadprog_bemporad_simple(Q, c, G, g)
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

# Solve quadratic program with SCS (use trick from http://www.seas.ucla.edu/~vandenbe/publications/socp.pdf)
function quadprog_scs(Q, c, G, g)
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
function qp_jump(Q, c, G, g)
    m = Model(solver=NNLS.QPSolver())
    @variable m z[1:n]
    @constraint m G * z .<= g
    @static if VERSION < "v0.6-"
        @objective m Min 0.5 * (z' * Q * z)[1] + c' * z
    else
        @objective m Min 0.5 * (z' * Q * z) + c' * z
    end
    status = solve(m, suppress_warnings = true)
    z = status == :Optimal ? getvalue(z) : fill(NaN, n)
    status, z
end


function qp_test(work, Q, c, G, g)
    status_scs, z_scs, λ_scs = quadprog_scs(Q, c, G, g)
    status_basic, z_basic = quadprog_bemporad_simple(Q, c, G, g)
    status_nnlsqp, z_nnlsqp = qp_jump(Q, c, G, g)
    load!(work, Q, c, G, g)
    z, λ = solve!(work)

    norminf = x -> norm(x, Inf)
    @test status_scs == status_basic
    @test status_scs == work.status
    @test status_scs == status_basic

    if work.status == :Optimal
        @test isapprox(z_basic, z; norm = norminf, atol = 1e-2)
        @test isapprox(z_scs, z; norm = norminf, atol = 5e-2)
        @test isapprox(λ_scs, λ; norm = norminf, atol = 5e-2)
        @test isapprox(z_scs, z_nnlsqp; norm = norminf, atol = 5e-2)
    end
end

@testset "qp" begin
    srand(1)
    n, q = 100, 50
    work = QPWorkspace{Float64, Int}(n, q)
    for i = 1 : 100
        Q, c, G, g = rand_qp_data(n, q)
        qp_test(work, Q, c, G, g)
    end
    for j = 1 : 100
        Q, c, G, g = rand_infeasible_qp_data(n, q)
        qp_test(work, Q, c, G, g)
    end
end
