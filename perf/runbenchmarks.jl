using NNLS
using BenchmarkTools
import NonNegLeastSquares
using PyCall

const pyopt_nnls = pyimport_conda("scipy.optimize", "scipy")[:nnls]

function runbenchmarks()
    srand(1)
    println("M\tN\tt_nnls\tt_nnls_with_workspace_reuse")
    for m in 50:50:250
        for n_over_m in [0.5, 1, 2]
            n = round(Int, m * n_over_m)
            inputs = [(randn(m, n), randn(m)) for i in 1:50]

            for (A, b) in inputs
                @assert nnls(A, b) ≈ NonNegLeastSquares.nnls(A, b)
                @assert nnls(A, b) == pyopt_nnls(A, b)[1]
            end

            t0 = @elapsed begin
                for (A, b) in inputs
                    pyopt_nnls(A, b)
                end
            end

            x1, t1, m1, gc1, mem1 = @timed begin
                for (A, b) in inputs
                    nnls(A, b)
                end
            end
            x2, t2, m2, gc2, mem2 = @timed begin
                work = NNLSWorkspace(m, n)
                for (A, b) in inputs
                    nnls!(work, A, b)
                end
            end
            # @show (t0, t1, t2)
            @printf("%d\t%d\t%.2f\t%.2f\n", m, n, t1 / t0, t2 / t0)
            # b1 = median(@benchmark nnls(A, b) setup=(A = randn($m, $n); b = randn($m)))
            # b2 = median(@benchmark NonNegLeastSquares.nnls(A, b) setup=(A = randn($m, $n); b = randn($m)))
            # r = ratio(b2, b1)
            # @printf "%d\t%d\t%.1f\t%.1f\n" m n r.time r.memory
        end
    end
end

runbenchmarks()
