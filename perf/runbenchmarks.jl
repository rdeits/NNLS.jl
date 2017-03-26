using NNLS
using BenchmarkTools
import NonNegLeastSquares

function runbenchmarks()
    srand(1)
    println("M\tN\ttime_ratio\tmemory_ratio")
    for m in 10:20:70
        for n_over_m in [0.5, 1, 2]
            n = round(Int, m * n_over_m)
            inputs = [(randn(m, n), randn(m)) for i in 1:100]

            for (A, b) in inputs
                @assert nnls(A, b) ≈ NonNegLeastSquares.nnls(A, b)
            end

            x1, t1, m1, gc1, mem1 = @timed begin
                for (A, b) in inputs
                    nnls(A, b)
                end
            end
            x2, t2, m2, gc2, mem2 = @timed begin
                for (A, b) in inputs
                    NonNegLeastSquares.nnls(A, b)
                end
            end
            @printf("%d\t%d\t%.1f\t%.1f\n", m, n, t2/t1, m2/m1)
            # b1 = median(@benchmark nnls(A, b) setup=(A = randn($m, $n); b = randn($m)))
            # b2 = median(@benchmark NonNegLeastSquares.nnls(A, b) setup=(A = randn($m, $n); b = randn($m)))
            # r = ratio(b2, b1)
            # @printf "%d\t%d\t%.1f\t%.1f\n" m n r.time r.memory
        end
    end
end

runbenchmarks()
