using NNLS
using BenchmarkTools
import NonNegLeastSquares

srand(1)
println("M\tN\ttime_ratio\tmemory_ratio")
for m in 10:10:50
    for n_over_m in [0.5, 1, 2]
        n = round(Int, m * n_over_m)
        b1 = median(@benchmark nnls(A, b) setup=(A = randn($m, $n); b = randn($m)))
        b2 = median(@benchmark NonNegLeastSquares.nnls(A, b) setup=(A = randn($m, $n); b = randn($m)))
        r = ratio(b2, b1)
        @printf "%d\t%d\t%.1f\t%.1f\n" m n r.time r.memory
        # print(m, "\t", n, "\t", round(Int, r.time), "\t", round(Int, r.memory))
    end
end
