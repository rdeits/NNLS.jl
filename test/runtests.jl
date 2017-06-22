using NNLS
using Base.Test
import NonNegLeastSquares
using PyCall
using ECOS
using JuMP

const pyopt = pyimport_conda("scipy.optimize", "scipy")

const libnnls = joinpath(dirname(@__FILE__), "libnnls")
libnnls_path = libnnls * "." * Libdl.dlext
run(`gfortran -shared -fPIC -o $libnnls_path nnls.f`)
@test isfile(libnnls_path)

# Allocation measurement doesn't work reliably on Julia v0.5 when
# code coverage checking is enabled.
const test_allocs = VERSION >= v"0.6-" || Base.JLOptions().code_coverage == 0

macro wrappedallocs(expr)
    argnames = [gensym() for a in expr.args]
    quote
        function g($(argnames...))
            @allocated $(Expr(expr.head, argnames...))
        end
        $(Expr(:call, :g, [esc(a) for a in expr.args]...))
    end
end

include("nnls.jl")
include("qp.jl")