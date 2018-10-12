using NNLS
using Test
# import NonNegLeastSquares
using PyCall
using ECOS
using JuMP
using Random
using LinearAlgebra

import Libdl


const pyopt = pyimport_conda("scipy.optimize", "scipy")

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