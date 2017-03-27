# NNLS.jl: Non-Negative Least Squares in Julia

[![Build Status](https://travis-ci.org/rdeits/NNLS.jl.svg?branch=master)](https://travis-ci.org/rdeits/NNLS.jl)

[![codecov.io](http://codecov.io/github/rdeits/NNLS.jl/coverage.svg?branch=master)](http://codecov.io/github/rdeits/NNLS.jl?branch=master)

This package implements the non-negative least squares solver from Lawson and Hanson [1]. Given a matrix A and vector b, `nnls(A, b)` computes:

    min. || Ax - b ||_2
       x

     s.t. x[i] >= 0 forall i

The code contained here is a direct port of the original Fortran code to Julia.

# References

[1] Lawson, C.L. and R.J. Hanson, Solving Least-Squares Problems, Prentice-Hall, Chapter 23, p. 161, 1974