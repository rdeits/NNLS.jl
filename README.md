# NNLS.jl: Non-Negative Least Squares in Julia

[![Build Status](https://travis-ci.org/rdeits/NNLS.jl.svg?branch=master)](https://travis-ci.org/rdeits/NNLS.jl)
[![codecov.io](http://codecov.io/github/rdeits/NNLS.jl/coverage.svg?branch=master)](http://codecov.io/github/rdeits/NNLS.jl?branch=master)

This package implements the non-negative least squares solver from Lawson and Hanson [1]. Given a matrix A and vector b, `nnls(A, b)` computes:

    min. || Ax - b ||_2
       x

     s.t. x[i] >= 0 forall i

The code contained here is a direct port of the original Fortran code to Julia.

# Usage

```julia
A = randn(100, 200)
b = randn(100)
x = nnls(A, b)
```

# Reducing memory allocation

The NNLS implementation (and the Fortran code on which it is based) have been implemented to allocate as little memory as possible. If you want direct control over the memory usage, you can pre-allocate an `NNLSWorkspace` which will hold all data used in the NNLS algorithm:

```julia
A = randn(100, 200)
b = randn(100)

work = NNLSWorkspace(A, b)
solve!(work)
@show work.x
```

The call to `solve!(work)` should allocate no memory. You can re-use the same workspace multiple times:

```julia
A2 = randn(100, 200)
b2 = randn(100)

load!(work, A2, b2)
solve!(work)
@show work.x
```

If `A2` and `b2` match the size of the arrays `A` and `b` used to create the workspace, then `load!(work, A2, b2)` will not allocate. If they do not match, then the workspace will be resized and some memory will be allocated.

# Solving Quadratic Programs

The NNLS approach can also be used to solve Quadratic Programs, using the approach from section II of  Bemporad, *A quadratic programming algorithm based on nonnegative least squares with applications to embedded model predictive control*, IEEE Transactions on Automatic Control, 2016.

The problem must be of the form:

    Minimize 1/2 z' Q z + c' z
    Subject to G z <= g

The `QP` struct holds all of the relevant matrices:

```julia
qp = QP(Q, c, G, g)
```

and a `QPWorkspace` allocates all of the scratch workspace necessary to solve the QP:

```julia
work = QPWorkspace(qp)
```

Solving a QP returns the primal solution `z` and dual solution `\lambda`:

```julia
z, λ = solve!(work)
```

You can check the solution status by looking at `work.status`:

```julia
@assert work.status == :Optimal
```

The function `check_optimality_conditions` checks violation of the KKT optimality conditions for a given problem and solution. It should return a value close to zero for a feasible & optimal solution:

```julia
@assert check_optimality_conditions(qp, z, λ) <= 1e-6
```

# References

[1] Lawson, C.L. and R.J. Hanson, Solving Least-Squares Problems, Prentice-Hall, Chapter 23, p. 161, 1974