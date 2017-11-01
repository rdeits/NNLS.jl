__precompile__()

module NNLS
using Compat: @compat

export nnls,
       solve!,
       NNLSWorkspace,
       QPWorkspace,
       QP,
       primal_infeasibility,
       dual_infeasibility,
       stationarity_violation,
       slackness_violation,
       check_solution,
       load!,
       NNLSSolver

"""
CONSTRUCTION AND/OR APPLICATION OF A SINGLE
HOUSEHOLDER TRANSFORMATION..     Q = I + U*(U**T)/B

The original version of this code was developed by
Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory
1973 JUN 12, and published in the book
"SOLVING LEAST SQUARES PROBLEMS", Prentice-HalL, 1974.
Revised FEB 1995 to accompany reprinting of the book by SIAM.
"""
function construct_householder!{T}(u::AbstractVector{T}, up::T)::T
    m = length(u)
    if m <= 1
        return up
    end

    cl = maximum(abs, u)
    @assert cl > 0
    clinv = 1 / cl
    sm = zero(T)
    for ui in u
        sm += (ui * clinv)^2
    end
    cl *= sqrt(sm)
    if u[1] > 0
        cl = -cl
    end
    result = u[1] - cl
    u[1] = cl

    return result
end

"""
CONSTRUCTION AND/OR APPLICATION OF A SINGLE
HOUSEHOLDER TRANSFORMATION..     Q = I + U*(U**T)/B

The original version of this code was developed by
Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory
1973 JUN 12, and published in the book
"SOLVING LEAST SQUARES PROBLEMS", Prentice-HalL, 1974.
Revised FEB 1995 to accompany reprinting of the book by SIAM.
"""
function apply_householder!{T}(u::AbstractVector{T}, up::T, c::AbstractVector{T})
    m = length(u)
    if m > 1
        cl = abs(u[1])
        @assert cl > 0
        b = up * u[1]
        if b >= 0
            return
        end
        b = 1 / b

        sm = c[1] * up
        for i in 2:m
            sm += c[i] * u[i]
        end
        if sm != 0
            sm *= b
            c[1] += sm * up
            for i in 2:m
                c[i] += sm * u[i]
            end
        end
    end
end

"""
   COMPUTE ORTHOGONAL ROTATION MATRIX..
The original version of this code was developed by
Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory
1973 JUN 12, and published in the book
"SOLVING LEAST SQUARES PROBLEMS", Prentice-HalL, 1974.
Revised FEB 1995 to accompany reprinting of the book by SIAM.

   COMPUTE.. MATRIX   (C, S) SO THAT (C, S)(A) = (SQRT(A**2+B**2))
                      (-S,C)         (-S,C)(B)   (   0          )
   COMPUTE SIG = SQRT(A**2+B**2)
      SIG IS COMPUTED LAST TO ALLOW FOR THE POSSIBILITY THAT
      SIG MAY BE IN THE SAME LOCATION AS A OR B .
"""
function orthogonal_rotmat{T}(a::T, b::T)::Tuple{T, T, T}
    if abs(a) > abs(b)
        xr = b / a
        yr = sqrt(1 + xr^2)
        c = (1 / yr) * sign(a)
        s = c * xr
        sig = abs(a) * yr
    elseif b != 0
        xr = a / b
        yr = sqrt(1 + xr^2)
        s = (1 / yr) * sign(b)
        c = s * xr
        sig = abs(b) * yr
    else
        sig = zero(T)
        c = zero(T)
        s = one(T)
    end
    return c, s, sig
end

"""
The original version of this code was developed by
Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory
1973 JUN 15, and published in the book
"SOLVING LEAST SQUARES PROBLEMS", Prentice-HalL, 1974.
Revised FEB 1995 to accompany reprinting of the book by SIAM.
"""
function solve_triangular_system!(zz, A, idx, nsetp, jj)
    for l in 1:nsetp
        ip = nsetp + 1 - l
        if (l != 1)
            for ii in 1:ip
                zz[ii] -= A[ii, jj] * zz[ip + 1]
            end
        end
        jj = idx[ip]
        zz[ip] /= A[ip, jj]
    end
    return jj
end

type NNLSWorkspace{T, I <: Integer}
    QA::Matrix{T}
    Qb::Vector{T}
    x::Vector{T}
    w::Vector{T}
    zz::Vector{T}
    idx::Vector{I}
    rnorm::T
    mode::I
    nsetp::I

    function (::Type{NNLSWorkspace{T, I}}){T, I <: Integer}(m, n)
        new{T, I}(Matrix{T}(m, n), # A
            Vector{T}(m),    # b
            Vector{T}(n),    # x
            Vector{T}(n),    # w
            Vector{T}(m),    # zz
            Vector{I}(n),    # idx
            zero(T), # rnorm
            zero(I), # mode
            zero(I)  # nsetp
        )
    end
end

function Base.resize!{T}(work::NNLSWorkspace{T}, m::Integer, n::Integer)
    work.QA = Matrix{T}(m, n)
    work.Qb = Vector{T}(m)
    resize!(work.x, n)
    resize!(work.w, n)
    resize!(work.zz, m)
    resize!(work.idx, n)
end

function load!{T}(work::NNLSWorkspace{T}, A::AbstractMatrix{T}, b::AbstractVector{T})
    m, n = size(A)
    @assert size(b) == (m,)
    if size(work.QA, 1) != m || size(work.QA, 2) != n
        resize!(work, m, n)
    end
    work.QA .= A
    work.Qb .= b
    work
end

NNLSWorkspace{T, I}(m::Integer, n::Integer,
                    eltype::Type{T}=Float64,
                    indextype::Type{I}=Int) = NNLSWorkspace{T, I}(m, n)

function NNLSWorkspace{T, I}(A::Matrix{T}, b::Vector{T}, indextype::Type{I}=Int)
    m, n = size(A)
    @assert size(b) == (m,)
    work = NNLSWorkspace{T, I}(m, n)
    load!(work, A, b)
    work
end


"""
Views in Julia still allocate some memory (since they need to keep
a reference to the original array). This type allocates no memory
and does no bounds checking. Use it with caution.
"""
immutable UnsafeVectorView{T} <: AbstractVector{T}
    offset::Int
    len::Int
    ptr::Ptr{T}
end

UnsafeVectorView{T}(parent::DenseArray{T}, start_ind::Integer, len::Integer) = UnsafeVectorView{T}(start_ind - 1, len, pointer(parent))
Base.size(v::UnsafeVectorView) = (v.len,)
Base.getindex(v::UnsafeVectorView, idx) = unsafe_load(v.ptr, idx + v.offset)
Base.setindex!(v::UnsafeVectorView, value, idx) = unsafe_store!(v.ptr, value, idx + v.offset)
Base.length(v::UnsafeVectorView) = v.len
@static if VERSION >= v"0.6-"
    Base.IndexStyle{V <: UnsafeVectorView}(::Type{V}) = Base.IndexLinear()
else
    Base.linearindexing{V <: UnsafeVectorView}(::Type{V}) = Base.LinearFast()
end

"""
UnsafeVectorView only works for isbits types. For other types, we're already
allocating lots of memory elsewhere, so creating a new View is fine.

This function looks type-unstable, but the isbits(T) test can be evaluated
by the compiler, so the result is actually type-stable.
"""
function fastview{T}(parent::Array{T}, start_ind::Integer, len::Integer)
    if isbits(T)
        UnsafeVectorView(parent, start_ind, len)
    else
        @view(parent[start_ind:(start_ind + len - 1)])
    end
end

"""
Fallback for non-contiguous arrays, for which UnsafeVectorView does not make
sense.
"""
fastview(parent::AbstractArray, start_ind::Integer, len::Integer) = @view(parent[start_ind:(start_ind + len - 1)])

@noinline function checkargs(work::NNLSWorkspace)
    m, n = size(work.QA)
    @assert size(work.Qb) == (m,)
    @assert size(work.x) == (n,)
    @assert size(work.w) == (n,)
    @assert size(work.zz) == (m,)
    @assert size(work.idx) == (n,)
end

function largest_positive_dual{T, TI}(w::AbstractVector{T}, idx::AbstractVector{TI}, range)
    wmax = zero(T)
    izmax = zero(TI)
    for i in range
        j = idx[i]
        if w[j] > wmax
            wmax = w[j]
            izmax = i
        end
    end
    wmax, izmax
end


"""
Algorithm NNLS: NONNEGATIVE LEAST SQUARES

The original version of this code was developed by
Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory
1973 JUN 15, and published in the book
"SOLVING LEAST SQUARES PROBLEMS", Prentice-HalL, 1974.
Revised FEB 1995 to accompany reprinting of the book by SIAM.

GIVEN AN M BY N MATRIX, A, AND AN M-VECTOR, B,  COMPUTE AN
N-VECTOR, X, THAT SOLVES THE LEAST SQUARES PROBLEM
                 A * X = B  SUBJECT TO X .GE. 0
"""
function solve!{T, TI}(work::NNLSWorkspace{T, TI}, max_iter::Integer=(3 * size(work.QA, 2)))
    checkargs(work)

    A = work.QA
    b = work.Qb
    x = work.x
    w = work.w
    zz = work.zz
    idx = work.idx
    const factor = 0.01
    work.mode = 1

    m = convert(TI, size(A, 1))
    n = convert(TI, size(A, 2))

    iter = 0
    x .= 0
    idx .= 1:n

    iz2 = n
    iz1 = one(TI)
    iz = zero(TI)
    j = zero(TI)
    jj = zero(TI)
    nsetp = zero(TI)
    up = zero(T)

    terminated = false

    # ******  MAIN LOOP BEGINS HERE  ******
    while true
        # println("jl main loop")
        # QUIT IF ALL COEFFICIENTS ARE ALREADY IN THE SOLUTION.
        # OR IF M COLS OF A HAVE BEEN TRIANGULARIZED.
        if (iz1 > iz2 || nsetp >= m)
            terminated = true
            break
        end

        # COMPUTE COMPONENTS OF THE DUAL (NEGATIVE GRADIENT) VECTOR W().
        for i in iz1:iz2
            idxi = idx[i]
            sm = zero(T)
            for l in (nsetp + 1):m
                sm += A[l, idxi] * b[l]
            end
            w[idxi] = sm
        end

        while true
            # FIND LARGEST POSITIVE W(J).
            wmax, izmax = largest_positive_dual(w, idx, iz1:iz2)

            # IF WMAX .LE. 0. GO TO TERMINATION.
            # THIS INDICATES SATISFACTION OF THE KUHN-TUCKER CONDITIONS.
            if wmax <= 0
                terminated = true
                break
            end

            iz = izmax
            j = idx[iz]

            # THE SIGN OF W(J) IS OK FOR J TO BE MOVED TO SET P.
            # BEGIN THE TRANSFORMATION AND CHECK NEW DIAGONAL ELEMENT TO AVOID
            # NEAR LINEAR DEPENDENCE.
            Asave = A[nsetp + 1, j]
            up = construct_householder!(
                fastview(A, sub2ind(A, nsetp + 1, j), m - nsetp),
                up)
            unorm::T = zero(T)
            for l in 1:nsetp
                unorm += A[l, j]^2
            end
            unorm = sqrt(unorm)

            if ((unorm + abs(A[nsetp + 1, j]) * factor) - unorm) > 0
                # COL J IS SUFFICIENTLY INDEPENDENT.  COPY B INTO ZZ, UPDATE ZZ
                # AND SOLVE FOR ZTEST ( = PROPOSED NEW VALUE FOR X(J) ).
                # println("copying b into zz")
                zz .= b
                apply_householder!(
                    fastview(A, sub2ind(A, nsetp + 1, j), m - nsetp),
                    up,
                    fastview(zz, nsetp + 1, m - nsetp))
                ztest = zz[nsetp + 1] / A[nsetp + 1, j]

                # SEE IF ZTEST IS POSITIVE
                if ztest > 0
                    break
                end
            end

            # REJECT J AS A CANDIDATE TO BE MOVED FROM SET Z TO SET P.
            # RESTORE A(NPP1,J), SET W(J)=0., AND LOOP BACK TO TEST DUAL
            # COEFFS AGAIN.
            A[nsetp + 1, j] = Asave
            w[j] = 0
        end
        if terminated
            break
        end

        # THE INDEX  J=INDEX(IZ)  HAS BEEN SELECTED TO BE MOVED FROM
        # SET Z TO SET P.    UPDATE B,  UPDATE INDICES,  APPLY HOUSEHOLDER
        # TRANSFORMATIONS TO COLS IN NEW SET Z,  ZERO SUBDIAGONAL ELTS IN
        # COL J,  SET W(J)=0.
        b .= zz

        idx[iz] = idx[iz1]
        idx[iz1] = j
        iz1 += one(TI)
        nsetp += one(TI)

        if iz1 <= iz2
            for jz in iz1:iz2
                jj = idx[jz]
                apply_householder!(
                    fastview(A, sub2ind(A, nsetp, j), m - nsetp + 1),
                    up,
                    fastview(A, sub2ind(A, nsetp, jj), m - nsetp + 1))
            end
        end

        if nsetp != m
            for l in (nsetp + 1):m
                A[l, j] = 0
            end
        end

        w[j] = 0

        # SOLVE THE TRIANGULAR SYSTEM.
        # STORE THE SOLUTION TEMPORARILY IN ZZ().
        jj = solve_triangular_system!(zz, A, idx, nsetp, jj)

        # ******  SECONDARY LOOP BEGINS HERE ******
        #
        # ITERATION COUNTER.
        while true
            iter += 1
            if iter > max_iter
                work.mode = 3
                terminated = true
                println("NNLS quitting on iteration count")
                break
            end

            # SEE IF ALL NEW CONSTRAINED COEFFS ARE FEASIBLE.
            # IF NOT COMPUTE ALPHA.
            alpha = convert(T, 2)
            for ip in one(TI):nsetp
                l = idx[ip]
                if zz[ip] <= 0
                    t = -x[l] / (zz[ip] - x[l])
                    if alpha > t
                        alpha = t
                        jj = ip
                    end
                end
            end

            # IF ALL NEW CONSTRAINED COEFFS ARE FEASIBLE THEN ALPHA WILL
            # STILL = 2.    IF SO EXIT FROM SECONDARY LOOP TO MAIN LOOP.
            if alpha == 2
                break
            end

            # OTHERWISE USE ALPHA WHICH WILL BE BETWEEN 0 AND 1 TO
            # INTERPOLATE BETWEEN THE OLD X AND THE NEW ZZ.
            for ip in one(TI):nsetp
                l = idx[ip]
                x[l] = x[l] + alpha * (zz[ip] - x[l])
            end

            # MODIFY A AND B AND THE INDEX ARRAYS TO MOVE COEFFICIENT I
            # FROM SET P TO SET Z.
            i = idx[jj]

            while true
                x[i] = 0

                if jj != nsetp
                    jj += one(TI)
                    for j in jj:nsetp
                        ii = idx[j]
                        idx[j - 1] = ii
                        cc, ss, sig = orthogonal_rotmat(A[j - 1, ii], A[j, ii])
                        A[j - 1, ii] = sig
                        A[j, ii] = 0
                        for l in one(TI):n
                            if l != ii
                                # Apply procedure G2 (CC,SS,A(J-1,L),A(J,L))
                                temp = A[j - 1, l]
                                A[j - 1, l] = cc * temp + ss * A[j, l]
                                A[j, l] = -ss * temp + cc * A[j, l]
                            end
                        end

                        # Apply procedure G2 (CC,SS,B(J-1),B(J))
                        temp = b[j - 1]
                        b[j - 1] = cc * temp + ss * b[j]
                        b[j] = -ss * temp + cc * b[j]
                    end
                end

                nsetp -= one(TI)
                iz1 -= one(TI)
                idx[iz1] = i

                # SEE IF THE REMAINING COEFFS IN SET P ARE FEASIBLE.  THEY SHOULD
                # BE BECAUSE OF THE WAY ALPHA WAS DETERMINED.
                # IF ANY ARE INFEASIBLE IT IS DUE TO ROUND-OFF ERROR.  ANY
                # THAT ARE NONPOSITIVE WILL BE SET TO ZERO
                # AND MOVED FROM SET P TO SET Z.
                allfeasible = true
                for jj in one(TI):nsetp
                    i = idx[jj]
                    if x[i] <= 0
                        allfeasible = false
                        break
                    end
                end
                if allfeasible
                    break
                end
            end

            # COPY B( ) INTO ZZ( ).  THEN SOLVE AGAIN AND LOOP BACK.
            zz .= b
            jj = solve_triangular_system!(zz, A, idx, nsetp, jj)
        end
        if terminated
            break
        end
        # ******  END OF SECONDARY LOOP  ******

        for i in 1:nsetp
            x[idx[i]] = zz[i]
        end
        # ALL NEW COEFFS ARE POSITIVE.  LOOP BACK TO BEGINNING.
    end

    # ******  END OF MAIN LOOP  ******
    # COME TO HERE FOR TERMINATION.
    # COMPUTE THE NORM OF THE FINAL RESIDUAL VECTOR.

    sm = zero(T)
    if nsetp < m
        for i in (nsetp + 1):m
            sm += b[i]^2
        end
    else
        w .= 0
    end
    work.rnorm = sqrt(sm)
    work.nsetp = nsetp
    return work.x
end

function solve!{T}(work::NNLSWorkspace{T}, A::AbstractMatrix{T}, b::AbstractVector{T}, max_iter=(3 * size(A, 2)))
    load!(work, A, b)
    solve!(work, max_iter)
    work.x
end

function nnls{T}(A::DenseMatrix{T}, b::DenseVector{T}, max_iter=(3 * size(A, 2)))
    work = NNLSWorkspace(A, b)
    solve!(work, max_iter)
    work.x
end


# Implementation of an NNLS-based QP solver, based on section II of  Bemporad,
# "A quadratic programming algorithm based on nonnegative least squares with
# applications to embedded model predictive control", IEEE Transactions on
# Automatic Control, 2016.
# Variable names match the paper wherever possible

@static if VERSION < v"0.6-"       
    @compat const AllColsSubArray{T} = SubArray{T,2,Array{T,2},Tuple{UnitRange{Int},Colon},false}       
else      
    @compat const AllColsSubArray{T} = SubArray{T,2,Array{T,2},Tuple{UnitRange{Int},Base.Slice{Base.OneTo{Int}}},false}       
end

type QPWorkspace{T<:LinAlg.BlasReal, I}
    # Variables from paper:
    L::Matrix{T}
    c::Vector{T}
    G::Matrix{T}
    g::Vector{T}
    M::Matrix{T}
    d::Vector{T}
    r::Vector{T}

    # Additional variables:
    e::Vector{T} # intermediate variable
    A::Matrix{T} # 'A'-matrix of NNLS problem
    AM::AllColsSubArray{T} # upper block of A
    Ad::AllColsSubArray{T} # last row of A
    b::Vector{T} # 'b'-vector of NNLS problem
    nnlswork::NNLSWorkspace{T, I}
    status::Symbol

    function (::Type{QPWorkspace{T, I}}){T, I}(q::Integer, n::Integer)
        work = new{T, I}()
        resize!(work, q, n)
    end
end

QPWorkspace(q::Integer, n::Integer) = QPWorkspace{Float64, Int}(q, n)

"""
Structure describing the QP:

Minimize ``\\frac{1}{2} z' Q z + c' z``
Subject to ``G z \\leq g``
"""
struct QP{T}
    Q::Matrix{T}
    c::Vector{T}
    G::Matrix{T}
    g::Vector{T}
end

"""
    QPWorkspace(qp::QP)

Construct a workspace and load problem data for the QP

Minimize ``\\frac{1}{2} z' Q z + c' z``
Subject to ``G z \\leq g``
"""
function QPWorkspace(qp::QP{T}) where T
    work = QPWorkspace{T, Int}(size(qp.G, 1), size(qp.G, 2))
    load!(work, qp.Q, qp.c, qp.G, qp.g)
    work
end

function Base.resize!{T}(work::QPWorkspace{T}, q::Integer, n::Integer)
    work.L = Matrix{T}(n, n)
    work.c = Vector{T}(n)
    work.G = Matrix{T}(q, n)
    work.g = Vector{T}(q)
    work.M = Matrix{T}(q, n)
    work.d = Vector{T}(q)
    work.r = Vector{T}(n + 1)
    work.e = Vector{T}(n)
    work.A = Matrix{T}(n + 1, q)
    work.AM = view(work.A, 1 : n, :)
    work.Ad = view(work.A, n + 1 : n + 1, :)
    work.b = Vector{T}(n + 1)
    work.nnlswork = NNLSWorkspace{T, Int}(size(work.A)...)
    work.status = :Unsolved
    work
end

"""
    load!(work::QPWorkspace, Q, c, G, g)

Load problem data for the QP

Minimize ``\\frac{1}{2} z' Q z + c' z``
Subject to ``G z \\leq g``
"""
function load!{T}(work::QPWorkspace{T}, Q::AbstractMatrix{T}, c::AbstractVector{T}, G::AbstractMatrix{T}, g::AbstractVector{T})
    work.L .= Q
    work.c .= c
    work.G .= G
    work.g .= g
    work.status = :Unsolved
    nothing
end

load!(work::QPWorkspace{T}, qp::QP{T}) where {T} = load!(work, qp.Q, qp.c, qp.G, qp.g)

"""
    z, λ = solve!(work::QPWorkspace)

Solve the QP that was loaded into `work` using `load!`. Returns the primal 
solution ``z`` and the dual solution ``λ``.
"""
function solve!{T}(work::QPWorkspace{T}, eps_infeasible = 1e-4)
    work.status == :Unsolved || error("Problem was already solved.")

    L = work.L
    c = work.c
    G = work.G
    g = work.g
    M = work.M
    d = work.d
    e = work.e
    A = work.A
    AM = work.AM
    Ad = work.Ad
    b = work.b
    r = work.r
    nnlswork = work.nnlswork

    # Compute M
    LinAlg.LAPACK.potrf!('U', L) # L <- upper cholesky factor of Q
    M .= G
    LinAlg.BLAS.trsm!('R', 'U', 'N', 'N', 1., L, M) # M <- G L⁻¹

    # Compute d
    e .= c
    LinAlg.LAPACK.potrs!('U', L, e) # e <- Q⁻¹ c

    d .= g
    LinAlg.BLAS.gemv!('N', 1., G, e, 1., d) # d <- g + G Q⁻¹ c

    # Populate A
    transpose!(AM, M)
    transpose!(Ad, d)
    scale!(A, -1)

    # Populate b
    γ = one(T)
    b[:] = 0
    b[end] = γ

    # Solve the NNLS
    load!(nnlswork, A, b)
    solve!(nnlswork)
    y = nnlswork.x

    # Compute the residual
    r = b
    LinAlg.BLAS.gemv!('N', 1., A, y, -1., r) # r <- A * y - b

    # Check for feasibility
    work.status = sum(abs, r) < eps_infeasible ? :Infeasible : :Optimal

    # Back out solution and dual
    z = c
    λ = y
    if work.status == :Optimal
        # Note: r[end] == -(γ + d ⋅ y)
        LinAlg.BLAS.gemv!('T', 1 / r[end], G, y, -1., c) # z <- -1 / (γ + d ⋅ y) G^ᵀ y - c
        LinAlg.LAPACK.potrs!('U', L, z)
        scale!(λ, -1 / sqrt(-r[end])) # the sqrt appears to be missing in (12) in the paper
    else
        fill!(z, NaN)
        fill!(λ, NaN)
    end

    z, λ
end


"""
    primal_infeasibility(qp::QP, z)

Measure of primal infeasibility of the solution to the QP 

Minimize ``\\frac{1}{2} z' Q z + c' z``
Subject to ``G z \\leq g`

where ``z`` is the primal solution and ``λ`` is the dual solution.

For a primal-feasible solution, this will return a value <= 0.
"""
primal_infeasibility(qp::QP, z, λ) = maximum(qp.G * z .- qp.g)

"""
    dual_infeasibility(qp::QP, λ)

Measure of primal infeasibility of the solution to the QP 

Minimize ``\\frac{1}{2} z' Q z + c' z``
Subject to ``G z \\leq g`

where ``z`` is the primal solution and ``λ`` is the dual solution.

For a dual-feasible solution, this will return a value <= 0.
"""
dual_infeasibility(qp::QP, z, λ) = maximum(λ)

"""
    stationarity_violation(Q, c, G, g, z, λ)

Measure of the violation of the stationarity KKT condition for the QP. 

For an optimal solution, this should return a value near 0.
"""
function stationarity_violation(qp::QP, z, λ) 
    slack_grad = qp.G' * λ
    cost_grad = qp.Q * z + qp.c
    scale = mean(cost_grad ./ slack_grad)
    maximum(abs, cost_grad .- scale .* slack_grad)
end

"""
    slackness_violation(Q, c, G, g, z, λ)

Measure of the violation of the complementary slackness condition for the QP.

For an optimal solution, this should return a value near 0.
"""
slackness_violation(qp::QP, z, λ) = maximum(abs, (qp.G * z .- qp.g) .* λ)

function check_solution(qp::QP, z, λ)
    return max(
               primal_infeasibility(qp, z, λ),
               dual_infeasibility(qp, z, λ),
               stationarity_violation(qp, z, λ),
               slackness_violation(qp, z, λ)
               )
end

include("NNLSSolverInterface.jl")
using .NNLSSolverInterface: NNLSSolver

end # module
