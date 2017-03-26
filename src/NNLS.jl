module NNLS

export nnls,
       nnls!,
       NNLSWorkspace

"""
CONSTRUCTION AND/OR APPLICATION OF A SINGLE   
HOUSEHOLDER TRANSFORMATION..     Q = I + U*(U**T)/B   
 
The original version of this code was developed by
Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory
1973 JUN 12, and published in the book
"SOLVING LEAST SQUARES PROBLEMS", Prentice-HalL, 1974.
Revised FEB 1995 to accompany reprinting of the book by SIAM.
"""
function construct_householder!(u::AbstractVector, up)
    m = length(u)
    if m <= 1
        return up
    end
    
    cl = maximum(abs, u)
    @assert cl > 0
    clinv = 1 / cl
    sm = zero(eltype(u))
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
        # i2 = 1 - m + lpivot - 1
        # incr = 1
        # i2 = lpivot
        # i3 = lpivot + 1
        # i4 = lpivot + 1

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
CONSTRUCTION AND/OR APPLICATION OF A SINGLE   
HOUSEHOLDER TRANSFORMATION..     Q = I + U*(U**T)/B   
 
The original version of this code was developed by
Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory
1973 JUN 12, and published in the book
"SOLVING LEAST SQUARES PROBLEMS", Prentice-HalL, 1974.
Revised FEB 1995 to accompany reprinting of the book by SIAM.
"""
function orthogonal_rotmat{T}(a::T, b::T)
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
function solve_triangular_system(zz, A, idx, nsetp, jj)
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

immutable NNLSWorkspace{T, I <: Integer}
    x::Vector{T}
    w::Vector{T}
    zz::Vector{T}
    idx::Vector{I}
    rnorm::Ref{T}
    mode::Ref{I}
end

function NNLSWorkspace{T}(::Type{T}, m::Integer, n::Integer)
    NNLSWorkspace{T, Int}(
        zeros(T, n),
        zeros(T, n),
        zeros(T, m),
        zeros(Int, n),
        zero(T),
        0)
end

function NNLSWorkspace{T, I <: Integer}(::Type{T}, ::Type{I}, m::Integer, n::Integer)
    NNLSWorkspace{T, I}(
        zeros(T, n),
        zeros(T, n),
        zeros(T, m),
        zeros(Int, n),
        zero(T),
        0)
end

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
Base.linearindexing{V <: UnsafeVectorView}(::Type{V}) = Base.LinearFast()

function nnls{T}(A::DenseMatrix{T}, b::DenseVector{T}, itermax=(3 * size(A, 2)))
    work = NNLSWorkspace(T, size(A, 1), size(A, 2))
    nnls!(work, copy(A), copy(b), itermax)
    work.x
end

@noinline function checkargs(work::NNLSWorkspace, A::DenseMatrix, b::DenseVector)
    m, n = size(A)
    @assert size(b) == (m,)
    @assert size(work.x) == (n,)
    @assert size(work.w) == (n,)
    @assert size(work.zz) == (m,)
    @assert size(work.idx) == (n,)
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
function nnls!{T}(work::NNLSWorkspace{T}, A::DenseMatrix{T}, b::DenseVector{T}, itermax=(3 * size(A, 2)))
    checkargs(work, A, b)

    x = work.x
    w = work.w
    zz = work.zz
    idx = work.idx
    const factor = 0.01
    work.mode[] = 1
    
    m, n = size(A)
    
    iter = 0
    x .= 0
    idx .= 1:n
    
    izmax = 0
    iz2 = n
    iz1 = 1
    iz = 0
    j = 0
    nsetp = 0
    npp1 = 1
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
        for iz in iz1:iz2
            j = idx[iz]
            sm = zero(T)
            for l in npp1:m
                sm += A[l, j] * b[l]
            end
            w[j] = sm
        end
        
        # FIND LARGEST POSITIVE W(J).
        while true
            wmax = zero(T)
            for iz in iz1:iz2
                j = idx[iz]
                if w[j] > wmax
                    wmax = w[j]
                    izmax = iz
                end
            end
            
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
            Asave = A[npp1, j]
            up = construct_householder!(
                UnsafeVectorView(A, sub2ind(A, npp1, j), m - npp1 + 1),
                up)
            # up = construct_householder!(@view(A[npp1:end, j]), up)
            unorm = zero(T)
            if nsetp != 0
                for l in 1:nsetp
                    unorm += A[l, j]^2
                end
            end
            unorm = sqrt(unorm)

            if ((unorm + abs(A[npp1, j]) * factor) - unorm) > 0 
                # COL J IS SUFFICIENTLY INDEPENDENT.  COPY B INTO ZZ, UPDATE ZZ
                # AND SOLVE FOR ZTEST ( = PROPOSED NEW VALUE FOR X(J) ).   
                # println("copying b into zz")
                zz .= b
                apply_householder!(
                    UnsafeVectorView(A, sub2ind(A, npp1, j), m - npp1 + 1),
                    up,
                    UnsafeVectorView(zz, npp1, m - npp1 + 1))
                # apply_householder!(@view(A[npp1:end, j]), up, @view(zz[npp1:end]))
                # print("after h12: ")
                ztest = zz[npp1] / A[npp1, j]

                # SEE IF ZTEST IS POSITIVE  
                if ztest > 0
                    break
                end
            end

            # REJECT J AS A CANDIDATE TO BE MOVED FROM SET Z TO SET P.  
            # RESTORE A(NPP1,J), SET W(J)=0., AND LOOP BACK TO TEST DUAL
            # COEFFS AGAIN.
            A[npp1, j] = Asave
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
        iz1 += 1
        nsetp = npp1
        npp1 += 1

        if iz1 <= iz2
            for jz in iz1:iz2
                jj = idx[jz]
                apply_householder!(
                    UnsafeVectorView(A, sub2ind(A, nsetp, j), m - nsetp + 1),
                    up,
                    UnsafeVectorView(A, sub2ind(A, nsetp, jj), m - nsetp + 1))
                # apply_householder!(@view(A[nsetp:end, j]), up, @view(A[nsetp:end, jj]))
            end
        end

        if nsetp != m
            for l in npp1:m
                A[l, j] = 0
            end
        end

        w[j] = 0

        # SOLVE THE TRIANGULAR SYSTEM.   
        # STORE THE SOLUTION TEMPORARILY IN ZZ().
        jj = solve_triangular_system(zz, A, idx, nsetp, jj)

        # ******  SECONDARY LOOP BEGINS HERE ******  
        # 
        # ITERATION COUNTER.   
        while true
            iter += 1
            if iter > itermax
                work.mode[] = 3
                terminated = true
                println("NNLS quitting on iteration count")
                break
            end

            # SEE IF ALL NEW CONSTRAINED COEFFS ARE FEASIBLE. 
            # IF NOT COMPUTE ALPHA.    
            alpha = 2.0
            for ip in 1:nsetp
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
            for ip in 1:nsetp
                l = idx[ip]
                x[l] = x[l] + alpha * (zz[ip] - x[l])
            end

            # MODIFY A AND B AND THE INDEX ARRAYS TO MOVE COEFFICIENT I
            # FROM SET P TO SET Z.
            i = idx[jj]

            while true
                x[i] = 0

                if jj != nsetp
                    jj += 1
                    for j in jj:nsetp
                        ii = idx[j]
                        idx[j - 1] = ii
                        cc, ss, sig = orthogonal_rotmat(A[j - 1, ii], A[j, ii])
                        A[j - 1, ii] = sig
                        A[j, ii] = 0
                        for l in 1:n
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

                npp1 = nsetp
                nsetp = nsetp - 1
                iz1 = iz1 - 1
                idx[iz1] = i

                # SEE IF THE REMAINING COEFFS IN SET P ARE FEASIBLE.  THEY SHOULD
                # BE BECAUSE OF THE WAY ALPHA WAS DETERMINED.
                # IF ANY ARE INFEASIBLE IT IS DUE TO ROUND-OFF ERROR.  ANY   
                # THAT ARE NONPOSITIVE WILL BE SET TO ZERO   
                # AND MOVED FROM SET P TO SET Z. 
                allfeasible = true
                for jj in 1:nsetp
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
            jj = solve_triangular_system(zz, A, idx, nsetp, jj)
        end
        if terminated
            break
        end
        # ******  END OF SECONDARY LOOP  ******

        for ip in 1:nsetp
            i = idx[ip]
            x[i] = zz[ip]
        end
        # ALL NEW COEFFS ARE POSITIVE.  LOOP BACK TO BEGINNING.
    end

    # ******  END OF MAIN LOOP  ******   
    # COME TO HERE FOR TERMINATION. 
    # COMPUTE THE NORM OF THE FINAL RESIDUAL VECTOR.

    sm = zero(T)
    if npp1 <= m
        for i in npp1:m
            sm += b[i]^2
        end
    else
        for j in 1:n
            w[j] = 0
        end
    end
    work.rnorm[] = sqrt(sm)
end


end # module