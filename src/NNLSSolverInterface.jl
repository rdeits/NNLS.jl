module NNLSSolverInterface

export NNLSSolver

using NNLS: QPWorkspace, load!, solve!
using MathProgBase.SolverInterface

struct NNLSSolver <: AbstractMathProgSolver
end

mutable struct NNLSModel <: AbstractLinearQuadraticModel
    workspace::QPWorkspace{Float64, Int}
    lb::Vector{Float64}
    ub::Vector{Float64}
    A::Matrix{Float64}
    constr_lb::Vector{Float64}
    constr_ub::Vector{Float64}
    Q::Matrix{Float64}
    q::Vector{Float64}
    sense::Symbol
    solution::Vector{Float64}
    duals::Vector{Float64}

    function NNLSModel()
        m = new(QPWorkspace{Float64, Int}(0, 0))
        resize!(m, 0, 0)
        m
    end
end

function Base.resize!(m::NNLSModel, q::Integer, n::Integer)
    resize!(m.workspace, q, n)
    m.lb = zeros(n)
    m.ub = zeros(n)
    m.A = zeros(q, n)
    m.constr_lb = zeros(q)
    m.constr_ub = zeros(q)
    m.Q = zeros(n, n)
    m.q = zeros(n)
    m.sense = :Min
    m.solution = fill(NaN, n)
    m.duals = fill(NaN, q)
end

LinearQuadraticModel(s::NNLSSolver) = NNLSModel()

numvar(work::QPWorkspace) = size(work.G, 2)
numconstr(work::QPWorkspace) = size(work.G, 1)

numvar(m::NNLSModel) = numvar(m.workspace)
numconstr(m::NNLSModel) = numconstr(m.workspace)

function loadproblem!(m::NNLSModel, A, lb, ub, obj, constr_lb, constr_ub, sense)
    q, n = size(A)
    if (q != size(m.A, 1)) || (n != size(m.A, 2))
        resize!(m, q, n)
    end
    m.sense = sense
    m.A .= A
    m.lb .= lb
    m.ub .= ub
    m.constr_lb .= constr_lb
    m.constr_ub .= constr_ub
    m.q .= obj
    m.Q .= 0
    m.solution .= NaN
    m.duals .= NaN
    m.workspace.status = :Unsolved
end

setquadobj!(m::NNLSModel, Q) = m.Q = Q
function setquadobj!(m::NNLSModel, rowidx, colidx, quadval)
    m.Q .= 0
    for i in 1:length(rowidx)
        x = m.Q[rowidx[i], colidx[i]] + quadval[i]
        m.Q[rowidx[i], colidx[i]] = x
        m.Q[colidx[i], rowidx[i]] = x
    end
end

function optimize!(m::NNLSModel)
    nvars = size(m.A, 2)
    nrows = size(m.A, 1)
    @assert nrows == length(m.constr_lb)
    @assert nrows == length(m.constr_ub)
    @assert nvars == length(m.lb)
    @assert nvars == length(m.ub)
    @assert nvars == length(m.q)

    nconstr = 2 * nrows + 2 * nvars

    if nvars != numvar(m.workspace) || nconstr != numconstr(m.workspace)
        resize!(m.workspace, nconstr, nvars)
        m.solution = fill(NaN, nvars)
    end

    m.workspace.L .= m.Q
    m.workspace.c .= m.q

    m.workspace.G .= 0

    for j in 1:nvars, i in 1:nrows
        m.workspace.G[i, j] = m.A[i, j]
    end
    for i in 1:nrows
        m.workspace.g[i] = m.constr_ub[i]
    end
    for j in 1:nvars, i in 1:nrows
        m.workspace.G[i + nrows, j] = -m.A[i, j]
    end
    for i in 1:nrows
        m.workspace.g[i + nrows] = -m.constr_lb[i]
    end
    for j in 1:nvars
        m.workspace.G[2 * nrows + j, j] = 1
    end
    for i in 1:nvars
        m.workspace.g[2 * nrows + i] = m.ub[i]
    end
    for i in 1:nvars
        m.workspace.G[2 * nrows + nvars + i, i] = -1
    end
    for i in 1:nvars
        m.workspace.g[2 * nrows + nvars + i] = -m.lb[i]
    end

    for i in 1:length(m.workspace.g)
        if isinf(m.workspace.g[i])
            m.workspace.G[i, :] .= 0
            m.workspace.g[i] = 0
        end
    end

    m.workspace.status = :Unsolved

    m.solution, λ = solve!(m.workspace)
    for i in 1:nrows
        if abs(λ[i]) > abs(λ[i + nrows])
            m.duals[i] = λ[i]
        else
            m.duals[i] = λ[i + nrows]
        end
    end
end

getsolution(m::NNLSModel) = m.solution
getobjval(m::NNLSModel) = 0.5 * m.solution' * m.Q * m.solution + m.q' * m.solution
getconstrduals(m::NNLSModel) = m.duals

status(m::NNLSModel) = m.workspace.status

end
