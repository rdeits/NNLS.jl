module NNLSSolverInterface

export QPSolver

using NNLS: QPWorkspace, load!, solve!
importall MathProgBase.SolverInterface

immutable QPSolver <: AbstractMathProgSolver
    workspace::QPWorkspace{Float64, Int}

    QPSolver() = new(QPWorkspace{Float64, Int}(0, 0))
end

type QPModel <: AbstractLinearQuadraticModel
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

    QPModel(work::QPWorkspace) = new(work)
end

LinearQuadraticModel(s::QPSolver) = QPModel(s.workspace)

numvar(work::QPWorkspace) = size(work.G, 2)
numconstr(work::QPWorkspace) = size(work.G, 1)

numvar(m::QPModel) = numvar(m.workspace)
numconstr(m::QPModel) = numconstr(m.workspace)

function loadproblem!(m::QPModel, A, lb, ub, q, constr_lb, constr_ub, sense)
    m.sense = sense
    m.A = A
    m.lb = lb
    m.ub = ub
    m.constr_lb = constr_lb
    m.constr_ub = constr_ub
    m.q = q
end

setquadobj!(m::QPModel, Q) = m.Q = Q
setquadobj!(m::QPModel, rowidx, colidx, quadval) = m.Q = sparse(rowidx, colidx, quadval)

# TODO: this is only needed for Julia v0.5
minus(x) = -x

function optimize!(m::QPModel)
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

    m.workspace.L = m.Q
    m.workspace.c = m.q

    m.workspace.G .= 0

    m.workspace.G[1:nrows, :] .= m.A
    m.workspace.g[1:nrows] .= m.constr_ub
    m.workspace.G[nrows + (1:nrows), :] .= minus.(m.A)
    m.workspace.g[nrows + (1:nrows)] .= minus.(m.constr_lb)
    for i in 1:nvars
        m.workspace.G[(2 * nrows) + i, i] = 1
    end
    m.workspace.g[(2 * nrows) + (1:nvars)] .= m.ub
    for i in 1:nvars
        m.workspace.G[(2 * nrows + nvars) + i, i] = -1
    end
    m.workspace.g[(2 * nrows + nvars) + (1:nvars)] .= minus.(m.lb)

    for i in 1:length(m.workspace.g)
        if isinf(m.workspace.g[i])
            m.workspace.G[i, :] .= 0
            m.workspace.g[i] = 0
        end
    end

    m.workspace.status = :Unsolved

    m.solution, m.duals = solve!(m.workspace)
end

getsolution(m::QPModel) = m.solution
getobjval(m::QPModel) = 0.5 * m.solution' * m.Q * m.solution + m.q' * m.solution

status(m::QPModel) = m.workspace.status

end
