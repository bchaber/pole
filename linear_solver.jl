import IterativeSolvers: cg!

struct LinearSolver{D, T <: Tuple}
    n :: NTuple{D,Int64}
    Δ :: Matrix{Float64}
    δ :: Matrix{Float64}
    
    A :: SparseMatrixCSC{Float64, Int64}
    b :: Vector{Float64}
    u :: Vector{Float64}
    
    bcs :: T
    bnds :: Tuple{Symbol, Symbol}
end

function LinearSolver(xf; left=DirichletBC(), right=DirichletBC())
    Δx = diff(xf)
    nx = length(Δx)
    xc = cumsum(Δx) .- 0.5Δx
    δx = diff([first(xf); xc; last(xf)])
    
    A  = spdiagm(0 => 2ones(nx),
                -1 => -ones(nx-1),
                +1 => -ones(nx-1))
    b  = zeros(nx)
    ls = LinearSolver((nx,), reshape(Δx,:,1), reshape(δx,:,1), A, b, similar(xc), (left, right), (:left, :right))
    for (bnd, bc) in zip(ls.bnds, ls.bcs)
        apply!(ls, bc, bnd)
    end
    return ls 
end

function apply!(solver::LinearSolver, bc::PeriodicBC, boundary::Symbol)
    A, b = solver.A, solver.b
    n, = solver.n
    h, = solver.δ
    m  = n - 1

    if boundary == :left
        A[1, :].= 0.
        A[1, 1] = 2.
        A[1, 2] =-1.
        A[1, n] =-1.
        b[1]    = 0.
    end

    if boundary == :right
        A[n, :].= 0.
        A[n, n] = 2.
        A[n, m] =-1.
        A[n, 1] =-1.
        b[n]    = 0.
    end
end

function apply!(solver::LinearSolver, bc::DirichletBC, boundary::Symbol)
    A, b = solver.A, solver.b
    n, = solver.n
    h, = solver.δ
    m  = n - 1

    if boundary == :left
        A[1, :].= 0.
        A[1, 1] = 4.
        A[1, 2] =-4/3
        b[1]    = 8/3 * bc.value
    end

    if boundary == :right
        A[n, :].= 0.
        A[n, n] = 4.
        A[n, m] =-4/3
        b[n]    = 8/3 * bc.value
    end
end

function solve!(ps::LinearSolver, ρ)
    @inbounds for i in eachindex(ρ)
      ρ[i] = ps.b[i] - ρ[i] * ps.Δ[i]^2
    end
    ps.u .= (ps.A \ ρ)
    return nothing
end
