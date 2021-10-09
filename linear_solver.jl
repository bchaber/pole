struct LinearSolver
    nx :: Int64
    Δx :: Vector{Float64}
    δx :: Vector{Float64}
    
    A :: SparseMatrixCSC{Float64, Int64}
    b :: Vector{Float64}
    u :: Vector{Float64}
    
    bcs :: Dict{Symbol, BoundaryCondition}
end

LinearSolver(xf) = begin
    Δx = diff(xf)
    nx = length(Δx)
    xc = cumsum(Δx) .- 0.5Δx
    δx = diff([first(xf); xc; last(xf)])
    
    A  = spdiagm(0 => 2ones(nx),
                -1 => -ones(nx-1),
                +1 => -ones(nx-1))
    b  = zeros(nx)
    return LinearSolver(nx, Δx, δx, A, b, similar(xc), Dict())
end

apply!(solver::LinearSolver, bc::PeriodicBC, boundary::Symbol) = begin
    A, n = solver.A, solver.nx
    if boundary == :left  A[1, n] = -1.0 end
    if boundary == :right A[n, 1] = -1.0 end
end

apply!(solver::LinearSolver, bc::DirichletBC, boundary::Symbol) = begin
    A, b = solver.A, solver.b
    n, h = solver.nx, solver.δx
    m = n -1
    
    if boundary == :left
        A[1, 1] = 4.0
        A[1, 2] =-4/3
        b[1]   += 8/3 * bc.value
    end
    if boundary == :right
        A[n, n] = 4.0
        A[n, m] =-4/3
        b[n]   += 8/3 * bc.value
    end
    
end