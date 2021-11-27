struct InPlacePeriodicSolver
    nx :: Int64
    Δx :: Vector{Float64}
    δx :: Vector{Float64}
    u  :: Vector{Float64}
    
    tol :: Float64
    maxiter :: Int64
end

InPlacePeriodicSolver(xf; tol=1e-15, maxiter=100) = begin
    Δx = diff(xf)
    nx = length(Δx)
    xc = cumsum(Δx) .- 0.5Δx
    δx = diff([first(xf); xc; last(xf)])
    
    return InPlacePeriodicSolver(nx, Δx, δx, similar(xc), tol, maxiter)
end

function solve!(ps::InPlacePeriodicSolver, ρ)
    rhs = -ρ .* ps.Δx .^ 2 # extra allocation
    ps.u .= 0.0
    jacobi!(ps.u, rhs; K=ps.maxiter, ϵ=ps.tol)
    return nothing
end