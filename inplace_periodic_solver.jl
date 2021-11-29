struct InPlacePeriodicSolver
    nx :: Int64
    Δx :: Vector{Float64}
    δx :: Vector{Float64}
    u  :: Vector{Float64}
    # solver cofiguration
    tol :: Float64
    maxiter :: Int64
    # scratchpad
    rhs :: Vector{Float64}
    xi :: Vector{Float64}
    xj :: Vector{Float64}
end

InPlacePeriodicSolver(xf; tol=1e-15, maxiter=100) = begin
    Δx = diff(xf)
    nx = length(Δx)
    xc = cumsum(Δx) .- 0.5Δx
    δx = diff([first(xf); xc; last(xf)])
    
    return InPlacePeriodicSolver(nx, Δx, δx, similar(xc), tol, maxiter,
            similar(Δx), similar(Δx), similar(Δx))
end

function solve!(ps::InPlacePeriodicSolver, ρ)
    ps.rhs .= -ρ .* ps.Δx .^ 2 # extra allocation
    ps.u .= 0.0
    jacobi!(ps.u, ps.rhs; K=ps.maxiter, ϵ=ps.tol, xi=ps.xi, xj=ps.xj)
    return nothing
end