import IterativeSolvers: cg!

struct LinearSolver{D, T <: Tuple}
    n :: NTuple{D,Int64}
    Δ :: Matrix{Float64}
    δ :: Matrix{Float64}
    
    A :: SparseMatrixCSC{Float64, Int64}
    b :: Vector{Float64}
    u :: Vector{Float64}

    rhs :: Vector{Float64}   
    dof :: Matrix{Int64}
    bcs :: T
end

function LinearSolver(xf, yf; left=DirichletBC(), right=DirichletBC(), upper=DirichletBC(), lower=DirichletBC())
    Δx = diff(xf)
    Δy = diff(xf)
    nx = length(Δx)
    ny = length(Δy)
    xc = cumsum(Δx) .- 0.5Δx
    yc = cumsum(Δy) .- 0.5Δy
    δx = diff([first(xf); xc; last(xf)])
    δy = diff([first(yf); yc; last(yf)])
    
    N  = nx * ny
    M  = nx
    A  = spdiagm(0 => 4ones(N),
                -1 => -ones(N-1),
                +1 => -ones(N-1),
                -M => -ones(N-M),
                +M => -ones(N-M))
    b  = zeros(N)
    dof = collect(reshape(1:N, nx, ny))

    @inbounds for i = 1, j = 1:ny
        n = dof[i,  j]
        m = dof[i+1,j]
        k = dof[nx, j]
        apply!(A, left, n, m, k)
        apply!(b, left, n)
    end

    @inbounds for i = nx, j = 1:ny
        n = dof[i,  j]
        m = dof[i-1,j]
        k = dof[1,  j]
        apply!(A, right, n, m, k)
        apply!(b, right, n)
    end

    @inbounds for i = 1:nx, j = 1
        n = dof[i,j]
        m = dof[i,j+1]
        k = dof[i,ny]
        apply!(A, upper, n, m, k)
        apply!(b, upper, n)
    end

    @inbounds for i = 1:nx, j = ny
        n = dof[i,j]
        m = dof[i,j-1]
        k = dof[i,1]
        apply!(A, lower, n, m, k)
        apply!(b, lower, n)
    end

    return LinearSolver((nx,ny), hcat(Δx, Δy), hcat(δx, δy), A, b, similar(b), similar(b), dof, (left, right, upper, lower))
end

function LinearSolver(xf; left=DirichletBC(), right=DirichletBC())
    Δx = diff(xf)
    nx = length(Δx)
    xc = cumsum(Δx) .- 0.5Δx
    δx = diff([first(xf); xc; last(xf)])
    
    N  = nx
    A  = spdiagm(0 => 2ones(N),
                -1 => -ones(N-1),
                +1 => -ones(N-1))
    b  = zeros(N)
    dof = collect(reshape(1:N, nx, 1))
    
    @inbounds for i = 1
        n = dof[i]
        m = dof[i+1]
        k = dof[nx]
        apply!(A, left, n, m, k)
        apply!(b, left, n)
    end

    @inbounds for i = nx
        n = dof[i]
        m = dof[i-1]
        k = dof[1]
        apply!(A, right, n, m, k)
        apply!(b, right, n)
    end

    return LinearSolver((nx,), hcat(Δx), hcat(δx), A, b, similar(b), similar(b), dof, (left, right))
end

@inline function apply!(A::SparseMatrixCSC{Float64, Int64}, bc::DirichletBC, n, m, k)
    A[n, n] += 6/3
    A[n, m] -= 1/3
    return nothing
end

@inline function apply!(b::Vector{Float64}, bc::DirichletBC, n)
    b[n]   += 8/3 * bc.value
    return nothing
end

@inline function apply!(A::SparseMatrixCSC{Float64, Int64}, bc::PeriodicBC, n, m, k)
    A[n, k] -= 1.
    return nothing
end

@inline function apply!(b::Vector{Float64}, bc::PeriodicBC, n)
    return nothing
end

function solve!(ps::LinearSolver{1, T}, ρ) where {T}
    @inbounds for i in eachindex(ρ)
      ps.rhs[i] = ps.b[i] - ρ[i] * ps.Δ[i]^2 # assume uniform mesh in all directions
    end
    ps.u .= (ps.A \ ps.rhs)
    return nothing
end
