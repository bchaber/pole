import IterativeSolvers: cg!

struct LinearSolver{D, T <: Tuple}
    n :: NTuple{D,Int64}
    Δ :: NTuple{D, Vector{Float64}}
    δ :: NTuple{D, Vector{Float64}}

    h :: Float64
    
    A :: SparseMatrixCSC{Float64, Int64}
    b :: Vector{Float64}
    u :: Vector{Float64}

    rhs :: Vector{Float64}   
    dof :: Matrix{Int64}
    bcs :: T
end

function LinearSolver(xf, yf; left=DirichletBC(), right=DirichletBC(), upper=DirichletBC(), lower=DirichletBC())
    Δx = diff(xf)
    Δy = diff(yf)
    nx = length(Δx)
    ny = length(Δy)
    xc = cumsum(Δx) .- 0.5Δx
    yc = cumsum(Δy) .- 0.5Δy
    δx = diff([first(xf); xc; last(xf)])
    δy = diff([first(yf); yc; last(yf)])
    @assert minimum(Δx) ≈ maximum(Δx) ≈
            minimum(Δy) ≈ maximum(Δy) "Currently only uniform cells are supported :("

    h  = first(Δx)
    N  = nx * ny
    A  = spzeros(N, N)
    b  = zeros(N)
    dof = collect(reshape(1:N, nx, ny))

    @inbounds for i = 2:nx-1, j = 2:ny-1
        n = dof[i-1,j]
        m = dof[i+1,j]
        o = dof[i,j]
        k = dof[i,j-1]
        l = dof[i,j+1]
        A[o, o] = 4.
        A[o, n] =-1.
        A[o, m] =-1.
        A[o, k] =-1.
        A[o, l] =-1.
    end

    @inbounds for i = 1, j = 1:ny
        n = dof[i+1,j]
        m = dof[nx, j]
        o = dof[i,  j]
        apply!(A, left, o, n, m)
        apply!(b, left, o)
    end

    @inbounds for i = nx, j = 1:ny
        n = dof[i-1,j]
        m = dof[1,  j]
        o = dof[i,  j]
        apply!(A, right, o, n, m)
        apply!(b, right, o)
    end

    @inbounds for i = 1:nx, j = 1
        n = dof[i,j+1]
        m = dof[i,ny]
        o = dof[i,j]
        apply!(A, upper, o, n, m)
        apply!(b, upper, o)
    end

    @inbounds for i = 1:nx, j = ny
        n = dof[i,j-1]
        m = dof[i,1]
        o = dof[i,j]
        apply!(A, lower, o, n, m)
        apply!(b, lower, o)
    end
    
    return LinearSolver((nx,ny), (Δx,Δy), (δx,δy), h, A, b, similar(b), similar(b), dof, (left, right, upper, lower))
end

function LinearSolver(xf; left=DirichletBC(), right=DirichletBC())
    Δx = diff(xf)
    nx = length(Δx)
    xc = cumsum(Δx) .- 0.5Δx
    δx = diff([first(xf); xc; last(xf)])
    @assert minimum(Δx) ≈ maximum(Δx) "Currently only uniform cells are supported :("

    h  = first(Δx)   
    N  = nx
    A  = spzeros(N, N)
    b  = zeros(N)
    dof = collect(reshape(1:N, nx, 1))
    
    @inbounds for i = 2:nx-1
        n = dof[i-1]
        m = dof[i+1]
        o = dof[i]
        A[o, o] = 2.
        A[o, n] =-1.
        A[o, m] =-1.
    end

    @inbounds for i = 1
        n = dof[i+1]
        m = dof[nx]
        o = dof[i]
        apply!(A, left, o, n, m)
        apply!(b, left, o)
    end

    @inbounds for i = nx
        n = dof[i-1]
        m = dof[1]
        o = dof[i]
        apply!(A, right, o, n, m)
        apply!(b, right, o)
    end

    return LinearSolver((nx,), (Δx,), (δx,), h, A, b, similar(b), similar(b), dof, (left, right))
end

@inline function reset!(A::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, o)
    @assert b[o] == 0. "You are going to reset a degree of freedom that has b[dof] != 0"
    A[o, :] .= 0.
end

@inline function fixed!(A::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, value, o)
    reset!(A, b, o)
    A[o, o] = 1.
    b[o]    = value
    return nothing
end

@inline function apply!(A::SparseMatrixCSC{Float64, Int64}, bc::NeumannBC, o, n, m)
    A[o, o] += 1.
    A[o, n] -= 1.
    return nothing
end

@inline function apply!(b::Vector{Float64}, bc::NeumannBC, o)
    b[o] -= bc.value
    return nothing
end

@inline function apply!(A::SparseMatrixCSC{Float64, Int64}, bc::DirichletBC, o, n, m)
    A[o, o] += 8/3
    A[o, n] -= 4/3
    return nothing
end

@inline function apply!(b::Vector{Float64}, bc::DirichletBC, o)
    b[o]   += 4/3 * bc.value
    return nothing
end

@inline function apply!(A::SparseMatrixCSC{Float64, Int64}, bc::PeriodicBC, o, n, m)
    A[o, o] += 2.
    A[o, n] -= 1.
    A[o, m] -= 1.
    return nothing
end

@inline function apply!(b::Vector{Float64}, bc::PeriodicBC, o)
    return nothing
end

function solve!(ps::LinearSolver{D, T}, ρ) where {D, T}
    @inbounds for i in eachindex(ρ)
      ps.rhs[i] = ps.b[i] - ρ[i] * ps.h^2 # assume uniform mesh in all directions
    end
    ps.u .= (ps.A \ ps.rhs)
    return nothing
end

function cylindrical!(ps::LinearSolver, rc)
    nz, nr = ps.n
    h  = ps.h
    A  = ps.A
    b  = ps.b

    for i = 1:nz, j = 2:nr-1
        r = rc[j]
        if r ≈ 0.0 continue end
        n = ps.dof[i,j-1]
        m = ps.dof[i,j+1]
        o = ps.dof[i,j]
        A[o, n] += 0.5(h/r)
        A[o, m] -= 0.5(h/r)
    end
    return nothing
end