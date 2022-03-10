struct Stencil end
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
const Forward = Val{:+}
const Reverse = Val{:-}

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

    return LinearSolver((nx,), (Δx,), (δx,), h, A, b, similar(b), similar(b), dof, (left, right))
end

function cartesian!(ps::LinearSolver{1, T}) where {T}
    left, right = ps.bcs
    dof = ps.dof
    nx, = ps.n
    A  = ps.A
    b  = ps.b

    stencil = Stencil()
    fwd = Val(:+)
    rev = Val(:-)

    @inbounds for i = 2:nx-1
        n = dof[i-1]
        m = dof[i+1]
        o = dof[i]
        cartesian!(A, stencil, fwd, o, n)
        cartesian!(A, stencil, fwd, o, m)
    end

    @inbounds for i = 1
        n = dof[nx]
        m = dof[i+1]
        o = dof[i]
        cartesian!(A, left,    fwd, o, m, n)
        cartesian!(b, left,    fwd, o)
        cartesian!(A, stencil, fwd, o, m)
    end

    @inbounds for i = nx
        n = dof[i-1]
        m = dof[1]
        o = dof[i]
        cartesian!(A, stencil, fwd, o, n)
        cartesian!(A, right,   fwd, o, n, m)
        cartesian!(b, right,   fwd, o)
    end
end

function cylindrical!(ps::LinearSolver, rf)
    left, right = ps.bcs
    dof = ps.dof
    nr, = ps.n
    h  = ps.h
    A  = ps.A
    b  = ps.b

    stencil = Stencil()
    fwd = Val(:+)
    rev = Val(:-)

    @inbounds for i = 2:nr-1
        n = dof[i-1]
        m = dof[i+1]
        o = dof[i]
        radial!(A, stencil, rev, h/2rf[i],   o, n)
        radial!(A, stencil, fwd, h/2rf[i+1], o, m)
    end

    @inbounds for i = 1
        n = dof[nr]
        m = dof[i+1]
        o = dof[i]
        radial!(A, left,    rev, h/2rf[i], o, m, n)
        radial!(b, left,    rev, h/2rf[i], o)
        radial!(A, stencil, fwd, h/2rf[i+1], o, m)
    end

    @inbounds for i = nr
        n = dof[i-1]
        m = dof[1]
        o = dof[i]
        radial!(A, stencil, rev, h/2rf[i], o, n)
        radial!(A, right,   fwd, h/2rf[i+1], o, n, m)
        radial!(b, right,   fwd, h/2rf[i+1], o)
    end
    return nothing
end

# Stencil
@inline function cartesian!(A::SparseMatrixCSC{Float64, Int64}, bc::Stencil, dir::Forward, i, j)
    A[i, i] -= 1.; A[i, j] += 1.
end
@inline function cartesian!(A::SparseMatrixCSC{Float64, Int64}, bc::Stencil, dir::Reverse, i, j)
    A[i, i] += 1.; A[i, j] -= 1.
end
# Periodic Boundary Condition
@inline function cartesian!(A::SparseMatrixCSC{Float64, Int64}, bc::PeriodicBC, dir::Forward, i, _, k)
    A[i, i] -= 1.; A[i, k] += 1.
end
@inline function cartesian!(A::SparseMatrixCSC{Float64, Int64}, bc::PeriodicBC, dir::Reverse, i, _, k)
    A[i, j] += 1.; A[i, k] -= 1.
end
@inline function cartesian!(b::Vector{Float64}, bc::PeriodicBC, dir::Forward, i) end
@inline function cartesian!(b::Vector{Float64}, bc::PeriodicBC, dir::Reverse, i) end
# Neumann Boundary Condition
@inline function cartesian!(A::SparseMatrixCSC{Float64, Int64}, bc::NeumannBC, dir::Forward, i, j, _) end
@inline function cartesian!(A::SparseMatrixCSC{Float64, Int64}, bc::NeumannBC, dir::Reverse, i, j, _) end
@inline function cartesian!(b::Vector{Float64}, bc::NeumannBC, dir::Forward, i) b[i] -= bc.value end
@inline function cartesian!(b::Vector{Float64}, bc::NeumannBC, dir::Reverse, i) b[i] += bc.value end
# Dirichlet Boundary Condition
@inline function cartesian!(A::SparseMatrixCSC{Float64, Int64}, bc::DirichletBC, dir::Forward, i, j, _)
    A[i, i] -= 3.; A[i, j] += 1/3
end
@inline function cartesian!(A::SparseMatrixCSC{Float64, Int64}, bc::DirichletBC, dir::Reverse, i, j, _)
    A[i, j] += 3.; A[i, j] -= 1/3
end
@inline function cartesian!(b::Vector{Float64}, bc::DirichletBC, dir::Forward, i)
    b[i] -= (8/3)bc.value
end
@inline function cartesian!(b::Vector{Float64}, bc::DirichletBC, dir::Reverse, i)
    b[i] += (8/3)bc.value
end

# Stencil
@inline function radial!(A::SparseMatrixCSC{Float64, Int64}, bc::Stencil, dir::Forward, α, i, j)
    A[i, i] -= α; A[i, j] += α
end
@inline function radial!(A::SparseMatrixCSC{Float64, Int64}, bc::Stencil, dir::Reverse, α, i, j)
    A[i, i] += α; A[i, j] -= α
end
# Neumann Boundary Condition
@inline function radial!(A::SparseMatrixCSC{Float64, Int64}, bc::NeumannBC, dir::Forward, α, i, j, _) end
@inline function radial!(A::SparseMatrixCSC{Float64, Int64}, bc::NeumannBC, dir::Reverse, α, i, j, _) end
@inline function radial!(b::Vector{Float64}, bc::NeumannBC, dir::Forward, α, i) b[i] -= α*bc.value end
@inline function radial!(b::Vector{Float64}, bc::NeumannBC, dir::Reverse, α, i) b[i] += α*bc.value end
# Dirichlet Boundary Condition
@inline function radial!(A::SparseMatrixCSC{Float64, Int64}, bc::DirichletBC, dir::Forward, α, i, j, _)
    A[i, i] -= 3.0α; A[i, j] += α/3.0
end
@inline function radial!(A::SparseMatrixCSC{Float64, Int64}, bc::DirichletBC, dir::Reverse, α, i, j, _)
    A[i, i] += 3.0α; A[i, j] -= α/3.0
end
@inline function radial!(b::Vector{Float64}, bc::DirichletBC, dir::Forward, α, i)
    b[i] -= α*(8/3)bc.value
end
@inline function radial!(b::Vector{Float64}, bc::DirichletBC, dir::Reverse, α, i)
    b[i] += α*(8/3)bc.value
end

# Solver
function solve!(ps::LinearSolver{D, T}, ρ) where {D, T}
    @inbounds for i in eachindex(ρ)
      ps.rhs[i] = ps.b[i] - ρ[i] * ps.h^2 # assume uniform mesh in all directions
    end
    ps.u .= (ps.A \ ps.rhs)
    return nothing
end