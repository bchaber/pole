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

function solve!(ps::LinearSolver{D, T}, ρ) where {D, T}
    @inbounds for i in eachindex(ρ)
      ps.rhs[i] = ps.b[i] - ρ[i] * ps.h^2 # assume uniform mesh in all directions
    end
    ps.u .= (ps.A \ ps.rhs)
    return nothing
end

@inline function fixed!(ps::LinearSolver{2, T}, value, i, j) where {T}
    dof = ps.dof[i,j]
    @assert ps.b[dof] == 0. "You are going to reset a degree of freedom that has b[i] != 0"
    ps.A[dof, :]  .= 0.
    ps.A[dof, dof] = 1.
    ps.b[dof]      = value
    return nothing
end

# Cartesian coordinate system
@inline function cartesian!(A::SparseMatrixCSC{Float64, Int64}, bc::Stencil, dir::Forward, i, j)
    A[i, i] += 1.; A[i, j] -= 1.
end
@inline function cartesian!(A::SparseMatrixCSC{Float64, Int64}, bc::Stencil, dir::Reverse, i, j)
    A[i, i] -= 1.; A[i, j] += 1.
end
@inline function cartesian!(A::SparseMatrixCSC{Float64, Int64}, bc::PeriodicBC, dir::Forward, i, _, k)
    A[i, i] += 1.; A[i, k] -= 1.
end
@inline function cartesian!(A::SparseMatrixCSC{Float64, Int64}, bc::PeriodicBC, dir::Reverse, i, _, k)
    A[i, i] -= 1.; A[i, k] += 1.
end
@inline function cartesian!(b::Vector{Float64}, bc::PeriodicBC, dir::Forward, i) end
@inline function cartesian!(b::Vector{Float64}, bc::PeriodicBC, dir::Reverse, i) end
@inline function cartesian!(A::SparseMatrixCSC{Float64, Int64}, bc::NeumannBC, dir::Forward, i, j, _) end
@inline function cartesian!(A::SparseMatrixCSC{Float64, Int64}, bc::NeumannBC, dir::Reverse, i, j, _) end
@inline function cartesian!(b::Vector{Float64}, bc::NeumannBC, dir::Forward, i) b[i] += bc.value end
@inline function cartesian!(b::Vector{Float64}, bc::NeumannBC, dir::Reverse, i) b[i] -= bc.value end
@inline function cartesian!(A::SparseMatrixCSC{Float64, Int64}, bc::DirichletBC, dir::Forward, i, j, _)
    A[i, i] += 3.; A[i, j] -= 1/3
end
@inline function cartesian!(A::SparseMatrixCSC{Float64, Int64}, bc::DirichletBC, dir::Reverse, i, j, _)
    A[i, i] -= 3.; A[i, j] += 1/3
end
@inline function cartesian!(b::Vector{Float64}, bc::DirichletBC, dir::Forward, i)
    b[i] += (8/3)bc.value
end
@inline function cartesian!(b::Vector{Float64}, bc::DirichletBC, dir::Reverse, i)
    b[i] -= (8/3)bc.value
end
# Cylindrical coordinate coordinate system
@inline function radial!(A::SparseMatrixCSC{Float64, Int64}, bc::Stencil, dir::Forward, α, i, j)
    A[i, i] += α; A[i, j] -= α
end
@inline function radial!(A::SparseMatrixCSC{Float64, Int64}, bc::Stencil, dir::Reverse, α, i, j)
    A[i, i] -= α; A[i, j] += α
end
@inline function radial!(A::SparseMatrixCSC{Float64, Int64}, bc::NeumannBC, dir::Forward, α, i, j, _) end
@inline function radial!(A::SparseMatrixCSC{Float64, Int64}, bc::NeumannBC, dir::Reverse, α, i, j, _) end
@inline function radial!(b::Vector{Float64}, bc::NeumannBC, dir::Forward, α, i) b[i] += α*bc.value end
@inline function radial!(b::Vector{Float64}, bc::NeumannBC, dir::Reverse, α, i) b[i] -= α*bc.value end
@inline function radial!(A::SparseMatrixCSC{Float64, Int64}, bc::DirichletBC, dir::Forward, α, i, j, _)
    A[i, i] += 3.0α; A[i, j] -= α/3.0
end
@inline function radial!(A::SparseMatrixCSC{Float64, Int64}, bc::DirichletBC, dir::Reverse, α, i, j, _)
    A[i, i] -= 3.0α; A[i, j] += α/3.0
end
@inline function radial!(b::Vector{Float64}, bc::DirichletBC, dir::Forward, α, i)
    b[i] += α*(8/3)bc.value
end
@inline function radial!(b::Vector{Float64}, bc::DirichletBC, dir::Reverse, α, i)
    b[i] -= α*(8/3)bc.value
end