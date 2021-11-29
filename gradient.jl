struct GradientOperator{T <: Tuple} # 1D, only x-coord
    nx :: Int64
    δx :: Vector{Float64}
    
    bcs :: T
    bnds :: Tuple{Symbol, Symbol}
end

@inline gradx(u, δx, i, bnd::Type{Val{:Left}},  bc::PeriodicBC) = (first(u) - last(u)) / (first(δx) + last(δx))
@inline gradx(u, δx, i, bnd::Type{Val{:Right}}, bc::PeriodicBC) = (first(u) - last(u)) / (first(δx) + last(δx))

@inline gradx(u, δx, i, bnd::Type{Val{:Left}},  bc::DirichletBC) = (first(u) - bc.value) / first(δx)
@inline gradx(u, δx, i, bnd::Type{Val{:Right}}, bc::DirichletBC) = (bc.value -  last(u)) / last(δx)

@inline gradx(u, δx, i, j) = (u[i,j] - u[i-1,j]) / δx[i]
@inline grady(u, δy, i, j) = (u[i,j] - u[i,j-1]) / δy[j]

function (∇::GradientOperator)(u::Vector; result=nothing)
    ∇u = isnothing(result) ? zeros(SVector{3, eltype(u)}, ∇.nx + 1) : result
    n  = ∇.nx
    left,right = ∇.bcs
    dx, dy, dz = 0.0, 0.0, 0.0
    for i=1
        dx = gradx(u, ∇.δx, i, Val{:Left},   left)
        ∇u[i] = @SVector [dx, dy, dz]
    end
    for i=n+1
        dx = gradx(u, ∇.δx, i, Val{:Right}, right)
        ∇u[i] = @SVector [dx, dy, dz]
    end
    for i=2:n, j=1
        dx = gradx(u, ∇.δx, i, j)
        ∇u[i] = @SVector [dx, dy, dz]
    end
    
    return ∇u
end