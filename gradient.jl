struct GradientOperator{T <: Tuple}
    nx :: Int64
    δx :: Vector{Float64}
    bcs :: T
end

struct GradientOperator2{T <: Tuple}
    nx :: Int64
    ny :: Int64
    δx :: Vector{Float64}
    δy :: Vector{Float64}
    bcs :: T
end

@inline gradx(u, δx, i, bnd::Type{Val{:Left}},  bc::PeriodicBC) = (first(u) - last(u)) / (first(δx) + last(δx))
@inline gradx(u, δx, i, bnd::Type{Val{:Right}}, bc::PeriodicBC) = (first(u) - last(u)) / (first(δx) + last(δx))
@inline grady(u, δy, j, bnd::Type{Val{:Lower}}, bc::PeriodicBC) = (first(u) - last(u)) / (first(δy) + last(δy))
@inline grady(u, δy, j, bnd::Type{Val{:Upper}}, bc::PeriodicBC) = (first(u) - last(u)) / (first(δy) + last(δy))

@inline gradx(u, δx, i, bnd::Type{Val{:Left}},  bc::DirichletBC) = (first(u) - bc.value) / first(δx)
@inline gradx(u, δx, i, bnd::Type{Val{:Right}}, bc::DirichletBC) = (bc.value -  last(u)) / last(δx)
@inline grady(u, δy, j, bnd::Type{Val{:Lower}}, bc::DirichletBC) = (first(u) - bc.value) / first(δy)
@inline grady(u, δy, j, bnd::Type{Val{:Upper}}, bc::DirichletBC) = (bc.value -  last(u)) / last(δy)

@inline gradx(u, δx, i, bnd::Type{Val{:Left}},  bc::NeumannBC) = bc.value
@inline gradx(u, δx, i, bnd::Type{Val{:Right}}, bc::NeumannBC) = bc.value
@inline grady(u, δy, j, bnd::Type{Val{:Lower}}, bc::NeumannBC) = bc.value
@inline grady(u, δy, j, bnd::Type{Val{:Upper}}, bc::NeumannBC) = bc.value

@inline gradx(u, δx, i, j) = (u[i,j] - u[i-1,j]) / δx[i]
@inline grady(u, δy, i, j) = (u[i,j] - u[i,j-1]) / δy[j]

function (∇::GradientOperator)(u; result=nothing)
    ∇u = isnothing(result) ? zeros(SVector{3, eltype(u)}, ∇.nx + 1) : result
    left,right = ∇.bcs
    dx, dy, dz = 0.0, 0.0, 0.0
    nx = ∇.nx
    # boundaries
    @inbounds for i=1
        ux = @SVector [u[1], u[nx]]
        dx = gradx(ux, ∇.δx, i, Val{:Left},   left)
        ∇u[i] = @SVector [dx, dy, dz]
    end

    @inbounds for i=nx+1
        ux = @SVector [u[1], u[nx]]
        dx = gradx(ux, ∇.δx, i, Val{:Right}, right)
        ∇u[i] = @SVector [dx, dy, dz]
    end
    # interior
    @inbounds for i=2:nx, j=1
        dx = gradx(u, ∇.δx, i, j)
        ∇u[i] = @SVector [dx, dy, dz]
    end
    
    return ∇u
end

function (∇::GradientOperator2)(u; result=nothing)
    ∇u = isnothing(result) ? zeros(SVector{3, eltype(u)}, ∇.nx + 1, ∇.ny + 1) : result
    left, right, upper, lower = ∇.bcs
    dx, dy, dz = 0.0, 0.0, 0.0
    nx, ny = ∇.nx, ∇.ny
    
    # corners
    @inbounds for i=1, j=1
        ux = @SVector [u[1,1], u[nx,1]]
        uy = @SVector [u[1,1], u[1,ny]]
        dx = gradx(ux, ∇.δx, i, Val{:Left},   left)
        dy = grady(uy, ∇.δy, j, Val{:Lower}, lower)
        ∇u[i,j] = @SVector [dx, dy, dz]
    end

    @inbounds for i=1, j=ny+1
        ux = @SVector [u[1,ny], u[nx,ny]]
        uy = @SVector [u[1, 1], u[ 1,ny]]
        dx = gradx(ux, ∇.δx, i, Val{:Left},   left)
        dy = grady(uy, ∇.δy, j, Val{:Upper}, upper)
        ∇u[i,j] = @SVector [dx, dy, dz]
    end

    @inbounds for i=nx+1, j=1
        ux = @SVector [u[1,ny], u[nx,ny]]
        uy = @SVector [u[nx,1], u[nx,ny]]
        dx = gradx(ux, ∇.δx, i, Val{:Right}, right)
        dy = grady(uy, ∇.δy, j, Val{:Lower}, lower)
        ∇u[i,j] = @SVector [dx, dy, dz]
    end

    @inbounds for i=nx+1, j=ny+1
        ux = @SVector [u[1,ny], u[nx,ny]]
        uy = @SVector [u[nx,1], u[nx,ny]]
        dx = gradx(ux, ∇.δx, i, Val{:Right}, right)
        dy = grady(uy, ∇.δy, j, Val{:Lower}, lower)
        ∇u[i,j] = @SVector [dx, dy, dz]
    end
    # sides
    @inbounds for i=1, j=2:ny
        ux = @SVector [u[1,1], u[nx,1]]
        dx = gradx(ux, ∇.δx, i, Val{:Left},   left)
        dy = grady(u,  ∇.δy, i, j)
        ∇u[i,j] = @SVector [dx, dy, dz]
    end

    @inbounds for i=nx+1, j=2:ny
        ux = @SVector [u[1,1], u[nx,1]]
        dx = gradx(ux, ∇.δx, nx, Val{:Right}, right)
        dy = grady(u,  ∇.δy, nx, j)
        ∇u[i,j] = @SVector [dx, dy, dz]
    end

    @inbounds for i=2:nx, j=1
        uy = @SVector [u[i,1], u[i,ny]]
        dx = gradx(u,  ∇.δx, i, j)
        dy = grady(uy, ∇.δy, j, Val{:Lower}, lower)
        ∇u[i,j] = @SVector [dx, dy, dz]
    end

    @inbounds for i=2:nx, j=ny+1
        uy = @SVector [u[i,1], u[i,ny]]
        dx = gradx(u,  ∇.δx, i, ny)
        dy = grady(uy, ∇.δy, ny, Val{:Upper}, upper)
        ∇u[i,j] = @SVector [dx, dy, dz]
    end
    # interior
    @inbounds for i=2:nx, j=2:ny
        dx = gradx(u, ∇.δx, i, j)
        dy = grady(u, ∇.δy, i, j)
        ∇u[i,j] = @SVector [dx, dy, dz]
    end
    
    return ∇u
end
