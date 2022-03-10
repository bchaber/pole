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
    ps = LinearSolver((nx,ny), (Δx,Δy), (δx,δy), h, A, b, similar(b), similar(b), dof, (left, right, upper, lower))

    cartesian!(ps)
    return ps
end

function cartesian!(ps::LinearSolver{2, T}) where {T}
    left, right, upper, lower = ps.bcs
    dof = ps.dof
    nx,ny = ps.n
    A  = ps.A
    b  = ps.b

    stencil = Stencil()
    fwd = Val(:+)
    rev = Val(:-)

    @inbounds for i = 2:nx-1, j = 1:ny
        n = dof[i-1,j]
        m = dof[i+1,j]
        o = dof[i,  j]
        cartesian!(A, stencil, fwd, o, n)
        cartesian!(A, stencil, fwd, o, m)
    end

    @inbounds for i = 1, j = 1:ny
        n = dof[nx, j]
        m = dof[i+1,j]
        o = dof[i,  j]
        cartesian!(A, left,    fwd, o, m, n)
        cartesian!(b, left,    fwd, o)
        cartesian!(A, stencil, fwd, o, m)
    end

    @inbounds for i = nx, j = 1:ny
        n = dof[i-1,j]
        m = dof[1,  j]
        o = dof[i,  j]
        cartesian!(A, stencil, fwd, o, n)
        cartesian!(A, right,   fwd, o, n, m)
        cartesian!(b, right,   fwd, o)
    end

    @inbounds for i = 1:nx, j = 2:ny-1
        n = dof[i,j-1]
        m = dof[i,j+1]
        o = dof[i,j]
        cartesian!(A, stencil, fwd, o, n)
        cartesian!(A, stencil, fwd, o, m)
    end

    @inbounds for i = 1:nx, j = 1
        n = dof[i, ny]
        m = dof[i,j+1]
        o = dof[i,j]
        cartesian!(A, lower,   fwd, o, m, n)
        cartesian!(b, lower,   fwd, o)
        cartesian!(A, stencil, fwd, o, m)
    end

    @inbounds for i = 1:nx, j = ny
        n = dof[i,j-1]
        m = dof[i,1]
        o = dof[i,j]
        cartesian!(A, stencil, fwd, o, n)
        cartesian!(A, upper,   fwd, o, n, m)
        cartesian!(b, upper,   fwd, o)
    end
end

function cylindrical!(ps::LinearSolver{2, T}, rf) where {T}
    left, right, upper, lower = ps.bcs
    dof = ps.dof
    nz,nr = ps.n
    A  = ps.A
    b  = ps.b

    stencil = Stencil()
    fwd = Val(:+)
    rev = Val(:-)

    @inbounds for i = 1:nz, j = 2:nr-1
        n = dof[i,j-1]
        m = dof[i,j+1]
        o = dof[i,j]
        radial!(A, stencil, rev, h/2rf[j],   o, n)
        radial!(A, stencil, fwd, h/2rf[j+1], o, m)
    end

    @inbounds for i = 1:nz, j = 1
        n = dof[i,nr]
        m = dof[i,j+1]
        o = dof[i,j]
        radial!(A, left,    rev, h/2rf[j], o, m, n)
        radial!(b, left,    rev, h/2rf[j], o)
        radial!(A, stencil, fwd, h/2rf[j+1], o, m)
    end

    @inbounds for i = 1:nz, j = nr
        n = dof[i,j-1]
        m = dof[i,1]
        o = dof[i,j]
        radial!(A, stencil, rev, h/2rf[j], o, n)
        radial!(A, right,   fwd, h/2rf[j+1], o, n, m)
        radial!(b, right,   fwd, h/2rf[j+1], o)
    end
    return nothing
end
