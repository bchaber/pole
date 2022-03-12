function init(xf, yf, left, right, upper, lower)
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
    return LinearSolver((nx,ny), (Δx,Δy), (δx,δy), h, A, b, similar(b), similar(b), dof, (left, right, upper, lower))
end

function LinearSolver(xf, yf; left=DirichletBC(), right=DirichletBC(), upper=DirichletBC(), lower=DirichletBC())
    ps = init(xf, yf, left, right, upper, lower)
    cartesian!(ps)
    return ps
end

function LinearSolver(cs::Val{:xy}, xf, yf; left=DirichletBC(), right=DirichletBC(), upper=DirichletBC(), lower=DirichletBC())
    ps = init(xf, yf, left, right, upper, lower)
    cartesian!(ps)
    return ps
end

function LinearSolver(cs::Val{:zr}, zf, rf; left=DirichletBC(), right=DirichletBC(), upper=NeumannBC(), lower=NeumannBC())
    rc = [0.5rf[i] + 0.5rf[i-1] for i=2:length(rf)]
    ps = init(zf, rf, left, right, upper, lower)
    cartesian!(ps)
    radial!(ps, rc)
    return ps
end

function LinearSolver(cs::Val{:θr}, xf, yf; left=PeriodicBC(), right=PeriodicBC(), upper=DirichletBC(), lower=DirichletBC())
    rc = [0.5rf[i] + 0.5rf[i-1] for i=2:length(rf)]
    ps = init(xf, yf, left, right, upper, lower)
    polar!(ps, rc)
    radial!(ps, rc)
    return ps
end

function polar!(ps::LinearSolver{2, T}, rc) where {T}
    left, right, upper, lower = ps.bcs
    dof = ps.dof
    nt,nr = ps.n
    A  = ps.A
    b  = ps.b
    Δr = ps.h

    stencil = Stencil()
    fwd = Val(:+)
    rev = Val(:-)

    @inbounds for i = 2:nt-1, j = 1:nr
        n = dof[i-1,j]
        m = dof[i+1,j]
        o = dof[i,  j]
        polar!(A, stencil, fwd, 1.0/rc[j], o, n)
        polar!(A, stencil, fwd, 1.0/rc[j], o, m)
    end

    @inbounds for i = 1, j = 1:nr
        n = dof[nt, j]
        m = dof[i+1,j]
        o = dof[i,  j]
        polar!(A, left,    fwd, 1.0/rc[j], o, m, n)
        polar!(b, left,    fwd, 1.0/rc[j], o)
        polar!(A, stencil, fwd, 1.0/rc[j], o, m)
    end

    @inbounds for i = nt, j = 1:nr
        n = dof[i-1,j]
        m = dof[1,  j]
        o = dof[i,  j]
        polar!(A, stencil, fwd, 1.0/rc[j], o, n)
        polar!(A, right,   fwd, 1.0/rc[j], o, n, m)
        polar!(b, right,   fwd, 1.0/rc[j], o)
    end

    @inbounds for i = 1:nt, j = 2:nr-1
        n = dof[i,j-1]
        m = dof[i,j+1]
        o = dof[i,j]
        cartesian!(A, stencil, fwd, o, n)
        cartesian!(A, stencil, fwd, o, m)
    end

    @inbounds for i = 1:nt, j = 1
        n = dof[i, nr]
        m = dof[i,j+1]
        o = dof[i,j]
        cartesian!(A, lower,   fwd, o, m, n)
        cartesian!(b, lower,   fwd, o)
        cartesian!(A, stencil, fwd, o, m)
    end

    @inbounds for i = 1:nt, j = nr
        n = dof[i,j-1]
        m = dof[i,1]
        o = dof[i,j]
        cartesian!(A, stencil, fwd, o, n)
        cartesian!(A, upper,   fwd, o, n, m)
        cartesian!(b, upper,   fwd, o)
    end
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

function radial!(ps::LinearSolver{2, T}, rc) where {T}
    left, right, upper, lower = ps.bcs
    dof = ps.dof
    nz,nr = ps.n
    Δr = ps.h
    A  = ps.A
    b  = ps.b

    stencil = Stencil()
    fwd = Val(:+)
    rev = Val(:-)

    @inbounds for i = 1:nz, j = 2:nr-1
        n = dof[i,j-1]
        m = dof[i,j+1]
        o = dof[i,j]
        radial!(A, stencil, rev, Δr/2rc[j],   o, n)
        radial!(A, stencil, fwd, Δr/2rc[j], o, m)
    end

if first(rc) > Δr/2
    @inbounds for i = 1:nz, j = 1
        n = dof[i,nr]
        m = dof[i,j+1]
        o = dof[i,j]
        radial!(A, left,    rev, Δr/2rc[j], o, m, n)
        radial!(b, left,    rev, Δr/2rc[j], o)
        radial!(A, stencil, fwd, Δr/2rc[j], o, m)
    end
end
    @inbounds for i = 1:nz, j = nr
        n = dof[i,j-1]
        m = dof[i,1]
        o = dof[i,j]
        radial!(A, stencil, rev, Δr/2rc[j], o, n)
        radial!(A, right,   fwd, Δr/2rc[j], o, n, m)
        radial!(b, right,   fwd, Δr/2rc[j], o)
    end
    return nothing
end
