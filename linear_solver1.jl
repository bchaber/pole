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

    ps = LinearSolver((nx,), (Δx,), (δx,), h, A, b, similar(b), similar(b), dof, (left, right))

    cartesian!(ps)
    return ps
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

function cylindrical!(ps::LinearSolver{1, T}, rf) where {T}
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