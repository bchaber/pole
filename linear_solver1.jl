function init(xf, left, right)
    Δx = diff(xf)
    nx = length(Δx)
    xc = cumsum(Δx) .- 0.5Δx
    δx = diff([first(xf); xc; last(xf)])
    @assert minimum(Δx) ≈ maximum(Δx) "Currently only uniform cells are supported :("

    N  = nx
    A  = spzeros(N, N)
    b  = zeros(N)
    dof = collect(reshape(1:N, nx, 1))

    return LinearSolver((nx,), (Δx,), (δx,), first(Δx)^2, A, b, similar(b), similar(b), dof, (left, right))
end

function LinearSolver(xf; left=DirichletBC(), right=DirichletBC())
    ps = init(xf, left, right)
    cartesian!(ps)
    return ps
end

function LinearSolver(cs::Val{:x}, xf; left=DirichletBC(), right=DirichletBC())
    ps = init(xf, left, right)
    cartesian!(ps)
    return ps
end

function LinearSolver(cs::Val{:r}, rf; left=DirichletBC(), right=DirichletBC())
    rc = [0.5rf[i] + 0.5rf[i-1] for i=2:length(rf)]
    ps = init(rf, left, right)
    cartesian!(ps)
    radial!(ps, rc)
    return ps
end

function cartesian!(ps::LinearSolver{1, T}) where {T}
    left, right = ps.bcs
    dof = ps.dof
    nx, = ps.n
    A  = ps.A
    b  = ps.b
    α  = 1.0

    stencil = Stencil()
    fwd = Val(:+)
    rev = Val(:-)

    @inbounds for i = 2:nx-1
        n = dof[i-1]
        m = dof[i+1]
        o = dof[i]
        cartesian!(A, stencil, fwd, α, o, n)
        cartesian!(A, stencil, fwd, α, o, m)
    end

    @inbounds for i = 1
        n = dof[nx]
        m = dof[i+1]
        o = dof[i]
        cartesian!(A, left,    fwd, α, o, m, n)
        cartesian!(b, left,    fwd, α, o)
        cartesian!(A, stencil, fwd, α, o, m)
    end

    @inbounds for i = nx
        n = dof[i-1]
        m = dof[1]
        o = dof[i]
        cartesian!(A, stencil, fwd, α, o, n)
        cartesian!(A, right,   fwd, α, o, n, m)
        cartesian!(b, right,   fwd, α, o)
    end
end

function cylindrical!(ps::LinearSolver{1, T}, rf) where {T}
    left, right = ps.bcs
    dof = ps.dof
    nr, = ps.n
    Δr, = ps.Δ
    A  = ps.A
    b  = ps.b

    stencil = Stencil()
    fwd = Val(:+)
    rev = Val(:-)

    @inbounds for i = 2:nr-1
        α = 0.5rc[i]
        n = dof[i-1]
        m = dof[i+1]
        o = dof[i]
        radial!(A, stencil, rev, α, o, n)
        radial!(A, stencil, fwd, α, o, m)
    end
    
    @inbounds for i = 1
        α = 0.5rc[i]
        n = dof[nr]
        m = dof[i+1]
        o = dof[i]
        radial!(A, left,    rev, α, o, m, n)
        radial!(b, left,    rev, α, o)
        radial!(A, stencil, fwd, α, o, m)
    end
    
    @inbounds for i = nr
        α = 0.5rc[i]
        n = dof[i-1]
        m = dof[1]
        o = dof[i]
        radial!(A, stencil, rev, α, o, n)
        radial!(A, right,   fwd, α, o, n, m)
        radial!(b, right,   fwd, α, o)
    end
    return nothing
end