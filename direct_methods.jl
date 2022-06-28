function thomas!(φ, ρ)
    N = size(ρ, 1)
    A = [ 1.0 for i=1:N]
    B = [-2.0 for i=1:N]
    C = [ 1.0 for i=1:N]
    f = similar(ρ)
    w = similar(ρ)
    g = similar(ρ)

    for i=2:N-1
      f[i] = ρ[i]
    end

    f[2]   -= φ[1]
    f[N-1] -= φ[N]
    w[2] = C[2]/B[2]
    g[2] = f[2]/B[2]

    for i=3:N-1
        D    = (B[i] - A[i] * w[i-1])
        g[i] = (f[i] - A[i] * g[i-1]) / D
        w[i] =                   C[i] / D
    end

    φ[N-1] = g[N-1]
    for i=reverse(2:N-2)
      φ[i] = g[i] - w[i] * φ[i+1]
    end
    
    return nothing
end
