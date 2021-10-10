function jacobi!(x, b; K=100, ϵ=1e-10, residuals=Float64[])
    nx = length(x)
    xi = copy(x)
    xj = similar(x)
    dx = similar(x)
    
    it = K
    for k = 1:K
        dx .= abs.(xi .- xj)
        last_norm = sum(dx); push!(residuals, last_norm)
        @inbounds xj[1] =              (xi[2]       + xi[nx]    + b[1])      / 2.0
        @inbounds xj[2:nx-1] .= @views (xi[1:nx-2] .+ xi[3:nx] .+ b[2:nx-1])./ 2.0
        @inbounds xj[nx]=              (xi[nx-1]    + xi[1]     + b[nx])     / 2.0

        xi, xj = xj, xi
        
        dx .= abs.(xi .- xj)
        if abs(sum(dx) - last_norm) < ϵ
            it = k
            break
        end
    end
    
    x .= 0.5 .* (xi .+ xj .- mean(xi) .- mean(xj))
    return it
end