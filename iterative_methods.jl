function jacobi!(x, b; K=100, ϵ=1e-10)
    N  = length(x)
    xi = copy(x)
    xj = similar(x)
    
    for k = 1:K
        @inbounds xj[1] = (xi[2]   + xi[N] + b[1]) / 2.0
        @inbounds xj[N] = (xi[N-1] + xi[1] + b[N]) / 2.0
        for n = 2:N-1
            @inbounds xj[n] = (xi[n-1] + xi[n+1] + b[n]) / 2.0
        end

        xi, xj = xj, xi
    end

    x.= (xi .+ xj .- mean(xi) .- mean(xj)) ./ 2.0
    return nothing
end
