struct GradientOperator{T <: Tuple} # 1D, only x-coord
    nx :: Int64
    δx :: Vector{Float64}
    
    bcs :: T
    bnds :: Tuple{Symbol, Symbol}
end

calculate_gradient!(dudx, u, δx, n, boundary, bc::PeriodicBC) = begin
    if boundary == :left  @inbounds dudx[1]   = @views (u[1] - u[n]) / (first(δx) + last(δx)) end
    if boundary == :right @inbounds dudx[n+1] = @views (u[1] - u[n]) / (first(δx) + last(δx)) end
end

calculate_gradient!(dudx, u, δx, n, boundary, bc::DirichletBC) = begin
    if boundary == :left  @inbounds dudx[1]   = @views (u[1] - bc.value)  / first(δx) end
    if boundary == :right @inbounds dudx[n+1] = @views (bc.value - u[n]) / last(δx) end
end

calculate_gradient!(dudx, u, δx, n) = @inbounds dudx[2:n] .= @views (u[2:n] .- u[1:n-1]) ./ δx[2:n]