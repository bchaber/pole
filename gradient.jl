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

function (∇::GradientOperator)(u; result=nothing, reverse=false)
    dudx = isnothing(result) ? zeros(∇.nx + 1) : result
    
    calculate_gradient!(dudx, u, ∇.δx, ∇.nx)
    for (bnd, bc) in zip(∇.bnds, ∇.bcs)
        calculate_gradient!(dudx, u, ∇.δx, ∇.nx, bnd, bc)
    end
    
    if reverse dudx .*= -1.0 end
    return dudx
end