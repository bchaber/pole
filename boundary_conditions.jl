abstract type BoundaryCondition end
struct PeriodicBC <: BoundaryCondition end
mutable struct DirichletBC <: BoundaryCondition value :: Float64 end
mutable struct NeumannBC   <: BoundaryCondition value :: Float64 end

DirichletBC() = DirichletBC(0.0)
NeumannBC() = NeumannBC(0.0)