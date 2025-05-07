import DataFrames
import Rotations

"""
    generate_position_grid(p1::Vector{<:Real}, p2::Vector{<:Real}, resolution::Real)

Generate a grid of 3D points between two positions (p1 and p2) with the given resolution.
Each element in the returned Matrix is a 3-element vector `[x, y, z]`.

The grid spans from p1 to p2 across X and Y, interpolating Z linearly.
"""
function generate_position_grid(p1::Vector{<:Real}, p2::Vector{<:Real}, resolution::Real)
    # Extract coordinates
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    # Create ranges
    xs = x1:resolution:x2
    ys = y1:resolution:y2

    # Determine grid size
    nx = length(xs)
    ny = length(ys)

    # Initialize the matrix of positions
    grid = Matrix{Vector{Float64}}(undef, ny, nx)

    # Fill the grid
    for j in 1:ny
        for i in 1:nx
            x = xs[i]
            y = ys[j]
            # Linear interpolation of z based on relative x,y (optional: could just fix z)
            tx = (x - x1) / (x2 - x1 + eps())  # Avoid division by zero
            ty = (y - y1) / (y2 - y1 + eps())
            t = (tx + ty) / 2
            z = (1 - t) * z1 + t * z2
            grid[j, i] = [x, y, z]
        end
    end

    return grid
end

function get_antenna_angles_to_center(area, data)
    center = area[Int(floor(size(area,1)/2)), Int(floor(size(area,2)/2))];
    θs = Vector{Float64}(undef, DataFrames.nrow(data))
    ϕs = Vector{Float64}(undef, DataFrames.nrow(data))
    rs = Vector{Float64}(undef, DataFrames.nrow(data))
    for (idx, row) in enumerate(eachrow(data))
        direction = row.Position - center
        current_orientation = row.Orientation
        distance = LinearAlgebra.norm(direction)
        # orientation is with respect to x-axis. Rotate it back to get the uav direction
        # Antennas are rotated 90° in the x axis
        dir_antenna  = current_orientation * Rotations.RotZ(π/2) * [1., 0., 0.]
        
        θ = π/2 - acos(abs(direction[3]) / LinearAlgebra.norm(direction))
        ϕ = (atan(dir_antenna[2]/ dir_antenna[1]) - atan(direction[2] / direction[1]))
        ϕ = mod(ϕ + π, 2π) - π
        θs[idx] = θ
        ϕs[idx] = ϕ
        rs[idx] = distance
    end
    return ϕs, θs, rs
end

# ::Union{DataFrame, SubDataFrame, DataFrameRow}
function find_relevant_samples(area, data , maxrange :: Real, ϕ_bounds :: Vector{<:Real}, θ_bounds :: Vector{<:Real})
    indices = Vector{Bool}(undef, DataFrames.nrow(data))
    ϕs, θs, rs = get_antenna_angles_to_center(area, data);
    @. indices = (rs < maxrange) & 
        (abs(ϕs) >= ϕ_bounds[1]) & (abs(ϕs) < ϕ_bounds[2]) & 
        (abs(θs) >= θ_bounds[1]) & (abs(θs) < θ_bounds[2])
    return indices
end

function quaternion_between(v1::AbstractVector, v2::AbstractVector)
    v1 = LinearAlgebra.normalize(v1)
    v2 = LinearAlgebra.normalize(v2)
    dot_prod = LinearAlgebra.dot(v1, v2)

    if isapprox(dot_prod, 1.0; atol=1e-6)
        # Vectors are the same; no rotation needed
        return Rotations.UnitQuaternion(1.0, 0.0, 0.0, 0.0)
    elseif isapprox(dot_prod, -1.0; atol=1e-6)
        # Vectors are opposite; find an orthogonal axis
        axis = LinearAlgebra.normalize(LinearAlgebra.cross(v1, [1.0, 0.0, 0.0]))
        if LinearAlgebra.norm(axis) < 1e-6
            axis = LinearAlgebra.normalize(LinearAlgebra.cross(v1, [0.0, 1.0, 0.0]))
        end
        return Rotations.UnitQuaternion(Rotations.AngleAxis(π, axis))
    else
        axis = LinearAlgebra.cross(v1, v2)
        s = sqrt((1 + dot_prod) * 2)
        x, y, z = axis / s
        w = s / 2
        return Rotations.UnitQuaternion(w, x, y, z)
    end
end