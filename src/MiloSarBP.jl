module MiloSarBP

struct Config
	path :: String;
	range_offset :: Union{Nothing, <:Real}
end

include("helper.jl")
include("read.jl")
include("backprojection.jl")
include("Plot.jl")
include("Quality.jl")

export Config, read, generate_position_grid, backprojection_vec
end