# MiloSar Backprojection
Package specifically created for the MiloSar System. For system details see:

```text
Jordan, D., et al. "Development of the miloSAR testbed for the one kilogramme radioCamera SAR for small drones." 2019 IEEE Radar Conference (RadarConf). IEEE, 2019.
```

This project is meant to get one started in getting images focused using Backprojection. It should lead to a more fleshed out, Backprojection library. Minimum getting started:

```julia
import MiloSarBP
import PyPlot


config = MiloSarBP.Config(
	# Data Input Directory
	"PATH_TO_FLIGHT/23_06_21_12_30_14",
	# Range Offset due to system specifics
	14.7
);

# Read the radar data
# data: Dataframe with Timestamp, Position, Orientation and Raw Radar Signal
# bandconfig: Configruation of the radar parameters (ramp steepness etc.)
data, bandconfig = MiloSarBP.read(config);

# Area in UTM Coordinates
area = MiloSarBP.generate_position_grid(
	# Bottom Left
	[2.609400501848896e+05, 6220761.085691286, 50.417764027515660], 
	# Top Right
	[2.609948638294746e+05, 6220815.581316797, 49.323503814433934],
	# Resolution
	0.05
)

# Generate the Image. Only use part of the data, where the area is very visible
image = MiloSarBP.backprojection_vec(data[1140:5530], area, config, bandconfig);

scr, peak, clutter = MiloSarBP.Quality.signal_clutter_ratio(image);
println("SCR: $scr dB")

plt = MiloSarBP.Plot.backprojection(image, area)
PyPlot.clim(clutter-0.5, peak)
# PyPlot.gcf() # When in jupyter notebooks
```

## Development
The package `Revise` should be used when developing the package. E.g.
```julia
import Pkg
# Activate the environment of the package which is developed, so you use this version as working code and not the published package
Pkg.activate("path/to/MiloSarBP-jl/")
using Revise
import MiloSarBP

# When editing functions inside MiloSarBP, the package will be recompiled
```

## Getting Started
- Download [Julia](https://julialang.org/downloads/)
- [Getting Started](https://docs.julialang.org/en/v1/manual/getting-started/)