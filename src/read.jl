function read(config::Config)
    data, BC = MiloSar.read(config)
	return data, BC
end

module MiloSar
import IniFile
import JSON
import DataFrames
import Proj
import Interpolations
import Rotations
import LinearAlgebra
import DSP
import ..MiloSarBP
function read(config::MiloSarBP.Config)
    return read_radar(config.path)
end

function read_radar(folder::String)
    function get_file(folder::String, suffix::String)
        files = readdir(folder)
        file_idx = findfirst(f -> endswith(f, suffix), files)
        if isnothing(file_idx)
            error("Could not find file ending with $suffix in $folder")
        end
        return files[file_idx]
    end
    binfile = get_file(folder, ".bin")
    localizationfile = get_file(folder, ".json")
    summaryfile = get_file(folder, "summary.ini")

    if (isnothing(binfile) || isnothing(localizationfile) || isnothing(summaryfile))
        error("Could not find the correct files in the folder! Radar Data: $binfile Localozation: $localizationfile Summary: $summaryfile")
    end

    # Read Localozation Data
    localization = read_localization(joinpath(folder, localizationfile))
    # Read ini file for meta information
    BC, channel_size = read_bandconfig(joinpath(folder, summaryfile))
    # Read raw radar data
    ch1a, ch2a, ch1b, ch2b = read_radar_data(joinpath(folder, binfile), BC, channel_size)

	# Merge into one dataframe
	sample_times = LinRange(localization[1,:Time], localization[end,:Time], size(ch1a, 2))

	df = interpolate_positions(localization, sample_times);
    df_orientations = interpolate_rotations(localization, sample_times)
    df.Orientation = df_orientations.Orientations;
	df.VV = [vec(col) for col in eachcol(ch1a)];
	df.VH = [vec(col) for col in eachcol(ch1b)];
	df.HV = [vec(col) for col in eachcol(ch2a)];
	df.HH = [vec(col) for col in eachcol(ch2b)];
    return df, BC
end

function interpolate_positions(dataframe, interpo_times)
	times = dataframe.Time;
	positions = dataframe.Frame_Pos;

	xs = [p[1] for p in positions]
	ys = [p[2] for p in positions]
	zs = [p[3] for p in positions]

	# Create 1D linear interpolants for each coordinate
	interp_x = Interpolations.LinearInterpolation(times, xs, extrapolation_bc=Interpolations.Line())
	interp_y = Interpolations.LinearInterpolation(times, ys, extrapolation_bc=Interpolations.Line())
	interp_z = Interpolations.LinearInterpolation(times, zs, extrapolation_bc=Interpolations.Line())

	# Interpolate at new time values
	new_positions = [ [interp_x(t), interp_y(t), interp_z(t)] for t in interpo_times ]

	# Create new DataFrame
	return DataFrames.DataFrame(Time = interpo_times, Position = new_positions)
end

function interpolate_rotations(dataframe, interpo_times)
	times = dataframe.Time;
	orientations = dataframe.Frame_Orientations;

    new_orientations = Vector{Rotations.UnitQuaternion}(undef, length(interpo_times))
    for (i, time) in enumerate(interpo_times)
        if (i ==1 || i == length(interpo_times))
            continue;
        end
        idx = argmin(abs.(times.-time))
        delta = Int(sign(time - times[idx])) # positive if after times[idx]
        if (delta < 0)
            new_orientations[i] = slerp(orientations[idx-1],orientations[idx], (time-times[idx-1])/(times[idx]-times[idx-1]))
        else
            new_orientations[i] = slerp(orientations[idx],orientations[idx+1], (time-times[idx])/(times[idx+1]-times[idx]))
        end
    end
    new_orientations[1] = new_orientations[2];
    new_orientations[end] = new_orientations[end-1];

	# Create new DataFrame
	return DataFrames.DataFrame(Time = interpo_times, Orientations=new_orientations)
end

function slerp(q1::Rotations.UnitQuaternion, q2::Rotations.UnitQuaternion, t::Float64)
    # Ensure shortest path
    dotprod = q1.q.s * q2.q.s + LinearAlgebra.dot([q1.q.v1,q1.q.v2,q1.q.v3],[q2.q.v1,q2.q.v2,q2.q.v3])
    if dotprod < 0.0
        q2 = Rotations.UnitQuaternion(-q2.q)  # Negate to take shorter arc
        dotprod = -dotprod
    end

    # If the quaternions are very close, use linear interpolation
    if dotprod > 0.9995
        q = LinearAlgebra.normalize((1 - t) * q1.q + t * q2.q)
        return Rotations.UnitQuaternion(q)
    end

    # Slerp proper
    θ = acos(dotprod)
    sinθ = sin(θ)
    w1 = sin((1 - t) * θ) / sinθ
    w2 = sin(t * θ) / sinθ
    q = w1 * q1.q + w2 * q2.q
    return Rotations.UnitQuaternion(q)
end

function read_bandconfig(summaryfile::String)
    summary_contents = IniFile.read(IniFile.Inifile(), summaryfile)
    function readint(section::String, key::String)
        return parse(Int, IniFile.get(summary_contents, section, key, nothing))
    end
    up_ramp_length = readint("tx_synth", "up_ramp_length")
    integration_start = readint("integration", "start_index")
    integration_stop = readint("integration", "end_index")
    fn = readint("tx_synth", "fractional_numerator")
    up_ramp_increment = readint("tx_synth", "up_ramp_increment")
    n_seconds = readint("dataset", "n_seconds")
    prf = readint("dataset", "prf")
    n_pris = readint("integration", "n_pris")

	function get_rf_freq(fn)
        FD = 2^24 - 1;
        f = 125e6*(75 + fn/FD)/4;
		return f
    end

    BC = Dict{Symbol,Any}()
    BC[:F_start] = get_rf_freq(fn)
    BC[:F_stop] = get_rf_freq(up_ramp_increment * up_ramp_length + fn)
    BC[:M] = integration_stop - integration_start + 1
    BC[:F_sample] = 3.125e6
    BC[:T] = up_ramp_length * 1 / 125e6
    BC[:B] = BC[:F_stop] - BC[:F_start]
    BC[:K] = BC[:B] / BC[:T]
    BC[:Flipped] = true

    channel_size = BC[:M] * Int((n_seconds * prf) / n_pris)

    return BC, channel_size
end

function read_radar_data(binfile::String, BC, channel_size)
    file = open(binfile, "r") do io
        bindata = read!(io, Vector{Int16}(undef, filesize(io) ÷ sizeof(Int16)))
        return bindata
    end

    # Check if file is empty
    isempty(file) && error("ValueError: Binary file is empty.")

    # Ensure data length is a multiple of 4
    bindata_len = length(file)
    remainder = bindata_len % 4
    if remainder != 0
        file = file[1:end-remainder]
    end

    # Convert Data Vector to Matrix
    # chan_a = ComplexF32[]
    # chan_b = ComplexF32[]

    # for i in 1:4:length(file)
    # 	push!(chan_a, complex(file[i+1], file[i]))  # I + jQ
    # 	push!(chan_b, complex(file[i+3], file[i+2]))
    # end

    # Optional: Alternatively, use vectorized approach
    bindata = file
    chan_a = bindata[2:4:end] .+ im * bindata[1:4:end]
    chan_b = bindata[4:4:end] .+ im * bindata[3:4:end]

    # adc scaling. Sould maybe be omitted due to Int -> Float
    # But increased SCR to a looot?? like 10dB -> 21dB?
    chan_a = chan_a .* 0.00012207031;
    chan_b = chan_b .* 0.00012207031;

    # Resize to channel_size
    chan_a = chan_a[1:channel_size]
    chan_b = chan_b[1:channel_size]

    # Reshape to matrix: chan_a_shaped = reshape(chan_a, BC[:M], :)
    chan_a_shaped = reshape(chan_a, BC[:M], :)
    chan_b_shaped = reshape(chan_b, BC[:M], :)

    ch1a = chan_a_shaped[:, 1:2:end]
    ch2a = chan_a_shaped[:, 2:2:end]
    ch1b = chan_b_shaped[:, 1:2:end]
    ch2b = chan_b_shaped[:, 2:2:end]

    # Shift by remaining frequency frequency offset
    # TODO: fix!
    f_shift = (17.998e-6) * (BC[:B] / BC[:T]) - 7.5e6;
    ch1a = frequency_shift!(ch1a, BC, f_shift)
    ch2a = frequency_shift!(ch2a, BC, f_shift)
    ch1b = frequency_shift!(ch1b, BC, f_shift)
    ch2b = frequency_shift!(ch2b, BC, f_shift)

    return ch1a, ch2a, ch1b, ch2b
end

function frequency_shift!(data, BC, f_shift)
	# Constants from the BC Dict
	B = BC[:B]
	T = BC[:T]
	fa = BC[:F_sample]
	
	# Dimensions
	M,N = size(data)
	
	# Time vector
	t = collect(0:M-1) .* (T / M)  # t is a Vector{Float64} of size M
	
	# Downconversion matrix
	shift_carrier = exp.(1im .* 2π .* f_shift .* t)  # Vector of length M	f_downconvert_matrix = repeat(f_downconvert, 1, N)
	
	# Mix signal with complex exponential
	data = data .* shift_carrier
    return data; # Omit filtering becaue it takes a lot of time (and makes no difference)
	
	# Design high-pass FIR filter
	lpFilt = DSP.digitalfilter(DSP.Highpass(2 * 400e3 / fa), DSP.FIRWindow(DSP.hamming(201)))
	
	# Apply filter to each column
	for i in 1:N
		data[:, i] = DSP.filt(lpFilt, data[:, i])
	end	
	return data # signal_bb_filtered
end

function latlon2utm(lat::Real, lon::Real, alt::Real=0.0)
    # Determine UTM zone from longitude
    zone = floor(Int, (lon + 180) / 6) + 1
    hemisphere = lat >= 0 ? "north" : "south"

    # Build Proj4 string for the specific UTM zone and hemisphere
    proj_string = "+proj=utm +zone=$(zone) +datum=WGS84 +units=m +no_defs " *
                  (hemisphere == "south" ? "+south" : "")

    # Define projection
    pj = Proj.Transformation("EPSG:4326", proj_string)

    # Forward projection: (lat, lon) to (Easting, Northing)
    E, N = pj(lat, lon)

    return E, N, alt  # Return altitude unchanged
end

function read_localization(localizationfile::String)
    times = Float64[]
    easts = Float64[]
    norths = Float64[]
    ups = Float64[]

    eastings = Float64[]
    northings = Float64[]
    uppings = Float64[]

    open(localizationfile, "r") do io
        for tline in eachline(io)
            val = JSON.parse(tline)

            push!(times, val["timestamp"] / 1000)  # convert ms to seconds

            lat = val["lat"]
            lon = val["lon"]
            alt = val["alt"]
            E, N, h = latlon2utm(lat, lon, alt)  # You need to define or import this

            push!(easts, E)
            push!(norths, N)
            push!(ups, h)

            push!(eastings, val["e"])
            push!(northings, val["n"])
            push!(uppings, val["u"])
        end
    end

	frame_positions = [
		transpose(eastings .+ easts[1]) ;
		transpose(northings .+ norths[1]);
		transpose(uppings .- uppings[1] .+ ups[1]);
	]

	frame_positions = transpose(frame_positions)
    # each row depicts a xyz position

    # Milosar does not have a compass. Derive the orientation from the current flight
    orientations = Vector{Any}(undef, size(frame_positions,1))
    for i in 1:length(orientations)-1
        direction = frame_positions[i+1,:] - frame_positions[i,:]
        orientations[i] = MiloSarBP.quaternion_between( vec([1., 0., 0.]), vec(direction))
    end
    orientations[end] = orientations[end-1]
    # Antennas are oriented rotated 90° to the left
    # for rot in orientations
    #     rot = Rotations.UnitQuaternion(rot * Rotations.RotZ(π/2))
    # end
    # orientation_antenna = orientations .* Rotations.RotZ(π/2); #Rotations.UnitQuaternion()

	vec_of_vecs = [vec(row) for row in eachrow(frame_positions)]

	return DataFrames.DataFrame(
		Time = times,
		Frame_Pos = vec_of_vecs,
        Frame_Orientations = orientations
	);
end


end # Module MiloSar