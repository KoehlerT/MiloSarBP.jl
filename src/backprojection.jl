import LinearAlgebra
import Interpolations
import FFTW
import DSP
import IterTools
import Distances

function rangecompression(data, BC, config)	
	M,N = size(data) # N Number of Ramps (Slow Time); M Fast time

	NFFT = nextpow(2, M);
	fn = BC[:F_sample]/2;
	df = BC[:F_sample]/NFFT;

	f_steps = 0:df:fn-df;
	range   = abs.((3e8*BC[:T])/(2*BC[:B])*f_steps);
	
	data = data .* DSP.hann(M)

	data = vcat(data, zeros(NFFT - M, N))

	fftData = FFTW.fft(data,1)
	if BC[:Flipped]
		fftData = fftData[Int(NFFT/2)+1:end,:]
	else
		fftData = fftData[1:Int(NFFT/2),:]
	end
	range = range .- config.range_offset
	return range, fftData
end

@inline function phases(dR, BC)
	dt = dR./3e8; # TODO: Better c0
	return sign(BC[:K])*(2*pi*BC[:F_start]*dt - 1*pi*BC[:K]*dt.^2);
end

function components_for_pixel(data, location, config, BC)
	data_matrix = reduce(hcat, data.HH);
	range,fftData = rangecompression(data_matrix, BC, config);
	interp_fftData_r = Vector{Interpolations.Extrapolation}(undef, size(fftData, 2))
	interp_fftData_i = Vector{Interpolations.Extrapolation}(undef, size(fftData, 2))

	# push!(interp_fftData, Interpolations.LinearInterpolation(2 .* range, col, extrapolation_bc=Interpolations.Line()))
	ranges_range = range(minimum(2*range_m), maximum(2*range_m), length(range_m))
	for (i, col) in enumerate(eachcol(fftData))
		interp_fftData_r[i] = Interpolations.cubic_spline_interpolation(ranges_range, real.(col))
		interp_fftData_i[i] = Interpolations.cubic_spline_interpolation(ranges_range, imag.(col))
	end
	
	positions = data.Position;
	components = Vector{ComplexF64}(undef, length(positions))

	for (idx, position) in enumerate(positions)
		dR = 2* LinearAlgebra.norm(location - position)
		ph = phases(dR, BC)
		# fft_val = interp_fftData_r[idx](dR) + im*interp_fftData_i[idx](dR);
		fft_val = complex.(interp_fftData_r[idx].(dR), interp_fftData_i[idx].(dR)); # 271s (Dierckx) #22s (Interpolations)
		
		to_add = fft_val * exp(-im * ph)
		
		components[idx] = to_add;
	end
	return components
end

function backprojection(data, area, config, BC)
	data_matrix = reduce(hcat, data.HH);
	range,fftData = rangecompression(data_matrix, BC, config);

	# Alternative Interpolation
	# push!(interp_fftData_r, Dierckx.Spline1D(2 .* range, real.(col), k=3, s=0, bc="nearest"))
	# push!(interp_fftData_i, Dierckx.Spline1D(2 .* range, imag.(col), k=3, s=0, bc="nearest"))
	
	interp_fftData_r = Vector{Interpolations.Extrapolation}(undef, size(fftData, 2))
	interp_fftData_i = Vector{Interpolations.Extrapolation}(undef, size(fftData, 2))

	# push!(interp_fftData, Interpolations.LinearInterpolation(2 .* range, col, extrapolation_bc=Interpolations.Line()))
	ranges_range = range(minimum(2*range_m), maximum(2*range_m), length(range_m))
	for (i, col) in enumerate(eachcol(fftData))
		interp_fftData_r[i] = Interpolations.cubic_spline_interpolation(ranges_range, real.(col))
		interp_fftData_i[i] = Interpolations.cubic_spline_interpolation(ranges_range, imag.(col))
	end

	image :: Matrix{ComplexF32} = zeros(size(area,1), size(area,2));
	# phases = []
	positions = data.Position;

	Threads.@threads for i in 1:size(image,1)
		for j in 1:size(image,2)
			ground_pos = area[i,j]
			# ground_pos = [2.609612991848897e+05, 6.220806084691286e+06, 49.407781901388730]
			for (idx, position) in enumerate(positions)
				dR = 2* LinearAlgebra.norm(ground_pos - position)
				ph = phases(dR, BC)
				# fft_val = interp_fftData[idx](dR);
				fft_val = complex.(interp_fftData_r[idx].(dR), interp_fftData_i[idx].(dR)); # 271s (Dierckx) #22s (Interpolations)
				to_add = fft_val * exp(-im * ph)
				
				image[i,j] = image[i,j] + to_add;
				# push!(phases, angle(to_add))# angle(to_add))
			end
		end
	end
	return image;
end

function backprojection_vec(data, area, config, BC, nthreads=16)
	data_matrix = reduce(hcat, data.HH);
	range_m,fftData = rangecompression(data_matrix, BC, config);

	interp_fftData_r = Vector{Interpolations.Extrapolation}(undef, size(fftData, 2))
	interp_fftData_i = Vector{Interpolations.Extrapolation}(undef, size(fftData, 2))

	# push!(interp_fftData, Interpolations.LinearInterpolation(2 .* range, col, extrapolation_bc=Interpolations.Line()))
	ranges_range = range(minimum(2*range_m), maximum(2*range_m), length(range_m))
	for (i, col) in enumerate(eachcol(fftData))
		interp_fftData_r[i] = Interpolations.cubic_spline_interpolation(ranges_range, real.(col))
		interp_fftData_i[i] = Interpolations.cubic_spline_interpolation(ranges_range, imag.(col))
	end

	positions = data.Position;
	area_matrix = hcat(area...);
	chunks = collect(IterTools.Iterators.partition(1:length(positions),Int(ceil(length(positions)/nthreads))));
	images = [zeros(ComplexF32, size(area_matrix,2)) for _ in 1:length(chunks)];

	k_val = BC[:K]
	sign_k = sign(k_val)
	f_start = BC[:F_start]

	Threads.@threads for chunk_idx in 1:length(chunks) 
		
		for idx in chunks[chunk_idx]
			dR = 2 .* Distances.colwise(Distances.Euclidean(), area_matrix, positions[idx]) #14s
			dt = dR./3e8;
			ph = sign_k .* (2π * f_start .* dt .- π * k_val .* (dt .* dt))
			
			fft_val = complex.(interp_fftData_r[idx].(dR), interp_fftData_i[idx].(dR)); # 271s (Dierckx) #22s (Interpolations)
			
			to_add = fft_val .* exp.(-im .* ph) # 17ns
			images[chunk_idx] = images[chunk_idx] .+ to_add; #3.5ns
		end
	end
	image_vec = reduce(+, images)
	image = reshape(image_vec, size(area))
	return image;
end


function backprojection_vec_tiled(data, area, config, BC)
	data_matrix = reduce(hcat, data.HH);
	range,fftData = rangecompression(data_matrix, BC, config);
	image :: Matrix{ComplexF32} = zeros(size(area,1), size(area,2));
	positions = data.Position;
	handles = []
	cols_per_thread = Int(floor(size(image,1) / Threads.nthreads()))
	for i in 1:cols_per_thread:size(image,1)
		start_idx = i;
		end_idx = min(i+cols_per_thread-1,size(image,1))

		handle = Threads.@spawn compute_tile!(
			view(image, start_idx:end_idx, :), 
			range, 
			fftData, 
			positions, 
			view(area, start_idx:end_idx, :), 
			BC
		)
		push!(handles, handle)
	end
	# wait for all tasks to finish
	for h in handles
		wait(h)
	end
	return image;
end

function compute_tile!(subimage, range, rangecompressed, positions, subarea, BC)
	interp_fftData_r = Vector{Interpolations.Extrapolation}(undef, size(fftData, 2))
	interp_fftData_i = Vector{Interpolations.Extrapolation}(undef, size(fftData, 2))

	# push!(interp_fftData, Interpolations.LinearInterpolation(2 .* range, col, extrapolation_bc=Interpolations.Line()))
	ranges_range = range(minimum(2*range_m), maximum(2*range_m), length(range_m))
	for (i, col) in enumerate(eachcol(fftData))
		interp_fftData_r[i] = Interpolations.cubic_spline_interpolation(ranges_range, real.(col))
		interp_fftData_i[i] = Interpolations.cubic_spline_interpolation(ranges_range, imag.(col))
	end

	dR = similar(subarea, Float64)
	# chunk = similar(subimage, ComplexF32)
	fill!(subimage, 0)
	for (idx, pos) in enumerate(positions)
		# fill!(dR, 0);
		for i in eachindex(subarea)
			dR[i] = 2 * LinearAlgebra.norm(subarea[i] .- pos)
		end
		ph = phases(dR, BC)
		fft_val = interp_fftData_r[idx].(dR) + im*interp_fftData_i[idx].(dR);
		to_add = fft_val .* exp.(-im .* ph)
		
		subimage .= subimage .+ to_add;
	end
	# subimage .= chunk;
end