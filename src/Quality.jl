module Quality
import ..MiloSarBP
using StatsBase

function signal_clutter_ratio(image)
	magnitudes = 10 .* log10.(abs.(image .+ eps()))
	peak = maximum(magnitudes)

	nbins = 50  # Adjust depending on your data
	h = fit(Histogram, magnitudes[:], range(minimum(magnitudes),peak,50))

	maxIdx = argmax(h.weights)
	clutter = (h.edges[1][maxIdx] + h.edges[1][maxIdx + 1]) / 2
	scr = peak - clutter
	return scr, peak, clutter
end

# calculates the first (mean), second (σ) and fourth (kurtosis) circular moments
function circular_moments(angles::AbstractVector{<:Real})
    C = mean(cos.(angles))
    S = mean(sin.(angles))
    mean_angle = atan(S, C)

    kurt = mean(cos.(2 .* (angles .- mean_angle)))
	R = sqrt(C^2 + S^2)
    σ = sqrt(-2 * log(R))
    return mean_angle, σ, kurt
end

function circular_diff(angles :: AbstractVector{<:Real})
	dθ = diff(angles)
    return mod.(dθ .+ π, 2π) .- π
end

function windowed_phase_steadyness(phases, size=50)
	halfsize = Int(floor(size/2))
	mean = similar(phases)
	std = similar(phases)
	kurt = similar(phases)
	for i in eachindex(phases)
		end_idx = min(length(phases), i + halfsize - 1)
		start_idx = max(1, i - halfsize + 1)
		subvec = view(phases, start_idx:end_idx)
		mean[i], std[i], kurt[i] = circular_moments(subvec)
	end
	return mean, std, kurt
end

function windowed_mean_and_std(phases, size=50)
	halfsize = Int(floor(size/2))
	mean = similar(phases)
	std = similar(phases)
	for i in eachindex(phases)
		end_idx = min(length(phases), i + halfsize - 1)
		start_idx = max(1, i - halfsize + 1)
		subvec = view(phases, start_idx:end_idx)
		mean[i], std[i] = StatsBase.mean_and_std(subvec)
	end
	return mean, std
end

function surroundings(image,area, position::Vector{Float64}, padding=1)
    # minx = area[1][1,:]
    # miny = area[2][:,1]
    area = area
    x_idx = [argmin(abs.([p[1] - (position[1] - padding) for p in area[1,:]])),
        argmin(abs.([p[1] - (position[1] + padding) for p in area[1,:]]))]
    y_idx = [argmin(abs.([p[2] - (position[2] - padding) for p in area[:,1]])),
        argmin(abs.([p[2] - (position[2] + padding) for p in area[:,1]]))]

    return @views image[y_idx[1]:y_idx[2], x_idx[1]:x_idx[2]]
end

function target_deviation(image, area, references :: Vector{Vector{Float64}})
	offset_cum = [0, 0]
	search_area = 10
	for position in references
		# Extract values in a 5m range
		region = surroundings(image,area, position, search_area);
		amplitudes = abs.(region);
		maxidx = argmax(amplitudes);
		offset = [i/2 for i in size(region)] - [maxidx[1],maxidx[2]]
		offset_cum += offset;
	end
	average = offset_cum / length(references)
	return [average[2], average[1]];
end

import BetaML
# Calculates phase drift in two clusters (usually there is an increase and decrease)
# Returns the average distance of those clusters. Lower is better
function phase_drift(phases :: AbstractArray{<:Real}, window = 15)
	mean_phase, _, _ = windowed_phase_steadyness(phases, window)
	differentiated = circular_diff(mean_phase)
	mean_diff, std = windowed_mean_and_std(differentiated, 2*window)
	stable_idx = std .< 0.5;

	model = BetaML.KMeansClusterer(n_classes=2)
	BetaML.fit!(model, mean_diff[stable_idx])
	values = BetaML.parameters(model).representatives;
	return mean(abs.(values));
end

function get_quality_measures(data, BC,config, references, ranges, padding = [5,5,0], resolution=0.1)
	cum_amps = 0;
	cum_drift = 0;
	target_deviation = 0;

	# make a area from all areas of interest concatenated. So only one call to backprojection_vec
	area_all = reduce(hcat, [MiloSarBP.generate_position_grid(
			reference - padding, 
			reference + padding,
			resolution
	) for reference in references])
	
	# Initialize a matrix where all maxima across ranges to optimize are saved
	reference_locations = zeros(length(ranges), length(references), 3)

	for (range_idx, range) in enumerate(ranges)
		images = MiloSarBP.backprojection_vec(data[range,:], area_all, config, BC);
		b = Int(size(images,2)/length(references));

		for (image, area, reference_idx) in [ (view(images,:,i*b+1:(i+1)*b),view(area_all,:,i*b+1:(i+1)*b),i+1 ) for i in 0:length(references)-1]
			# Where is the reference (corner reflector)
			maximum_location = area[argmax(abs.(image))]
			reference_locations[range_idx, reference_idx, :] = maximum_location;

			# Find the phase drift at this pixel
			components = MiloSarBP.components_for_pixel(data[ranges[1],:], maximum_location, config, BC)
			cum_drift += phase_drift(angle.(components))

			# Save the amplitude
			amplitude = 10*log10(maximum(abs.(image))) 
			cum_amps = cum_amps + amplitude; # This is mathematical bs
		end
	end

	# Look at the standard deviation of every targets coordinate. Average it across all references across all coordinates
	average_std_per_coord = StatsBase.mean([StatsBase.std(reference_locations[:,ref,:], dims=1) for ref in 1:length(references)])
	target_deviation = StatsBase.mean(average_std_per_coord)

	offsets = [StatsBase.mean(reference_locations[:,ref,:]' .- references[ref],dims=2) for ref in 1:length(references)]
	average_offset = StatsBase.mean(offsets);

	# 'Normalize the drift and amplitudes'
	cum_amps /= length(references) * length(ranges)
	cum_drift /= length(references) * length(ranges)

	return cum_amps, cum_drift, target_deviation, average_offset;
end

export signal_clutter_ratio, get_quality_measures, phase_drift, target_deviation

end