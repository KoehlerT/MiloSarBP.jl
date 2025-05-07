module Plot
import ..MiloSarBP
import LinearAlgebra
using PyPlot
using StatsBase
using Printf

function rangecompression(range, fft)
	clf()
	plt = pcolormesh(1:size(fft,2), range, 10*log10.(abs.(fft)))
	colorbar()
	return plt
end

function rangecompression(data, BC, config, polarization, references :: Vector{Vector{Float64}})
	range, fft = MiloSarBP.rangecompression(reduce(hcat, data[!, polarization]), BC, config)
	clf()
	# times = data.Time .- minimum(data.Time)
	plt = pcolormesh(1:size(fft,2), range, 10*log10.(abs.(fft)))
	for reference in references 
		plot(1:size(fft,2), [LinearAlgebra.norm(x - reference) for x in data.Position])
	end
	colorbar()
	ylabel("range (m)")
	xlabel("data (idx)")
	return plt
end

function area_edges(area)
	xes = [p[1] for p in area[1,:]]
	ys = [p[2] for p in area[:,1]]
	xes, ys
end

function find_axes(area)
	xes, ys = area_edges(area)
	(xes .- minimum(xes)), (ys .- minimum(ys))
end

function backprojection(image, area)
	PyPlot.clf()
	xs, ys = find_axes(area)
	plt = pcolormesh(xs, ys, 10 .*log10.(abs.(image)))
	xlabel("x (east) [m]")
	ylabel("y (north) [m]")
	colorbar()
	return plt;
end

function SCR(image)
	magnitudes = 10 .* log10.(abs.(image .+ eps()))
	peak = maximum(magnitudes)

	nbins = 50  # Adjust depending on your data
	h = fit(Histogram, magnitudes[:], range(minimum(magnitudes),peak,50))
	counts = h.weights
    edges = h.edges[1]

	maxIdx = argmax(h.weights)
	clutter = (h.edges[1][maxIdx] + h.edges[1][maxIdx + 1]) / 2
	scr = peak - clutter

	clf()
	bin_centers = (edges[1:end-1] + edges[2:end]) ./ 2
    bar(bin_centers, counts, width=edges[2]-edges[1], align="center", alpha=0.7)
    axvline(clutter, color="red", linestyle="--", label="Clutter level")
    text(clutter, maximum(counts)*0.9,
         @sprintf("Clutter = %.2f dB", clutter),
         rotation=90, color="red", va="top", ha="right")
	
	axvline(edges[end], color="green", linestyle="--", label="Peak level")
	text(edges[end], maximum(counts)*0.9,
         @sprintf("Maximum = %.2f dB", peak),
         rotation=90, color="green", va="top", ha="right")

    title(@sprintf("SCR = %.2f dB", scr))
    xlabel("Magnitude (dB)")
    ylabel("Count")
    legend()
    grid(true)
end

function phases(components)
	angles = angle.(components)
	# SnareEvaluation.Descriptors.unwrap!(angles)
	angles = rad2deg.(angles)
# y_label = range(minimum(angles),maximum(angles),50) / π * 180
	clf()
	plt = plot(angles, linewidth=0.5)
	ylabel("Phase [°]")
	xlabel("Sample [idx]")
	return plt
end


export backprojection

end