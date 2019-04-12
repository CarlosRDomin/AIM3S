function eventInds = computeEventActiveState(events, weights)
	eventInds = false(size(weights.w, 1), size(weights.w, ndims(weights.w)));  % numShelves x t
	for i = 1:length(events)
		eventInds(events(i).shelf, events(i).nB:events(i).nE) = true;
	end
end

