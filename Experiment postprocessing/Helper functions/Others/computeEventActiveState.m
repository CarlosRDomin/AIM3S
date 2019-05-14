function eventInds = computeEventActiveState(events, weights)
	eventInds = false(size(weights.w, 1), size(weights.w, 2));  % numShelves x t
	for i = 1:length(events)
		eventInds(events(i).nB:events(i).nE, events(i).shelf) = true;
	end
end

