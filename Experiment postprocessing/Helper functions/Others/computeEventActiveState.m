function eventInds = computeEventActiveState(events, weights)
    eventInds = cell(size(weights.w, 2), 1);  % numShelves x 1
% 	eventInds = false(size(weights.w, 1), size(weights.w, 2));  % numShelves x t
	for i = 1:length(events)
        eventInds{events(i).shelf}(end+1,:) = false(1, size(weights.w, 1));  % 1 x t
		eventInds{events(i).shelf}(end,events(i).nB:events(i).nE) = true;
	end
end
