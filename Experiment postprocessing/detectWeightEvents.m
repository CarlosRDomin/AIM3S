function [events, weightsPerBin, weightsPerShelf] = detectWeightEvents(weights, binWidth, systemParams)
	if nargin<3 || isempty(systemParams), systemParams = struct('epsVar',100, 'epsMean',10, 'Nb',60, 'Ne',60, 'movAvgWindowInSamples',90); end
	events = struct('t',{}, 'nB',{}, 'nE',{}, 'deltaW',{}, 'bins',{}, 'shelf',{});
	
	% Aggregate plates based on plateWidth
    weightsPerBin = aggregateWeightsIntoBins(weights, binWidth, systemParams);
	weightsPerShelf = aggregateWeightsIntoBins(weights, [], systemParams);
	
	% Detect events
	wVarIsActive = (weightsPerShelf.wVar > systemParams.epsVar);
	stateChanges = diff(wVarIsActive);
	for i = 1:size(wVarIsActive,2)
		stateChangeInds = find(stateChanges(:,i))';  % Convert to row vector
		stateLengths = diff([0 stateChangeInds length(wVarIsActive)]);
		activeInds = (2:2:length(stateLengths));  % wVarIsActive(1,:) is always 0 (since with a window of length 1, w-wMean=0) -> All even indices are times when the state was active (variance above threshold)
		stableInds = (3:2:length(stateLengths));
		validActiveIntervals = find(stateLengths(activeInds) > systemParams.Nb);  % E.g. 2nd, 3rd and 6th activeInds last longer than Nb
		validStableIntervals = find(stateLengths(stableInds) > systemParams.Ne);
		minNextActiveInterval = 0;
		for activeIdx = validActiveIntervals
			if activeIdx <= minNextActiveInterval, continue; end  % If it was active, then stable for not long enough, then active again, we should ignore this second active interval (as it is still part of the same event)
			
			% Find the end of the event
			stableIdx = validStableIntervals(find(validStableIntervals>=activeIdx, 1));  % Find the first stable interval that's valid (long enough) and is AFTER this active interval
			if isempty(stableIdx), break; end  % No more events -> DONE! :)
			
			% Determine the event timing
			nB = stateChangeInds(activeInds(activeIdx)-1) - systemParams.movAvgWindowInSamples/2;  % Last point with variance below threshold (subtract 1 because n-th interval starts at the (n-1)th change)
			nE = stateChangeInds(stableInds(stableIdx)-1) + 1 + systemParams.Ne;  % First point with variance below threshold + Ne samples
			wB = weightsPerShelf.wMean(nB, i);
			wE = weightsPerShelf.wMean(nE, i);
			deltaW = wE - wB;
            if abs(deltaW) < systemParams.epsMean, continue; end  % Ignore events with smaller weight change than epsMean
			if deltaW > 0, t = nB; else, t = nE; end  % Trigger time: beginning for putbacks, end for pickups
			
			% Determine contributing bins
			bins = find(abs(weightsPerBin.wMean(nE,i,:) - weightsPerBin.wMean(nB,i,:)) > systemParams.epsMean)';
			
			% Fill in new event entry
			events(end+1) = struct('t',t, 'nB',nB, 'nE',nE, 'deltaW',deltaW, 'bins',bins, 'shelf',i);
			
			% Make sure we don't analyze events that start before the end of this current one (would happen if it was active, then stable for not long enough, then active again...)
			minNextActiveInterval = stableIdx;
		end
	end
end

