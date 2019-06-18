function [events, weightsPerBin, weightsPerShelf] = detectWeightEvents(weights, binWidth, systemParams)
	if nargin<3 || isempty(systemParams), systemParams = struct('movAvgWindowInSamples',60, 'epsVar',500, 'epsMeanShelf',10, 'epsMeanPlate',5, 'N_high',30, 'N_low',30); end
	events = struct('tB',{}, 'tE',{}, 'nB',{}, 'nE',{}, 'deltaW',{}, 'bins',{}, 'shelf',{});
	
	% Aggregate plates based on plateWidth
    weightsPerBin = aggregateWeightsIntoBins(weights, binWidth, systemParams);
	weightsPerShelf = aggregateWeightsIntoBins(weights, [], systemParams);
	
	% Detect events
	wVarIsActive = (weightsPerShelf.wVar > systemParams.epsVar);
	stateChanges = diff(wVarIsActive);
	for iShelf = 1:size(wVarIsActive,2)
		stateChangeInds = find(stateChanges(:,iShelf))';  % Convert to row vector
		stateLengths = diff([0 stateChangeInds length(wVarIsActive)]);
		activeInds = (2:2:length(stateLengths));  % wVarIsActive(1,:) is always 0 (since with a window of length 1, w-wMean=0) -> All even indices are times when the state was active (variance above threshold)
		stableInds = (3:2:length(stateLengths));
		validActiveIntervals = find(stateLengths(activeInds) > systemParams.N_high);  % E.g. 2nd, 3rd and 6th activeInds last longer than Nb
		validStableIntervals = find(stateLengths(stableInds) > systemParams.N_low);
		minNextActiveInterval = 0;
		for activeIdx = validActiveIntervals
			if activeIdx <= minNextActiveInterval, continue; end  % If it was active, then stable for not long enough, then active again, we should ignore this second active interval (as it is still part of the same event)
			
			% Find the end of the event
			stableIdx = validStableIntervals(find(validStableIntervals>=activeIdx, 1));  % Find the first stable interval that's valid (long enough) and is AFTER this active interval
			if isempty(stableIdx), break; end  % No more events -> DONE! :)
			
			% Determine the event timing
			nB = stateChangeInds(activeInds(activeIdx)-1) - systemParams.N_low;  % Last point with variance below threshold (subtract 1 because n-th interval starts at the (n-1)th change) - N_low samples
			nE = stateChangeInds(stableInds(stableIdx)-1) + 1 + systemParams.N_low;  % First point with variance below threshold + N_low samples
			wB = weightsPerShelf.wMean(nB, iShelf);
			wE = weightsPerShelf.wMean(nE, iShelf);
			deltaW = wE - wB;
            
            % Keep the event if the weight change is larger than epsMeanShelf
            if abs(deltaW) > systemParams.epsMeanShelf
                %if deltaW > 0, t = nB; else, t = nE; end  % Trigger time: beginning for putbacks, end for pickups
                tB = weightsPerShelf.t(nB);
                tE = weightsPerShelf.t(nE);

                % Determine contributing bins
                bins = cell(length(binWidth),1);
                for iBinW = 1:length(binWidth)
                    bins{iBinW} = find(abs(diff(weightsPerBin(iBinW).wMean([nB nE],iShelf,:))) > systemParams.epsMeanPlate)';
                end

                % Fill in the new event entry
                events(end+1,1) = struct('tB',tB, 'tE',tE, 'nB',nB, 'nE',nE, 'deltaW',deltaW, 'bins',{bins}, 'shelf',iShelf);
            end
			
			% Make sure we don't analyze events that start before the end of this current one (would happen if it was active, then stable for not long enough, then active again...)
			minNextActiveInterval = stableIdx;
		end
	end
end

