function [events, weights] = detectWeightEvents(weightsOrig, plateWidth, systemParams, experimentInfo)
	if nargin<3 || isempty(systemParams), systemParams = struct('epsVar',100, 'epsMean',10, 'Nb',60, 'Ne',60, 'movAvgWindowInSamples',90); end
	if nargin<4, experimentInfo = []; end
	events = struct('t',{}, 'nB',{}, 'nE',{}, 'deltaW',{}, 'weightIDs',{}, 'plates',{}, 'shelf',{});
	
	% Aggregate plates based on plateWidth
	startInds = 1:plateWidth:size(weightsOrig.w,2);
	newSize = size(weightsOrig.w); newSize(2) = length(startInds);
	weights = struct('t',weightsOrig.t, 'w',zeros(newSize), 'wMean',zeros(newSize), 'wVar',zeros(newSize));
	weightIDs = cell(newSize(1:2));
	for i = 1:length(startInds)
		inds = startInds(i)+(0:plateWidth-1);
		weights.w(:,i,:) = sum(weightsOrig.w(:,inds,:), 2);
		
		if isempty(experimentInfo), continue; end
		for j = 1:size(weightIDs,1), weightIDs{j,i} = experimentInfo.weightIDs(j,inds); end
	end
	weights = computeWeightMeanAndVar(weights, systemParams);
	weightsPerShelf = aggregateWeightsPerShelf(weights, systemParams);
	
	% Detect events
	wVarIsActive = (weightsPerShelf.wVar > systemParams.epsVar);
	stateChanges = diff(wVarIsActive, 1,ndims(wVarIsActive));
	for i = 1:size(wVarIsActive,1)
		stateChangeInds = find(squeeze(stateChanges(i,:)));
		stateLengths = diff([0 stateChangeInds size(wVarIsActive,ndims(wVarIsActive))]);
		activeInds = (2:2:length(stateLengths));  % wVarIsActive(:,1) is always 0 (since with a window of length 1, w-wMean=0) -> All even indices are times when the state was active (variance above threshold)
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
			wB = weightsPerShelf.wMean(i, nB);
			wE = weightsPerShelf.wMean(i, nE);
			deltaW = wE - wB;
			if deltaW > 0, t = nB; else, t = nE; end  % Trigger time: beginning for putbacks, end for pickups
			
			% Determine contributing plates
			plates = find(abs(weights.wMean(i,:,nE) - weights.wMean(i,:,nB)) > systemParams.epsMean);
			eventWeightIDs = [weightIDs{i,plates}];
			
			% Fill in new event entry
			events(end+1) = struct('t',t, 'nB',nB, 'nE',nE, 'deltaW',deltaW, 'weightIDs',{eventWeightIDs}, 'plates',plates, 'shelf',i);
			
			% Make sure we don't analyze events that start before the end of this current one (would happen if it was active, then stable for not long enough, then active again...)
			minNextActiveInterval = stableIdx;
		end
	end
end

