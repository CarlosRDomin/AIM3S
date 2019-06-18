function [events, weightsPerBin, weightsPerShelf] = gt2events(gt, weights, binWidth, systemParams, marginEvent)
    if nargin<4 || isempty(systemParams), systemParams = struct('movAvgWindowInSamples',60, 'epsMeanPlate',5, 'N_high',30, 'N_low',30); end
    if nargin<5 || isempty(marginEvent), marginEvent = systemParams.N_low*diff(weights.t(1:2)); end
    if length(marginEvent)==1, marginEvent = repmat(marginEvent, 1,2); end  % Just in case, leave a bit of room on either side of the event tStart and tEnd so we get a better deltaW estimate
    slidingWinDelay = diff(weights.t(1:2)) * systemParams.movAvgWindowInSamples/2;  % Compensate for the sliding window filter
	events = struct('tB',cell(length(gt.ground_truth),1), 'tE',[], 'nB',[], 'nE',[], 'deltaW',[], 'bins',[], 'shelf',[]);
	
	% Aggregate plates based on plateWidth
    weightsPerBin = aggregateWeightsIntoBins(weights, binWidth, systemParams);
	weightsPerShelf = aggregateWeightsIntoBins(weights, [], systemParams);
    
    % Perform our calculations based on gt's t_start and t_end
    for iEv = 1:length(events)
        tB = parseStrDatetime(gt.ground_truth(iEv).t_start) - slidingWinDelay - marginEvent(1);
        tE = parseStrDatetime(gt.ground_truth(iEv).t_end) + slidingWinDelay + marginEvent(2);
        nB = find(weights.t >= tB, 1);
        nE = find(weights.t >= tE, 1); %if isempty(nE), nE = length(weights.t); end
        [~, iShelf] = max(abs(diff(weightsPerShelf.wMean([nB nE], :))));  % Choose the shelf with the highest weight difference
        wB = weightsPerShelf.wMean(nB, iShelf);
        wE = weightsPerShelf.wMean(nE, iShelf);
        deltaW = wE - wB;
        
		% Determine contributing bins
        bins = cell(length(binWidth),1);
        for iBinW = 1:length(binWidth)
            bins{iBinW} = find(abs(diff(weightsPerBin(iBinW).wMean([nB nE],iShelf,:))) > systemParams.epsMeanPlate)';
        end

        % Fill in the new event entry
        events(iEv) = struct('tB',tB, 'tE',tE, 'nB',nB, 'nE',nE, 'deltaW',deltaW, 'bins',{bins}, 'shelf',iShelf);
    end
end
