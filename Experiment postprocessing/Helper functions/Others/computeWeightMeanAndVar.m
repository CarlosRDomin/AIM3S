function weights = computeWeightMeanAndVar(weights, systemParams)
	if nargin<2 || ~isfield(systemParams, 'movAvgWindowInSamples'), systemParams = struct('movAvgWindowInSamples', 60); end
	movAvgWindowMat = [systemParams.movAvgWindowInSamples-1, 0];  % Use past movAvgWindowInSamples points
	
	% Compute moving mean and variance
% 	weights.wMean = movmean(weights.w, movAvgWindowMat);
% 	weights.wVar = movvar(weights.wMean, movAvgWindowMat);
	weights.wMean = movmean(weights.w, systemParams.movAvgWindowInSamples);
	weights.wVar = movvar(weights.wMean, systemParams.movAvgWindowInSamples);
end
