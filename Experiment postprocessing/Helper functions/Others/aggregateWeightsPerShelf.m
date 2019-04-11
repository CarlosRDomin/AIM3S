function weightsPerShelf = aggregateWeightsPerShelf(weights, systemParams)
	if nargin < 2, systemParams = []; end

	weights.w = squeeze(sum(weights.w,2));
	weightsPerShelf = computeWeightMeanAndVar(weights, systemParams);
end
