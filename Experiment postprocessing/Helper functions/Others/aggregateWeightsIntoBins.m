function weightsPerBin = aggregateWeightsIntoBins(weights, binWidth, systemParams)
	if nargin<2 || isempty(binWidth), binWidth = size(weights.w, 3); end  % By default, aggregate the whole shelf
	if nargin<3, systemParams = []; end
    
    weightsPerBin = struct('t',cell(length(binWidth),1), 'w',[], 'wMean',[], 'wVar',[]);
    for i = 1:length(binWidth)
        % 1 bin for every binWidth plates -> Last dimension is binWidth times smaller
        newSize = size(weights.w); newSize(3) = newSize(3)/binWidth(i);

        % Trick: we can use reshape to split weights in chunks of binWidth plates,
        % then sum along that (3rd) dimension, and squeeze to remove it
        aux = weights;
        aux.w = squeeze(sum(reshape(weights.w, newSize(1), newSize(2), [], newSize(3)), 3));
        weightsPerBin(i) = computeWeightMeanAndVar(aux, systemParams);
    end
end
