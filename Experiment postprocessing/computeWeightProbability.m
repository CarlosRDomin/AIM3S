function p = computeWeightProbability(deltaW, weightModel, weightScaleVar, plotLims)
    if nargin<3 || isempty(weightScaleVar), weightScaleVar=1; end  % std in grams of the weight plates
    if nargin<4 || isempty(plotLims), plotLims = []; end  % Empty => No plot; Otherwise => [xMin xMax] to plot the pdf of all gaussians in that range
    weightModel.std(weightModel.std<weightScaleVar) = weightScaleVar;  % It doesn't make sense to have an std smaller than our sensor's error std, it would underestimate the overlap probability
    
    p = zeros(length(weightModel.mean),1);
    for i = 1:length(weightModel.mean)
        p(i) = areaUnderTwoGaussians(weightModel.mean(i), weightModel.std(i), abs(deltaW), weightScaleVar, plotLims);
    end
end
