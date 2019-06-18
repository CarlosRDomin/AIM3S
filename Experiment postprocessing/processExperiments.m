clear all; close all;
loadCommonConstants;  % Load default systemParams, DATA_FOLDER, etc

% Load product models (weight, arrangement...)
[productArrangement, weightModelParams] = loadProductModels(DATA_FOLDER);

% Load experiment(s)
processedExperiments = [];
f = dir([DATA_FOLDER '/' experimentType]);
for i = 3:length(f)  % Start at 3 to ignore '.' and '..'
    if ~f(i).isdir, continue; end
    tStr = f(i).name;
    processedExperiments = [processedExperiments; processExperiment(tStr, experimentType, DATA_FOLDER, [], systemParams, productArrangement, weightModelParams)];
        
%         % Compute total probabilities by aggregating each source of information and normalizing
%         totalProbPlate = probWeight.*probPlate; totalProbPlate = totalProbPlate/sum(totalProbPlate);
%         totalProbHalfShelf = probWeight.*probHalfShelf; totalProbHalfShelf = totalProbHalfShelf/sum(totalProbHalfShelf);
%         totalProbShelf = probWeight.*probShelf; totalProbShelf = totalProbShelf/sum(totalProbShelf);
%         
%         aux = [1:length(totalProbPlate); totalProbPlate']; fprintf('Event %d prediction:\n', iEv); disp(aux(:,aux(2,:)>1e-6));
end
save('processedExperiments.mat', 'processedExperiments');
