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
    processedExperiments = [processedExperiments; processExperiment(tStr, experimentType, DATA_FOLDER, [], systemParams, productArrangement, weightModelParams, true)];
end
save('processedExperiments.mat', 'processedExperiments');
