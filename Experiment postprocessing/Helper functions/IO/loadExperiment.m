function [weights, weightsPerBin, weightsPerShelf, cams, events, gt] = loadExperiment(tStr, experimentType, DATA_FOLDER, binWidth, systemParams, loadCams)
    if nargin<2 || isempty(experimentType), experimentType = 'Evaluation'; end
    if nargin<3 || isempty(DATA_FOLDER), DATA_FOLDER = '../Dataset'; end
    if nargin<4 || isempty(binWidth), binWidth = [1 2 3 4 6]; end
    if nargin<5 || isempty(systemParams), systemParams = struct('movAvgWindowInSamples',60, 'epsVar',500, 'epsMeanShelf',10, 'epsMeanPlate',5, 'N_high',30, 'N_low',30, 'Nvision',seconds(3), 'radHandSq',([100 100 100 200]/2).^2); end
    if nargin<4 || isempty(loadCams), loadCams = false; end

    % Read ground truth
    gt = jsondecode(fileread(sprintf('%s/%s/%s/ground_truth_%s.json', DATA_FOLDER, experimentType, tStr, tStr)));
    gt.t_exit_store_str = gt.t_exit_store; gt.t_exit_store = parseStrDatetime(gt.t_exit_store_str);
    
    % Read weights
    weights = loadWeightsData(tStr, experimentType, DATA_FOLDER, systemParams, gt);
    [events, weightsPerBin, weightsPerShelf] = detectWeightEvents(weights, binWidth, systemParams);
    
    % Read cameras (have a bool flag because they take long to load)
    if loadCams
        cams = loadCamsData(tStr, experimentType, DATA_FOLDER);
    else
        cams = [];
    end
end
