function [weights, experimentInfo] = loadWeightsData(tStr, experimentType, DATA_FOLDER, systemParams, gt)
	if nargin<2 || isempty(experimentType), experimentType = 'Evaluation'; end
	if nargin<3 || isempty(DATA_FOLDER), DATA_FOLDER = '../Dataset'; end
	if nargin<4 || isempty(systemParams), systemParams = []; end
    if nargin<5 || isempty(gt), gt = struct('weight_to_cam_t_offset_float', 0); end
	weights = struct('t',[], 'w',[], 'wMean',[], 'wVar',[]);
	experimentInfo = struct('experimentType',[], 'tStr',[]);
    fprintf('Loading weight(s) from experiment t=%s...\n', tStr);
	
	% Load weights
	data = readHDF5([DATA_FOLDER '/' experimentType '/' tStr '/weights_' tStr '.h5']);
    
    % Fill in time and weights info
    weights.t = seconds(data.t) + parseStrDatetime(data.t_str{1}) - seconds(gt.weight_to_cam_t_offset_float);
    weights.w = double(permute(data.w, [1 3 2]));
    weights = computeWeightMeanAndVar(weights, systemParams);
	
	% Fill in experimentInfo
	fNames = fieldnames(experimentInfo);
	for f = 1:length(fNames)
		experimentInfo.(fNames{f}) = eval(fNames{f});
	end
end
