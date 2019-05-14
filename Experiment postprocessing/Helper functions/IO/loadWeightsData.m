function [weights, experimentInfo] = loadWeightsData(tStr, experimentType, DATA_FOLDER, systemParams)
	if nargin<2 || isempty(experimentType), experimentType = 'Evaluation'; end
	if nargin<3 || isempty(DATA_FOLDER), DATA_FOLDER = '../Dataset'; end
	if nargin<4 || isempty(systemParams), systemParams = struct('epsVar',100, 'epsMean',10, 'Nb',60, 'Ne',60, 'movAvgWindowInSamples',90); end
	weights = struct('t',[], 'w',[], 'wMean',[], 'wVar',[]);
	experimentInfo = struct('experimentType',[], 'tStr',[]);
	
	% Load weights
	data = readHDF5([DATA_FOLDER '/' experimentType '/' tStr '/weights_' tStr '.h5']);
    
    % Fill in time and weights info
    fmt = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
    weights.t = seconds(data.t) + datetime(data.t_str{1}(1:length(fmt)), 'InputFormat',fmt);
    weights.w = double(permute(data.w, [1 3 2]));
    %weights.w(:,1,3) = weights.w(:,1,3) + 8275;  % Manual offset compensation
    weights = computeWeightMeanAndVar(weights, systemParams);
	
	% Fill in experimentInfo
	fNames = fieldnames(experimentInfo);
	for f = 1:length(fNames)
		experimentInfo.(fNames{f}) = eval(fNames{f});
	end
end

