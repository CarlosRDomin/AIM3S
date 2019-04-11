function [weights, experimentInfo, weightsOrig] = loadWeightsData(experimentType, tStr, Fsamp, weightIDs, DATA_FOLDER)
	if nargin<3 || isempty(Fsamp), Fsamp = 60; end
	if nargin<4 || isempty(weightIDs), weightIDs = [5, 12]; end  % A 1x2 vector indicates numShelves x numPlatesPerShelf
	if nargin<5 || isempty(DATA_FOLDER), DATA_FOLDER = '../Dataset'; end
	if iscell(weightIDs), s = size(weightIDs); else, s = weightIDs; end
	weightsOrig = struct('t',cell(s), 'w',cell(s));
	weights = struct('t',[], 'w',[], 'wMean',[], 'wVar',[]);
	experimentInfo = struct('experimentType',[], 'tStr',[], 'weightIDs',[], 'Fsamp',[]);
	
	% Load weights
	data = readHDF5([DATA_FOLDER '/' experimentType '/' tStr '/weights_' tStr '.h5']);
	if ~iscell(weightIDs)  % A 1x2 vector indicates numShelves x numPlatesPerShelf -> Convert to cell with the actual weight IDs
		weightSensorNames = fieldnames(data);
		if prod(weightIDs) ~= length(weightSensorNames)
			warning('HEADS UP, some weight scales were not recorded in experiment "%s": expected %d, got %d!!', tStr, prod(weightIDs), length(weightSensorNames));
			% Manually add as many empty entries as needed... :/
			weightSensorNames = [weightSensorNames; repmat({''}, prod(weightIDs)-length(weightSensorNames), 1)];
		end
		weightIDs = reshape(weightSensorNames, weightIDs(2), weightIDs(1))';
	end
	tStart = []; tEnd = [];
	for iW = 1:size(weightIDs, 1)
		for jW = 1:size(weightIDs, 2)
			weightID = weightIDs{iW,jW};
			if isempty(weightID), continue; end
			
			weightsOrig(iW,jW).t = seconds(data.(weightID).t) + datetime(data.(weightID).t_str(1), 'InputFormat','yyyy-MM-dd HH:mm:ss.SSSSSS');
			weightsOrig(iW,jW).w = double(data.(weightID).w);

			% Keep track of time of latest weight shelf to start and earliest shelf to end
			tStart = max(weightsOrig(iW,jW).t(1), tStart, 'omitNaT');
			tEnd = min(weightsOrig(iW,jW).t(end), tEnd, 'omitNaT');
		end
	end
	
	% Resample all weights to a common time
	weights.t = tStart:seconds(1/Fsamp):tEnd;
	weights.w = zeros([size(weightIDs), length(weights.t)]);
	weights.wMean = zeros(size(weights.w));
	weights.wVar = zeros(size(weights.w));
	for iW = 1:size(weightIDs, 1)
		for jW = 1:size(weightIDs, 2)
			% First, remove values with exactly the same time (otherwise interp1 will error)
			tOrig = weightsOrig(iW,jW).t; wOrig = weightsOrig(iW,jW).w; indsRemove = (diff(tOrig)==0);
			tOrig(indsRemove) = []; wOrig(indsRemove) = [];
			if isempty(tOrig), continue; end  % Sometimes we forgot to record some weight scales, ignore those

			% Resample
			weights.w(iW,jW,:) = interp1(tOrig, wOrig, weights.t);
		end
	end
	
	% Fill in experimentInfo
	fNames = fieldnames(experimentInfo);
	for f = 1:length(fNames)
		experimentInfo.(fNames{f}) = eval(fNames{f});
	end
end

