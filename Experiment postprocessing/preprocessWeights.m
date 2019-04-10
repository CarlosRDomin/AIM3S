function [weights, experimentInfo] = preprocessWeights(experimentType, tStr, movAvgWindowInSec, Fsamp, numShelves, numPlatesPerShelf, DATA_FOLDER)
	if nargin<3 || isempty(movAvgWindowInSec), movAvgWindowInSec = 1.5; end
	if nargin<4 || isempty(Fsamp), Fsamp = 60; end
	if nargin<5 || isempty(numShelves), numShelves = 5; end
	if nargin<6 || isempty(numPlatesPerShelf), numPlatesPerShelf = 12; end
	if nargin<7 || isempty(DATA_FOLDER), DATA_FOLDER = '../Dataset'; end
	experimentInfo = struct('experimentType',experimentType, 'tStr',tStr, 'Fsamp',Fsamp, 'movAvgWindowInSec',movAvgWindowInSec);
	movAvgWindowMat = [round(movAvgWindowInSec*Fsamp)-1 0];  % Used for movmean and movvar
	aux = cell(numShelves,numPlatesPerShelf);
	weights = struct('tOrig',aux, 'wOrig',aux, 't',aux, 'w',aux, 'wMean',aux, 'wVar',aux);
	
	% Load weights
	data = readHDF5([DATA_FOLDER '/' experimentType '/' tStr '/weights_' tStr '.h5']);
	weightSensorNames = fieldnames(data);
	tStart = []; tEnd = [];
	for iS = 1:length(weightSensorNames)
		weightSensorName = weightSensorNames{iS};
		%t = datetime(data.(weightSensorName).t_str, 'InputFormat','yyyy-MM-dd HH:mm:ss.SSSSSS'); %t = data.(weightSensorName).t;
		t = seconds(data.(weightSensorName).t) + datetime(data.(weightSensorName).t_str(1), 'InputFormat','yyyy-MM-dd HH:mm:ss.SSSSSS');
		w = double(data.(weightSensorName).w);
		[jW,iW] = ind2sub([numPlatesPerShelf numShelves], iS);
		weights(iW,jW).tOrig = t; weights(iW,jW).wOrig = w;
		
		% Keep track of time of latest weight shelf to start and earliest shelf to end
		tStart = max(t(1), tStart, 'omitNaT'); tEnd = min(t(end), tEnd, 'omitNaT');
	end
	
	% Resample all weights to a common time
	t = tStart:seconds(1/Fsamp):tEnd;
	for iW = 1:size(weights, 1)
		for jW = 1:size(weights, 2)
			% First, remove values with exactly the same time (otherwise interp1 will error)
			tOrig = weights(iW,jW).tOrig; wOrig = weights(iW,jW).wOrig; indsRemove = (diff(tOrig)==0);
			tOrig(indsRemove) = []; wOrig(indsRemove) = [];
			if isempty(tOrig), continue; end  % Sometimes we forgot to record some weight scales, ignore those

			% Resample
			weights(iW,jW).t = t;
			weights(iW,jW).w = interp1(tOrig, wOrig, weights(iW,jW).t);
			% Compute moving mean and variance
			weights(iW,jW).wMean = movmean(weights(iW,jW).w, movAvgWindowMat);
			weights(iW,jW).wVar = movvar(weights(iW,jW).w, movAvgWindowMat);
		end
	end
end

