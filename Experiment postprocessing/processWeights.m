addpath(genpath('.'));	% Make sure all folders and subfolders are added to the path
cdToThisScriptsDirectory();	% Change directory to the folder containing this script
DATA_FOLDER = '../Dataset/Evaluation';
numShelves = 5;
numPlatesPerShelf = 12;
Fsamp = 60;
movAvgWindow = [round(1.5*Fsamp)-1 0]; % 1.5 s

%% Load weights
tStr = '2019-04-03_19-50-26';
data = readHDF5([DATA_FOLDER '/' tStr '/weights_' tStr '.h5']);
aux = cell(numShelves,numPlatesPerShelf);
weights = struct('tOrig',aux, 'wOrig',aux, 't',aux, 'w',aux, 'wMean',aux, 'wVar',aux);
weightSensorNames = fieldnames(data);
tStart = []; tEnd = [];
for iS = 1:length(weightSensorNames)
	weightSensorName = weightSensorNames{iS};
	%t = datetime(data.(weightSensorName).t_str, 'InputFormat','yyyy-MM-dd HH:mm:ss.SSSSSS'); %t = data.(weightSensorName).t;
	t = seconds(data.(weightSensorName).t) + datetime(data.(weightSensorName).t_str(1), 'InputFormat','yyyy-MM-dd HH:mm:ss.SSSSSS');
	w = double(data.(weightSensorName).w);
	[jW,iW] = ind2sub([numPlatesPerShelf numShelves], iS);
	weights(iW,jW).tOrig = t; weights(iW,jW).wOrig = w;
	tStart = max(t(1), tStart, 'omitNaT'); tEnd = min(t(end), tEnd, 'omitNaT');
end
%% Resample all weights to a common time
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
		weights(iW,jW).wMean = movmean(weights(iW,jW).w, movAvgWindow);
		weights(iW,jW).wVar = movvar(weights(iW,jW).w, movAvgWindow);
	end
end

%% Visualize weights
% One plot per plate
figure('WindowState','maximized'); pause(0.1);
for iW = 1:size(weights, 1)
	for jW = 1:size(weights, 2)
		if isempty(weights(iW,jW).t), continue; end  % Sometimes we forgot to record some weight scales, ignore those
		t = weights(iW,jW).t; w = weights(iW,jW).w;
		wMean = movmean(w, movAvgWindow); wVar = movvar(w, movAvgWindow);
		
		subtightplot(numShelves, numPlatesPerShelf, jW + size(weights,2)*(iW-1)); hold on; plot(t, w);
		plot(t, wMean, 'LineWidth', 3); plot(t, wVar.*max(abs(w))./max(wVar), '--r', 'LineWidth', 2);
		set(gca, 'YGrid','on', 'XLim',[t(1) t(end)]);
	end
end

% All plates in one plot
figure('WindowState','maximized');
for iW = 1:size(weights, 1)
	for jW = 1:size(weights, 2)
		if isempty(weights(iW,jW).t), continue; end  % Sometimes we forgot to record some weight scales, ignore those
		t = weights(iW,jW).t; w = weights(iW,jW).w;
		wMean = movmean(w, movAvgWindow); wVar = movvar(w, movAvgWindow);
		
		subplot(2,1,1); hold on; plot(t, w); set(gca, 'YGrid','on', 'XLim',[t(1) t(end)]);
		subplot(2,1,2); hold on; plot(t, wMean, 'LineWidth', 2); set(gca, 'YGrid','on', 'XLim',[t(1) t(end)]);
	end
end

% Sum of plates in a shelf per plot
figure('WindowState','maximized');
for iW = 1:size(weights, 1)
	nNotNaT = 0;
	for jW = 1:size(weights, 2)
		if isempty(weights(iW,jW).t), nNotNaT = nNotNaT + 1; end  % Sometimes we forgot to record some weight scales, ignore those
	end
	t = weights(iW,1).t; w = reshape([weights(iW,:).w], [],numPlatesPerShelf-nNotNaT); w = sum(w,2);
	wMean = movmean(w, movAvgWindow); wVar = movvar(w, movAvgWindow);
	
	subplot(size(weights,1), 1, iW); plot(t, w); hold on; plot(t, wMean, 'LineWidth', 2); plot(t, wVar.*max(abs(w))./max(wVar), '--r', 'LineWidth', 2);
	set(gca, 'YGrid','on', 'XLim',[t(1) t(end)]);
end
