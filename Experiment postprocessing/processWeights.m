addpath(genpath('.'));	% Make sure all folders and subfolders are added to the path
cdToThisScriptsDirectory();	% Change directory to the folder containing this script
systemParams = struct('movAvgWindowInSamples',90, 'epsVar',6e4, 'epsMean',100, 'Nb',60, 'Ne',60);

%% Load experiment
DATA_FOLDER = '../Dataset';
experimentType = 'Evaluation';
tStr = '2019-04-03_19-50-26';
[weights, experimentInfo, weightsOrig] = loadWeightsData(experimentType, tStr);
[events, weights, eventInds] = detectWeightEvents(weights, 3, systemParams, experimentInfo);
return

%% One plot per plate
figure('WindowState','maximized'); pause(0.1);
for iW = 1:size(weights.w, 1)
	for jW = 1:size(weights.w, 2)
		if isempty(weightsOrig(iW,jW).t), continue; end  % Sometimes we forgot to record some weight scales, ignore those
		t = weights.t; w = squeeze(weights.w(iW,jW,:));
		wMean = squeeze(weights.wMean(iW,jW,:)); wVar = squeeze(weights.wVar(iW,jW,:));
		
		subtightplot(size(weights.w,1), size(weights.w,2), jW + size(weights.w,2)*(iW-1)); hold on;
		plot(t, w); plot(t, wMean, 'LineWidth', 3);
		yyaxis right; plot(t, wVar, '--r', 'LineWidth', 2);
		set(gca, 'YGrid','on', 'XLim',[t(1) t(end)]);
	end
end

%% All plates in one plot
figure('WindowState','maximized');
for iW = 1:size(weights.w, 1)
	for jW = 1:size(weights.w, 2)
		if isempty(weightsOrig(iW,jW).t), continue; end  % Sometimes we forgot to record some weight scales, ignore those
		t = weights.t; w = squeeze(weights.w(iW,jW,:));
		wMean = squeeze(weights.wMean(iW,jW,:)); wVar = squeeze(weights.wVar(iW,jW,:));
		
		subplot(2,1,1); hold on; plot(t, w); set(gca, 'YGrid','on', 'XLim',[t(1) t(end)]);
		subplot(2,1,2); hold on; plot(t, wMean, 'LineWidth', 2); set(gca, 'YGrid','on', 'XLim',[t(1) t(end)]);
	end
end

%% Sum of plates in a shelf per plot
figure('WindowState','maximized');
weightsPerShelf = aggregateWeightsPerShelf(weights, systemParams);
for iW = 1:size(weightsPerShelf.w, 1)
	subplot(size(weightsPerShelf.w, 1), 1, iW); hold on;
	plot(weightsPerShelf.t, weightsPerShelf.w(iW,:)); plot(weightsPerShelf.t, weightsPerShelf.wMean(iW,:), 'LineWidth', 2);
	yyaxis right; plot(weightsPerShelf.t, eventInds(iW,:), '--r', 'LineWidth', 2); %plot(weightsPerShelf.t, weightsPerShelf.wVar(iW,:), '--r', 'LineWidth', 2);
	set(gca, 'YGrid','on', 'XLim',[weightsPerShelf.t(1) weightsPerShelf.t(end)]);
end
