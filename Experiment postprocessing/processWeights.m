addpath(genpath('.'));	% Make sure all folders and subfolders are added to the path
cdToThisScriptsDirectory();	% Change directory to the folder containing this script

%% Load experiment
DATA_FOLDER = '../Dataset';
experimentType = 'Evaluation';
tStr = '2019-04-03_19-50-26';
[weights, experimentInfo, weightsOrig] = preprocessWeights(experimentType, tStr);

%% Visualize weights
% One plot per plate
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

% All plates in one plot
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

% Sum of plates in a shelf per plot
figure('WindowState','maximized');
movAvgWindow = [round(experimentInfo.movAvgWindowInSec*experimentInfo.Fsamp)-1 0];
t = weights.t; w = squeeze(sum(weights.w,2));
wMean = movmean(w, movAvgWindow, 2); wVar = movvar(w, movAvgWindow, 1, 2);
for iW = 1:size(w, 1)
	subplot(size(w, 1), 1, iW); hold on;
	plot(t, w(iW,:)); plot(t, wMean(iW,:), 'LineWidth', 2);
	yyaxis right; plot(t, wVar(iW,:), '--r', 'LineWidth', 2);
	set(gca, 'YGrid','on', 'XLim',[t(1) t(end)]);
end
