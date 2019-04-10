addpath(genpath('.'));	% Make sure all folders and subfolders are added to the path
cdToThisScriptsDirectory();	% Change directory to the folder containing this script

% Load experiment
[weights, experimentInfo] = preprocessWeights('Evaluation', '2019-04-03_19-50-26');

%% Visualize weights
% One plot per plate
figure('WindowState','maximized'); pause(0.1);
for iW = 1:size(weights, 1)
	for jW = 1:size(weights, 2)
		if isempty(weights(iW,jW).t), continue; end  % Sometimes we forgot to record some weight scales, ignore those
		t = weights(iW,jW).t; w = weights(iW,jW).w;
		wMean = weights(iW,jW).wMean; wVar = weights(iW,jW).wVar;
		
		subtightplot(size(weights, 1), size(weights, 2), jW + size(weights,2)*(iW-1)); hold on; plot(t, w);
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
		wMean = weights(iW,jW).wMean; wVar = weights(iW,jW).wVar;
		
		subplot(2,1,1); hold on; plot(t, w); set(gca, 'YGrid','on', 'XLim',[t(1) t(end)]);
		subplot(2,1,2); hold on; plot(t, wMean, 'LineWidth', 2); set(gca, 'YGrid','on', 'XLim',[t(1) t(end)]);
	end
end

% Sum of plates in a shelf per plot
figure('WindowState','maximized');
movAvgWindow = [round(experimentInfo.movAvgWindowInSec*experimentInfo.Fsamp)-1 0];
for iW = 1:size(weights, 1)
	nNotNaT = 0;
	for jW = 1:size(weights, 2)
		if isempty(weights(iW,jW).t), nNotNaT = nNotNaT + 1; end  % Sometimes we forgot to record some weight scales, ignore those
	end
	t = weights(iW,1).t; w = reshape([weights(iW,:).w], [],size(weights,2)-nNotNaT); w = sum(w,2);
	wMean = movmean(w, movAvgWindow); wVar = movvar(w, movAvgWindow);
	
	subplot(size(weights,1), 1, iW); plot(t, w); hold on; plot(t, wMean, 'LineWidth', 2); plot(t, wVar.*max(abs(w))./max(wVar), '--r', 'LineWidth', 2);
	set(gca, 'YGrid','on', 'XLim',[t(1) t(end)]);
end
