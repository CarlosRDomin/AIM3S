close all;
loadCommonConstants;  % Load default systemParams, DATA_FOLDER, etc

%% Load experiment
tStr = '2019-04-03_20-10-22'; '2019-04-03_19-50-26';
binWidth = 1;
[weights, weightsPerBin, weightsPerShelf, cams, events, gt] = loadExperiment(tStr, experimentType, DATA_FOLDER, binWidth, systemParams);
events_gt = gt2events(gt, weights, binWidth, systemParams);

%% Sum of plates in a shelf per plot
for plotVar = 1:-1:0
    plotExperimentWeights(weightsPerShelf, {events; events_gt}, systemParams, gt.t_exit_store, plotVar);
end
return

%% One plot per bin
figure('WindowState','maximized'); pause(0.1);
for iS = 1:size(weightsPerBin.w, 2)
	for iB = 1:size(weightsPerBin.w, 3)
		subtightplot(size(weightsPerBin.w,2), size(weightsPerBin.w,3), iB + size(weightsPerBin.w,3)*(size(weightsPerBin.w, 2)-iS)); hold on;
		plot(weightsPerBin.t, weightsPerBin.w(:,iS,iB)); plot(weightsPerBin.t, weightsPerBin.wMean(:,iS,iB), 'LineWidth', 3);
		yyaxis right; plot(weightsPerBin.t, weightsPerBin.wVar(:,iS,iB), '--r', 'LineWidth', 2);
		set(gca, 'YGrid','on', 'XLim',[weightsPerBin.t(1) weightsPerBin.t(end)]);
		title(sprintf('s%d - b%d', iS, iB));
	end
end

%% All plates in one plot
figure('WindowState','maximized');
for iS = 1:size(weights.w, 2)
	for iB = 1:size(weights.w, 3)
		subplot(2,1,1); hold on; plot(weights.t, weights.w(:,iS,iB));
		subplot(2,1,2); hold on; plot(weights.t, weights.wMean(:,iS,iB), 'LineWidth', 2);
	end
end
subplot(2,1,1); set(gca, 'YGrid','on', 'XLim',[weights.t(1) weights.t(end)]);
subplot(2,1,2); set(gca, 'YGrid','on', 'XLim',[weights.t(1) weights.t(end)]);
