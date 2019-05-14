addpath(genpath('.'));	% Make sure all folders and subfolders are added to the path
cdToThisScriptsDirectory();	% Change directory to the folder containing this script
systemParams = struct('movAvgWindowInSamples',90, 'epsVar',500, 'epsMean',10, 'Nb',30, 'Ne',30);

%% Load experiment
DATA_FOLDER = '../Dataset';
experimentType = 'Evaluation';
tStr = '2019-04-03_19-50-26';
[weights, experimentInfo] = loadWeightsData(tStr, experimentType, DATA_FOLDER, systemParams);
[events, weightsPerBin, weightsPerShelf] = detectWeightEvents(weights, 1, systemParams);
eventInds = computeEventActiveState(events, weightsPerBin);

%% Sum of plates in a shelf per plot
for i = 1:2
    figure('WindowState','maximized');
    for iS = 1:size(weightsPerShelf.w, 2)
        subplot(size(weightsPerShelf.w, 2), 1, size(weightsPerShelf.w, 2)-iS+1); hold on;
        if i == 2
            plot(weightsPerShelf.t, weightsPerShelf.w(:,iS)); plot(weightsPerShelf.t, weightsPerShelf.wMean(:,iS), 'LineWidth', 2);
        else
            plot(weightsPerShelf.t, weightsPerShelf.wVar(:,iS), 'LineWidth', 2);
        end
        yyaxis right; plot(weightsPerShelf.t, eventInds(:,iS), '--r', 'LineWidth', 2); ylim([0 1]);
        set(gca, 'YGrid','on', 'XLim',[weightsPerShelf.t(1) weightsPerShelf.t(end)]);
        title(sprintf('Shelf %d', iS));
    end
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
