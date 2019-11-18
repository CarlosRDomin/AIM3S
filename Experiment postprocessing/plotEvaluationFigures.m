%clear all; 
close all;
loadCommonConstants;  % Load default systemParams, DATA_FOLDER, etc

%% Plot time-domain experiment example
if false
    figName = 'time-domain_weights_and_events';
    tStr = '2019-04-03_19-50-26';
    binWidth = 1;
    [weights, weightsPerBin, weightsPerShelf, cams, events, gt] = loadExperiment(tStr, experimentType, DATA_FOLDER, binWidth, systemParams, false);
    events_gt = gt2events(gt, weights, binWidth, systemParams);
    
    tOffs = duration(0,0,-7);
    f = plotExperimentWeights(weightsPerShelf, {events; events_gt}, systemParams, gt.t_exit_store, false, weights.t(1)-tOffs);
    for i = [1:4 6], xlim(f.Children(i), duration(0,[0 1],0)); end
    saveFigure(f, figName, FIG_FOLDER);
end

%% Plot time-domain event detection example
if false
    figName = 'time-domain_event_detection_example';
    tStr = '2019-04-03_20-10-22';
    binWidth = 1;
    [weights, weightsPerBin, weightsPerShelf, cams, events, gt] = loadExperiment(tStr, experimentType, DATA_FOLDER, binWidth, systemParams, false);
    events_gt = gt2events(gt, weights, binWidth, systemParams);
end

%% Plot weight distributions (Gaussians) of all products
if false
    figName = 'product_weights_distrubtion';
    [~, weightModelParams, productsInfo] = loadProductModels;
    weightModel = weightModelParams(1);
    productInfo = productsInfo{1};
    weightScaleVar = 1;  % std in grams of the weight plates
    weightModel.std(weightModel.std<weightScaleVar) = weightScaleVar;  % It doesn't make sense to have an std smaller than our sensor's error std, it would underestimate the overlap probability
    
    f = figure('Position', [0 0 1225 250]); subtightplot(1,1,1); hold on;
    xRange = 0:0.05:1200;
    y = normpdf(xRange, weightModel.mean, weightModel.std); yMargin = 0.03;
    cMap = colormap('cool');
    cMap = [0 1 0; 0 0 1; 1 0 0]; cMap = [158,1,66;213,62,79;244,109,67;200,0,225;94,79,162;50,136,189;102,194,165]/255; cMap = [210,0,225;94,79,162;]/255; %cMap = [165,0,38;15,165,15;0,104,55]/255;
    cMap = [0.05,0.03,0.53;0.78,0.26,0.48];c = interp1(linspace(0, xRange(end), size(cMap,1)), cMap, weightModel.mean);
    [~,I]=sort(weightModel.mean);  % Sort by mean
    for iI = 1:length(I)
        i = I(iI);
        plot(xRange, y(i,:), 'LineWidth',1.25, 'Display',sprintf('%2d - %s (\\mu=%.2fg, \\sigma=%.2fg)', iI, productInfo(i).name, weightModel.mean(i), weightModel.std(i)), 'Color',c(i,:));
        text(weightModel.mean(i), 1/(weightModel.std(i)*sqrt(2*pi)) + yMargin, num2str(iI), 'HorizontalAlignment','center', 'Color',c(i,:));
    end
    xlabel('Weight (g)'); ylabel('pdf');
    ylim([0 0.45]); legend('Location','SouthOutside', 'NumColumns',4);
    saveFigure(f, figName, FIG_FOLDER);
end

%% Plot accuracy
if true
    load('processedExperiments_backup.mat', 'processedExperiments'); processedExperiments_backup = processedExperiments;
    load('processedExperiments.mat', 'processedExperiments');
    cams = {1,2,3,4,[1 2], [3 4], [1 3], [2 3], [1 4], [2 4], [1 2 3], [1 2 4], [2 3 4], 1:4};
    camsNames = {'L','R','T','B','LR', 'TB', 'LT', 'RT', 'LB', 'RB', 'LRT', 'LRB', 'RTB', 'All'};
    
    locAcc = evaluateIDaccuracy(processedExperiments, 1);
    wChangeAcc = evaluateIDaccuracy(processedExperiments, 2);
    weightAcc = evaluateIDaccuracy(processedExperiments, 0, [], 1);
    for i = 1:length(cams)
        camAcc(i,:) = evaluateIDaccuracy(processedExperiments, 3, cams{i}, [], 0);
        camAccTop90(i,:) = evaluateIDaccuracy(processedExperiments, 3, cams{i}, [], 0.9);
        camAccTop90noBgndSubs(i,:) = evaluateIDaccuracy(processedExperiments_backup, 3, cams{i}, [], 0.9);
        camAim3s(i,:) = evaluateIDaccuracy(processedExperiments, 0, cams{i});
    end
    aim3sAcc = evaluateIDaccuracy(processedExperiments);
    
    figPos = [0 0 500 300]; fSize = 16;
    
    %% Accuracy vs binWidth
    f = figure('Position',figPos, 'DefaultAxesFontSize',fSize); subtightplot(1,1,1,[],[0 0.02],[0.12 0.02]); halfSep = 0.086;
    b=bar((1:6)' + (-3:2:3).*halfSep, [locAcc([1:5 end]); wChangeAcc([1:5 end]); weightAcc([1:5 end]); aim3sAcc([1:5 end])]', 'BarWidth', 5.5); %barvalues(f,1,true);
    xticks(1:6); xticklabels([1 2 3 4 6 12]); m = 0.52; xlim([1-m, 6+m]);
    c = [0.94,0.8,0.5; 0.98,0.63,0.15; 0.7,0.41,0.22; 0,0.44,0.71];
    for i=1:length(b), b(i).FaceColor = c(i,:); end
    xlabel('Bin width (# plates/bin)'); ylabel('Avg. ID Accuracy (%)'); set(gca, 'YGrid','on');
    legend(' Location-based (P_L)', ' Weight change-based (P_W)  ', ' Weight-based (P_{weight }) ', ' AIM3S (P_{fusion }) ', 'Location','SouthOutside', 'NumColumns',2);
    saveFigure(f, 'accuracyVsBinWidth', FIG_FOLDER);
    
    %% Accuracy vs arrangementResolution
    f = figure('Position',figPos, 'DefaultAxesFontSize',fSize); subtightplot(1,1,1,[],[0 0.02],[0.12 0.02]);
    b = bar(categorical(1:3, 1:3, {'Plate-level','Half-shelf-level','Shelf-level'}), [locAcc(1:5:end); wChangeAcc(1:5:end); weightAcc(1:5:end); aim3sAcc(1:5:end)]', 'BarWidth',0.84); barvalues;
    c = [0.94,0.8,0.5; 0.98,0.63,0.15; 0.7,0.41,0.22; 0,0.44,0.71];
    for i=1:length(b), b(i).FaceColor = c(i,:); end
    xlabel('Item layout resolution'); ylabel('Avg. ID Accuracy (%)'); set(gca, 'YGrid','on');
    legend(' Location-based (P_L)', ' Weight change-based (P_W)  ', ' Weight-based (P_{weight }) ', ' AIM3S (P_{fusion }) ', 'Location','SouthOutside', 'NumColumns',2);
    saveFigure(f, 'accuracyVsArrResolution', FIG_FOLDER);
    
    %% Accuracy vs camSetup
    f = figure('Position',[0 0 500 250], 'DefaultAxesFontSize',fSize-1); subtightplot(1,1,1,[],[0 0.02],[0.12 0.02]); halfSep = 0.17;
    b = bar((1:length(cams))' + (-1:2:1).*halfSep, [camAcc(:,1) camAccTop90(:,1)], 'BarWidth',3.5); %t=barvalues(f,1); for i=1:length(t), set(t{i}, 'FontSize',7.7); end
    xticks(1:length(cams)); xticklabels(camsNames); xtickangle(10); m = 0.6; xlim([1-m, length(cams)+m]);
    c = [0.68,0.15,0.17; 0.96,0.6,0.62];
    for i=1:length(b), b(i).FaceColor = c(i,:); end
    xlabel('Cameras used'); ylabel('Avg. ID Accuracy (%)');
    yyaxis right; set(gca, 'YColor',[0,0.47,0.76], 'YGrid','on'); p=get(gca, 'Position'); p(1)=p(1)-0.03; set(gca, 'Position',p);
    plot(1:length(cams), camAim3s(:,3), '--o', 'LineWidth',1.7, 'Color',[0,0.47,0.76], 'MarkerFaceColor',[0.3,0.75,0.93]); ylim([90 94]);
    l=legend(' Vision-based (argmax P_V)', ' Vision-based (P_V^i > 0.9)  ', ' AIM3S (P_{fusion }) ', 'Location','SouthOutside', 'Orientation','Horizontal', 'FontSize',fSize-3.7); p=get(l, 'Position'); p(1)=0.011; set(gca, 'ActivePositionProperty','OuterPosition'); set(l, 'Position',p);
    saveFigure(f, 'accuracyVsCamsUsed', FIG_FOLDER);
    
    %% Error vs binWidth
    f = figure('Position',[0 0 500 200], 'DefaultAxesFontSize',fSize); halfSep = 0.16; subtightplot(1,1,1,[],[0.2 0.04],[0.09 0.02]);
    if true
        b=bar((1:6)' + (-1:2:1).*halfSep, 100-[weightAcc([1:5 end]); aim3sAcc([1:5 end])]', 'BarWidth', 3.8); barvalues(f,1);
        xticks(1:6); xticklabels([1 2 3 4 6 12]); m = 0.6; xlim([1-m, 6+m]);
        c = [0.7,0.41,0.22; 0,0.44,0.71];
        for i=1:length(b), b(i).FaceColor = c(i,:); end
        xlabel('Bin width (# plates/bin)'); ylabel('Avg. ID Error (%)'); set(gca, 'YGrid','on');
        legend(' Weight-based (P_{weight }) ', ' AIM3S (P_{fusion }) ', 'Position',[0.11,0.67,0.36,0.24]);
    else
        b=bar(categorical([1 2 3 4 6 12]), diff([weightAcc([6:10 end]); aim3sAcc([6:10 end])])'); %barvalues(f,1,false);
        c = [0.93,0.69,0.13;0.35,0.85,0.07];
        for i=1:length(b), b(i).FaceColor = c(i,:); end
        xlabel('Bin width (# plates/bin)'); ylabel('Avg. Acc. Improvement (%)');
        legend(' AIM3S (P_{fusion }) ', 'Position',[0.155,0.625,0.37,0.24]);
    end
    saveFigure(f, 'errorVsBinWidth', FIG_FOLDER);
    
    hold on; plot([-1 13], [14 14], '--', 'Color',[0.76 0.05 0.05], 'LineWidth',1, 'DisplayName',' Self-Checkout');
    xticklabels(12./[1 2 3 4 6 12]); xlabel('Weight sensor density (# bins/shelf)');
    saveFigure(f, 'errorVsWeightDensity', FIG_FOLDER);
end

%% Plot 
if false
    load('processedExperiments_backup.mat', 'processedExperiments'); processedExperiments_backup = processedExperiments;
    load('processedExperiments.mat', 'processedExperiments');
    
    acc = evaluateIDaccuracy(processedExperiments, 0, [1:4], 0.6, 0.9);
    %bar(acc);
end