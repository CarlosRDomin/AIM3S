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
    f = figure('Position',figPos, 'DefaultAxesFontSize',fSize); subtightplot(1,1,1,[],[0 0.02],[0.12 0.02]);
    b=bar(categorical([1 2 3 4 6 12]), [locAcc([1:5 end]); wChangeAcc([1:5 end]); weightAcc([1:5 end]); aim3sAcc([1:5 end])]'); %barvalues(f,1,false);
    c = [0.00,0.45,0.74;0.85,0.33,0.10;0.93,0.69,0.13;0.35,0.85,0.07];
    for i=1:length(b), b(i).FaceColor = c(i,:); end
    xlabel('Bin width (# plates/bin)'); ylabel('Avg. ID Accuracy (%)');
    legend(' Location-based (P_L)', ' Weight change-based (P_W)  ', ' Weight-based (P_{weight }) ', ' AIM3S (P_{fusion }) ', 'Location','SouthOutside', 'NumColumns',2);
    saveFigure(f, 'accuracyVsBinWidth', FIG_FOLDER);
    
    %% Accuracy vs arrangementResolution
    f = figure('Position',figPos, 'DefaultAxesFontSize',fSize); subtightplot(1,1,1,[],[0 0.02],[0.12 0.02]);
    b = bar(categorical(1:3, 1:3, {'Plate-level','Half-shelf-level','Shelf-level'}), [locAcc(1:5:end); wChangeAcc(1:5:end); weightAcc(1:5:end); aim3sAcc(1:5:end)]'); barvalues;
    c = [0.00,0.45,0.74;0.85,0.33,0.10;0.93,0.69,0.13;0.35,0.85,0.07];
    for i=1:length(b), b(i).FaceColor = c(i,:); end
    xlabel('Arrangement resolution'); ylabel('Avg. ID Accuracy (%)');
    legend(' Location-based (P_L)', ' Weight change-based (P_W)  ', ' Weight-based (P_{weight }) ', ' AIM3S (P_{fusion }) ', 'Location','SouthOutside', 'NumColumns',2);
    saveFigure(f, 'accuracyVsArrResolution', FIG_FOLDER);
    
    %% Accuracy vs camSetup
    f = figure('Position',[0 0 500 250], 'DefaultAxesFontSize',fSize-1); subtightplot(1,1,1,[],[0 0.02],[0.12 0.02]);
    b = bar(categorical(1:length(cams), 1:length(cams), camsNames), [camAcc(:,1) camAccTop90(:,1) camAccTop90noBgndSubs(:,1) camAim3s(:,3)]); %barvalues;
    c = [0.64,0.08,0.18;1.00,0.07,0.65;1.00,0.07,0.65;0.39,0.83,0.07];
    for i=1:length(b), b(i).FaceColor = c(i,:); end
    xlabel('Cameras used'); ylabel('Avg. ID Accuracy (%)');
    legend(' Vision-based (argmax P_V)', ' Vision-based (P_V^i > 0.9)  ', ' Vision-based (P_V^i > 0.9)  ', ' AIM3S (P_{fusion }) ', 'Location','SouthOutside', 'Orientation','Horizontal', 'FontSize',fSize-3.7);
    saveFigure(f, 'accuracyVsCamsUsed', FIG_FOLDER);
    
    %% Error vs binWidth
    f = figure('Position',[0 0 500 200], 'DefaultAxesFontSize',fSize);% subtightplot(1,1,1,[],[0.18 0.02],[0.12 0.02]);
    if true
        b=bar(categorical([1 2 3 4 6 12]), 100-[weightAcc([6:10 end]); aim3sAcc([6:10 end])]'); %barvalues(f,1,false);
        c = [0.93,0.69,0.13;0.35,0.85,0.07];
        for i=1:length(b), b(i).FaceColor = c(i,:); end
        xlabel('Bin width (# plates/bin)'); ylabel('Avg. ID Error (%)');
        legend(' Weight-based (P_{weight }) ', ' AIM3S (P_{fusion }) ', 'Position',[0.155,0.625,0.37,0.24]);
    else
        b=bar(categorical([1 2 3 4 6 12]), diff([weightAcc([6:10 end]); aim3sAcc([6:10 end])])'); %barvalues(f,1,false);
        c = [0.93,0.69,0.13;0.35,0.85,0.07];
        for i=1:length(b), b(i).FaceColor = c(i,:); end
        xlabel('Bin width (# plates/bin)'); ylabel('Avg. Acc. Improvement (%)');
        legend(' AIM3S (P_{fusion }) ', 'Position',[0.155,0.625,0.37,0.24]);
    end
    saveFigure(f, 'errorVsBinWidth', FIG_FOLDER);
end

%% Plot 
if false
    load('processedExperiments_backup.mat', 'processedExperiments'); processedExperiments_backup = processedExperiments;
    load('processedExperiments.mat', 'processedExperiments');
    
    acc = evaluateIDaccuracy(processedExperiments, 0, [1:4], 0.6, 0.9);
    %bar(acc);
end