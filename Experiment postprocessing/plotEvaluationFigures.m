clear all; close all;
loadCommonConstants;  % Load default systemParams, DATA_FOLDER, etc

%% Plot time-domain experiment example
if true
    tStr = '2019-04-03_19-50-26';
    figName = 'time-domain_weights_and_events';
    binWidth = 1;
    [weights, weightsPerBin, weightsPerShelf, cams, events, gt] = loadExperiment(tStr, experimentType, DATA_FOLDER, binWidth, systemParams);
    events_gt = gt2events(gt, weights, binWidth, systemParams);
    f = plotExperimentWeights(weightsPerShelf, {events; events_gt}, systemParams, gt.t_exit_store);
    saveFigure(f, figName, FIG_FOLDER);
end

%% Plot accuracy
if true
    load('processedExperiments.mat', 'processedExperiments');

    results = false(0,11);
    cntEvents = 1;
    for iExp = 1:length(processedExperiments)
        for iEv = 1:length(processedExperiments(iExp).from_gt_events)
            aux = processedExperiments(iExp).from_gt_events(iEv);
            for iArr = 1:11
                if iArr <= 5
                    probArr = aux.probPlate{iArr};
                elseif iArr <= 10
                    probArr = aux.probHalfShelf{iArr-5};
                else
                    probArr = aux.probShelf;
                end
                totalProb = aux.probWeight .* probArr .* aux.probVision;
                totalProb = totalProb./sum(totalProb);
                [~, iMax] = max(totalProb);
                results(cntEvents,iArr) = (iMax == processedExperiments(iExp).gt.ground_truth(aux.iEv).item_id);
            end
            cntEvents = cntEvents+1;
        end
    end
    acc = 100 * sum(results)./length(results)
    %bar(acc);
end