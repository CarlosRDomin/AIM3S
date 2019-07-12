function [acc, results] = evaluateIDaccuracy(processedExperiments, fusionMethod, cams, alpha, camsTopPercent)
    if nargin<2 || isempty(fusionMethod), fusionMethod=0; end
    if nargin<3 || isempty(cams), cams=1:4; end
    if nargin<4 || isempty(alpha), alpha=0.6; end
    if nargin<5 || isempty(camsTopPercent), camsTopPercent=0.9; end
    results = false(0,11);
    cntEvents = 1;
    for iExp = 1:length(processedExperiments)
        for iEv = 1:length(processedExperiments(iExp).from_gt_events)
            aux = processedExperiments(iExp).from_gt_events(iEv);
            probVision = 1 - prod(1-aux.probVision(:,cams), 2);
            for iArr = 1:(length(aux.probPlate)+length(aux.probHalfShelf)+1)
                if iArr <= length(aux.probPlate)
                    probArr = aux.probPlate{iArr};
                elseif iArr <= length(aux.probPlate)+length(aux.probHalfShelf)
                    probArr = aux.probHalfShelf{iArr-length(aux.probPlate)};
                else
                    probArr = aux.probShelf;
                end
                probArr = probArr.*(probArr>=1e-2);
                
                %totalProb = aux.probWeight .* probArr .* probVision;
                totalProb = aux.probWeight .* probArr;
                totalProb = totalProb./sum(totalProb);
                
                switch(fusionMethod)
                case 0
                    totalProb = alpha*totalProb + (1-alpha)*probVision;
                    totalProb = totalProb./sum(totalProb);
                    [pMax, iMax] = max(totalProb);
                case 1
                    [pMax, iMax] = max(probArr);
                case 2
                    [pMax, iMax] = max(aux.probWeight);
                case 3
                    [pMax, iMax] = max(probVision);
                case 4
                    [pMax, iMax] = max(probArr.*probVision);
                end
                
                if fusionMethod == 3 && camsTopPercent>0
                    results(cntEvents,iArr) = probVision(processedExperiments(iExp).gt.ground_truth(aux.iEv).item_id) > camsTopPercent;
                else
                    results(cntEvents,iArr) = (iMax == processedExperiments(iExp).gt.ground_truth(aux.iEv).item_id);
                end
            end
            cntEvents = cntEvents+1;
        end
    end
    acc = 100 * sum(results)./size(results,1);
end
