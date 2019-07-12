function out = processExperiment(tStr, experimentType, DATA_FOLDER, binWidth, systemParams, productArrangement, weightModelParams, doVision)
    if nargin<2 || isempty(experimentType), experimentType = 'Evaluation'; end
    if nargin<3 || isempty(DATA_FOLDER), DATA_FOLDER = '../Dataset'; end
    if nargin<4 || isempty(binWidth), binWidth = [1 2 3 4 6]; end
    if nargin<5 || isempty(systemParams), systemParams = struct('movAvgWindowInSamples',60, 'epsVar',500, 'epsMeanShelf',10, 'epsMeanPlate',5, 'N_high',30, 'N_low',30, 'Nvision',seconds(3), 'radHandSq',([100 100 100 200]/2).^2); end
    if nargin<6 || isempty(productArrangement) || isempty(weightModelParams), [productArrangement, weightModelParams] = loadProductModels(DATA_FOLDER); end
    if nargin<8 || isempty(doVision), doVision = true; end
    
    [weights, weightsPerBin, ~, cams, events_detected, gt] = loadExperiment(tStr, experimentType, DATA_FOLDER, binWidth, systemParams, doVision);
    is5shelf = (size(weights.w, 2) == 5);
    events_gt = gt2events(gt, weights, binWidth, systemParams);
    out = struct('tStr',tStr, 'gt',gt, 'events_gt',events_gt, 'events_detected',events_detected, 'from_detected_events',[], 'from_gt_events',[]);
    
    for eventType = 2:-1:1
        if eventType == 1
            events = events_detected;
            outFieldName = 'from_detected_events';
        else
            events = events_gt;
            outFieldName = 'from_gt_events';
        end
        out.(outFieldName) = struct('iEv',{}, 'probWeight',{}, 'probPlate',{}, 'probHalfShelf',{}, 'probShelf',{}, 'probVision',{});

        for iEv = 1:length(events)
            if events(iEv).tE > gt.t_exit_store
                fprintf('\tEvent %d %s ignored (tE=%s after subject left the store)\n', iEv, outFieldName, events(iEv).tE);
                continue
            end
            if isempty(events(iEv).deltaW)
                fprintf('\tEvent %d %s ignored (deltaW is empty, probably there''s no weight info for tE=%s, last weight ends at %s)\n', iEv, outFieldName, events(iEv).tE, weights.t(end));
                continue
            end

            % Arrangement probability
            [probPlate, probHalfShelf, probShelf] = computeArrangementProbability(weightsPerBin, events(iEv), productArrangement(2-is5shelf));

            % Weight probability
            probWeight = computeWeightProbability(events(iEv).deltaW, weightModelParams(2-is5shelf));

            % Vision probability
            if doVision
                if events(iEv).deltaW <= 0  % Picked up something
                    tStart = weights.t(events(iEv).nB);
                    tEnd = weights.t(events(iEv).nE) + systemParams.Nvision;
                else  % Put back
                    tStart = weights.t(events(iEv).nB) - systemParams.Nvision;
                    tEnd = weights.t(events(iEv).nE);
                end
                camFrames = find((cams.t >= tStart) & (cams.t <= tEnd));

                %camPredictions = cell(1, size(cams.hands,2));  % 1xnumCams
                camPredictions = repmat({ones(length(probWeight),0)}, 1,size(cams.hands,2));
                for iFrame = 1:length(camFrames)
                    frame = camFrames(iFrame);
                    for iCam = 1:length(camPredictions)
                        hands = cams.hands{frame, iCam};
                        products = cams.products{frame, iCam};
                        if isempty(hands) || isempty(products), continue; end
                        % Sometimes there's multiple bboxes with exactly the same probability -> Keep only unique columns
                        [~,iProds]=unique(products(1:end-4,:)', 'rows');
                        products = products(:,iProds);

                        productCenters = (products(end-4 + (1:2),:) + products(end-4 + (3:4),:))/2;  % Equivalent to: products(1:2,:) + (products(3:4,:)-products(1:2,:))./2;
                        squaredDists = sum((productCenters - reshape(hands(1:2,:), 2,1,[])).^2);
                        productsToKeep = any(squaredDists <= systemParams.radHandSq(iCam), 3);
                        if sum(productsToKeep) > 0 && false
                            plotBboxesAndHands(products, hands, [], systemParams.radHandSq(iCam));
                            title(sprintf('Frame: %d, cam: %d', frame, iCam));
                        end
                        camPredictions{iCam} = [camPredictions{iCam}, 1-products(1:end-4,productsToKeep)];
                    end
                end
                camNotSeenProb = zeros(size(camPredictions{iCam},1), length(camPredictions));  % numProducts x numCams
                for iCam = 1:length(camPredictions)
                    camNotSeenProb(:,iCam) = prod(camPredictions{iCam}, 2);
                end
            else
                camNotSeenProb = zeros(size(probWeight,1), 4);
            end
            %probVision = 1 - prod(camNotSeenProb, 2);
            
            % Save results
            out.(outFieldName)(end+1,1) = struct('iEv',iEv, 'probWeight',probWeight, 'probPlate',{probPlate}, 'probHalfShelf',{probHalfShelf}, 'probShelf',probShelf, 'probVision',1-camNotSeenProb);
        end
    end
    
    fprintf('Done processing events from tStr=%s\n', tStr);
end
