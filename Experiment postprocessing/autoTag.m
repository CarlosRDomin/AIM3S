function out = autoTag(tStr, experimentType, DATA_FOLDER, binWidth, systemParams, productArrangement, weightModelParams)
    if nargin<2 || isempty(experimentType), experimentType = 'Evaluation'; end
    if nargin<3 || isempty(DATA_FOLDER), DATA_FOLDER = '../Dataset'; end
    if nargin<4 || isempty(binWidth), binWidth = 1; end
    if nargin<5 || isempty(systemParams), systemParams = struct('movAvgWindowInSamples',60, 'epsVar',500, 'epsMeanShelf',10, 'epsMeanPlate',5, 'N_high',30, 'N_low',30, 'Nvision',seconds(3), 'radHandSq',([100 100 100 200]/2).^2); end
    if nargin<6 || isempty(productArrangement) || isempty(weightModelParams), [productArrangement, weightModelParams] = loadProductModels(DATA_FOLDER); end
    OUTPUT_FOLDER = ['../AutoTag/' tStr];
    if ~exist(OUTPUT_FOLDER, 'dir'), mkdir(OUTPUT_FOLDER); end
    
    [weights, weightsPerBin, ~, cams, events_detected, gt] = loadExperiment(tStr, experimentType, DATA_FOLDER, binWidth, systemParams, true);
    is5shelf = (size(weights.w, 2) == 5);
    events_gt = gt2events(gt, weights, binWidth, systemParams);
    videos = cell(size(cams.frameNums, 2), 2);
    for iCam = 1:size(videos, 1)
        videoPrefix = [DATA_FOLDER '/' experimentType ' full contents/' tStr '/cam' num2str(iCam) '_' tStr];
        videos{iCam,1} = VideoReader([videoPrefix '.mp4']);
        videos{iCam,2} = VideoReader([videoPrefix '_mask.mp4']);
    end 
    out = struct('tStr',tStr, 'gt',gt, 'events_gt',events_gt, 'events_detected',events_detected, 'from_detected_events',[], 'from_gt_events',[]);
    
    for eventType = 2
        if eventType == 1
            events = events_detected;
            outFieldName = 'from_detected_events';
            eventTypeStr = 'aim3s';
        else
            events = events_gt;
            outFieldName = 'from_gt_events';
            eventTypeStr = 'gt';
        end
        out.(outFieldName) = struct('iEv',{}, 'probWeight',{}, 'probPlate',{}, 'probHalfShelf',{}, 'probShelf',{}, 'probVision',{});

        for iEv = 1:length(events)
            fprintf('Processing event %d (out of %d)...\n', iEv, length(events));
            if events(iEv).tE > gt.t_exit_store && false
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
            if events(iEv).deltaW <= 0  % Picked up something
                tStart = weights.t(events(iEv).nB);
                tEnd = weights.t(events(iEv).nE) + systemParams.Nvision;
            else  % Put back
                tStart = weights.t(events(iEv).nB) - systemParams.Nvision;
                tEnd = weights.t(events(iEv).nE);
            end
            camFrames = find((cams.t >= tStart) & (cams.t <= tEnd));

            EVENT_PREFIX = sprintf('%s/event_%02d_%s', OUTPUT_FOLDER, iEv, eventTypeStr); 
            mkdir(EVENT_PREFIX);
            camPredictions = repmat({ones(length(probWeight),0)}, 1,size(cams.hands,2));  % 1xnumCams
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
                    if sum(productsToKeep) <= 0, continue; end
                    if false
                        plotBboxesAndHands(products, hands, [], systemParams.radHandSq(iCam));
                        title(sprintf('Frame: %d, cam: %d', frame, iCam));
                    end
                    
                    frameNum = double(cams.frameNums(frame, iCam)-1);
                    videos{iCam,1}.CurrentTime = frameNum/videos{iCam,1}.FrameRate;
                    videos{iCam,2}.CurrentTime = frameNum/videos{iCam,2}.FrameRate;
                    img = videos{iCam,1}.readFrame;
                    imgMask = videos{iCam,2}.readFrame;
                    bboxes = round(products(end-3:end, productsToKeep));
                    for iBbox = 1:size(bboxes,2)
                        imgCropped = img(bboxes(2,iBbox):bboxes(4,iBbox), bboxes(1,iBbox):bboxes(3,iBbox), :);
                        imgMaskCropped = imgMask(bboxes(2,iBbox):bboxes(4,iBbox), bboxes(1,iBbox):bboxes(3,iBbox), :);
                        outPrefix = sprintf('%s/cam%d_f%05d_%02d', EVENT_PREFIX, iCam, frameNum+1, iBbox);
                        imwrite(imgCropped, [outPrefix '.jpg']);
                        imwrite(imgMaskCropped, [outPrefix '_mask.jpg']);
                    end
                    writeHDF5dataset([OUTPUT_FOLDER '/autoTag_' tStr '.h5'], sprintf('/cam%d/frame%05d', iCam, frameNum+1), products(:, productsToKeep));
                    camPredictions{iCam} = [camPredictions{iCam}, 1-products(1:end-4,productsToKeep)];
                end
            end
            camNotSeenProb = zeros(size(camPredictions{iCam},1), length(camPredictions));  % numProducts x numCams
            for iCam = 1:length(camPredictions)
                camNotSeenProb(:,iCam) = prod(camPredictions{iCam}, 2);
            end
            %probVision = 1 - prod(camNotSeenProb, 2);
            
            % Save results
            out.(outFieldName)(end+1,1) = struct('iEv',iEv, 'probWeight',probWeight, 'probPlate',{probPlate}, 'probHalfShelf',{probHalfShelf}, 'probShelf',probShelf, 'probVision',1-camNotSeenProb);
        end
    end
    
    fprintf('Done processing events from tStr=%s\n', tStr);
end
