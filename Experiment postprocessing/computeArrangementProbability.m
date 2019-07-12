function [probBin, probHalfShelf, probShelf] = computeArrangementProbability(weightsPerBin, event, arrangementInfo)
    numPlatesPerShelf = size(arrangementInfo.plate, ndims(arrangementInfo.plate));  % Last dimension of arrangementInfo.plate indicates how many plates there are in a shelf
    
    probBin = cell(length(weightsPerBin),1);
    probHalfShelf = cell(length(weightsPerBin),1);
    for iBinW = 1:length(weightsPerBin)
        w = weightsPerBin(iBinW).wMean;
        binWidth = numPlatesPerShelf/size(w, ndims(w));
        
        if false % Old method
            plateToBinAssignment = reshape(1:numPlatesPerShelf, binWidth,[]);
            plateToHalfShelfAssignment = reshape(1:numPlatesPerShelf, [],2);

            % Compute the plate and half-shelf indices to retrieve the item arrangement from
            eventPlates = reshape(plateToBinAssignment(:,event.bins{iBinW}), 1,[]);
            eventHalfShelves = any(ismember(plateToHalfShelfAssignment, eventPlates), 1);

            % Find out which items are in the eventBins, in the half-shelves corresponding to those eventBins, and in the eventShelf
            inBin = any(arrangementInfo.plate(:, event.shelf, eventPlates), 3);
            inHalfShelf = any(arrangementInfo.halfShelf(:, event.shelf, eventHalfShelves), 3);
        else
            contribBins = abs(sign(event.deltaW)*diff(w([event.nB event.nE], event.shelf, :)));
            contribPlates = reshape(repmat(contribBins, 1,binWidth), 1,1,[]);  % Repeat each bin's contribution binWidth times (e.g. [0.1 0.3] for binWidth=3 would become [0.1 0.1 0.1 0.3 0.3 0.3])
            
            % Figure out which bins contribute to which half shelf
            if mod(size(w, ndims(w)), 2) == 0  % Even number of bins -> Straightforward
                contribHalfShelves = sum(reshape(contribBins, 1, [], 2), 2);
            else  % Odd number of bins -> Middle bin contributes to both halves (e.g. [0.1 0.3 0.2] -> [0.1+0.3, 0.3+0.2])
                middleIdx = ceil(size(w, ndims(w))/2);
                contribHalfShelves = sum(reshape(contribBins(:,:,[1:middleIdx middleIdx:end]), 1, [], 2), 2);
            end
            
            inBin = sum(arrangementInfo.plate(:,event.shelf,:).*contribPlates, 3);
            inHalfShelf = sum(arrangementInfo.halfShelf(:,event.shelf,:).*contribHalfShelves, 3);
        end

        % Normalize to obtain probability (e.g. if there were 3 items, each gets 1/3 prob)
        probBin{iBinW} = inBin./sum(inBin);
        probHalfShelf{iBinW} = inHalfShelf./sum(inHalfShelf);

        % Should never happen, but just in case...
        if sum(inBin) == 0
            warning('No items placed on the bin that triggered this event (shelf=%d, bin=%s)', eventShelf, mat2str(eventBins{iBinW}));
        end
    end
    
    inShelf = arrangementInfo.shelf(:, event.shelf);
    probShelf = inShelf./sum(inShelf);
end
