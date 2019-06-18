function [probPlate, probHalfShelf, probShelf] = computeArrangementProbability(eventShelf, eventBins, arrangementInfo, binWidth)
    if nargin<4 || isempty(binWidth), binWidth = 1; end
    numPlatesPerShelf = size(arrangementInfo.plate, ndims(arrangementInfo.plate));  % Last dimension of arrangementInfo.plate indicates how many plates there are in a shelf
    
    probPlate = cell(length(binWidth),1);
    probHalfShelf = cell(length(binWidth),1);
    for iBinW = 1:length(binWidth)
        plateToBinAssignment = reshape(1:numPlatesPerShelf, binWidth(iBinW),[]);
        plateToHalfShelfAssignment = reshape(1:numPlatesPerShelf, [],2);

        % Compute the plate and half-shelf indices to retrieve the item arrangement from
        eventPlates = reshape(plateToBinAssignment(:,eventBins{iBinW}), 1,[]);
        eventHalfShelves = any(ismember(plateToHalfShelfAssignment, eventPlates), 1);

        % Find out which items are in the eventBins, in the half-shelves corresponding to those eventBins, and in the eventShelf
        inPlate = any(arrangementInfo.plate(:, eventShelf, eventPlates), 3);
        inHalfShelf = any(arrangementInfo.halfShelf(:, eventShelf, eventHalfShelves), 3);

        % Normalize to obtain probability (e.g. if there were 3 items, each gets 1/3 prob)
        probPlate{iBinW} = inPlate./sum(inPlate);
        probHalfShelf{iBinW} = inHalfShelf./sum(inHalfShelf);

        % Should never happen, but just in case...
        if sum(inPlate) == 0
            warning('No items placed on the bin that triggered this event (shelf=%d, bin=%s)', eventShelf, mat2str(eventBins{iBinW}));
        end
    end
    
    inShelf = arrangementInfo.shelf(:, eventShelf);
    probShelf = inShelf./sum(inShelf);
end
