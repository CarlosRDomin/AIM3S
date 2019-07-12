function [productArrangement, weightModelParams, productsInfo] = loadProductModels(DATA_FOLDER, numPlates)
    if nargin<1 || isempty(DATA_FOLDER), DATA_FOLDER = '../Dataset'; end
    if nargin<2 || isempty(numPlates), numPlates = 12; end
    
    productJson = jsondecode(fileread(sprintf('%s/product_info.json', DATA_FOLDER)));
    productArrangement = struct('plate',[], 'halfShelf',[], 'shelf',[], 'whatsInEachPlate',[], 'numShelves',{5;8});
    weightModelParams = struct('mean',[], 'std',[], 'numShelves',{5;8});
    productsInfo = cell(length(productArrangement), 1);

    for i = 1:length(productArrangement)
        if productArrangement(i).numShelves == 5
            productInfo = productJson.products(1:33);
        else
            productInfo = productJson.products(34:end);
        end
        
        % Save productInfo as a struct array
        %  (for that, all entries in productInfo need to have the same fields; 
        %  some items don't have a 'training_id' -> initialize it to [] first)
        fName = 'training_id';
        for iP = 1:length(productInfo)
            if ~isfield(productInfo{iP}, fName)
                productInfo{iP}.(fName) = [];
            end
        end
        productsInfo{i} = [productInfo{:}]';  % Convert cell array of structs to struct array

        % Compute weight model: mu and sigma of Gaussian for each product
        mu = zeros(length(productInfo), 1);
        sigma = zeros(length(productInfo), 1);
        for j = 1:length(productInfo)
            [mu(j), sigma(j)] = normfit(productInfo{j}.weights);
        end
        weightModelParams(i).mean = mu;
        weightModelParams(i).std = sigma;

        % Compute arrangement model: which items are in which plates
        inPlate = false(length(productInfo), productArrangement(i).numShelves, numPlates);
        for j = 1:length(productInfo)
            for k = 1:length(productInfo{j}.arrangement)
                inPlate(j, productInfo{j}.arrangement(k).shelf_id, productInfo{j}.arrangement(k).plate_ids) = true;
            end
        end
        % Trick for half shelf: reshape inPlate to have 2 in the last dimension,
        % which will create two halves of numPlates/2 plates in the 2nd-to-last dimension
        % Then find how many different items there are in each half
        inHalfShelf = squeeze(any(reshape(inPlate, size(inPlate,1), size(inPlate,2), [], 2), 3));
        inShelf = any(inPlate, 3);

        %%% Posterior arrangement per plate: 1/{how many items in plate}
        %%probPlate = inPlate./sum(inPlate,1); probPlate(isnan(probPlate)) = 0;

        %%% Posterior arrangement per half shelf: 1/{how many items in half shelf}
        %%%  -> Take each half (trick: reshape to have 2 in the last dimension, which will create two halves of numPlates/2 plates)
        %%%     and find how many different items there are in each half -> Normalize
        %%inShelfHalves = reshape(inPlate, length(productInfo), productArrangement(i).numShelves, [], 2);
        %%probHalfShelf = any(reshape(inPlate, size(inPlate,1), size(inPlate,2), [], 2), 3); probHalfShelf = squeeze(probHalfShelf./sum(probHalfShelf, 1)); probHalfShelf(isnan(probHalfShelf)) = 0;

        %%% Posterior arrangement per shelf: 1/{how many items in shelf}
        %%probShelf = any(inPlate, 3); probShelf = probShelf./sum(probShelf, 1); probShelf(isnan(probShelf)) = 0;

        productArrangement(i).plate = inPlate;
        productArrangement(i).halfShelf = inHalfShelf;
        productArrangement(i).shelf = inShelf;

        % For reference, also create a numShelves x numPlates cell with which items are in each plate
        productArrangement(i).whatsInEachPlate = cell(productArrangement(i).numShelves, numPlates);
        for j = 1:length(productInfo)
            for k = 1:length(productInfo{j}.arrangement)
                for l = 1:length(productInfo{j}.arrangement(k).plate_ids)
                    productArrangement(i).whatsInEachPlate{productInfo{j}.arrangement(k).shelf_id, productInfo{j}.arrangement(k).plate_ids(l)}(end+1) = productInfo{j}.id;
                end
            end
        end
    end
end
