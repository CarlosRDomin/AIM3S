function f = plotBboxesAndHands(products, hands, imgDims, Rsquared)
    if nargin<3 || isempty(imgDims), imgDims = [720 1280]; end
    if nargin<4 || isempty(Rsquared), Rsquared = 100.^2; end
    
    f = figure; axis([0 imgDims(2) 0 imgDims(1)], 'image', 'ij'); hold on;
    for iProd = 1:size(products,2)
        rectangle('Position', [products(1:2,iProd); products(3:4,iProd)-products(1:2,iProd)]);
    end
    productCenters = (products(1:2,:) + products(3:4,:))/2;
    squaredDists = sum((productCenters - reshape(hands(1:2,:), 2,1,[])).^2);
    productsWithinR = any(squaredDists <= Rsquared, 3);
    
    scatter(productCenters(1,not(productsWithinR)), productCenters(2,not(productsWithinR)), 15, 'filled', 'MarkerFaceColor','r');
    scatter(productCenters(1,productsWithinR), productCenters(2,productsWithinR), 30, 'filled', 'MarkerFaceColor','g');
    scatter(hands(1,:), hands(2,:), 30, 'filled', 'MarkerFaceColor','b');
    viscircles(hands(1:2,:)', repmat(sqrt(Rsquared), size(hands,2),1), 'Color','b', 'LineStyle','--');
end
