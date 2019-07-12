function A = areaUnderTwoGaussians(mu1, sigma1, mu2, sigma2, plotLims)
    if nargin<5 || isempty(plotLims), plotLims = []; end  % plotLims can specify [xMin xMax] if you want to plot both gaussians
    
    % Find the intersection of both pdf's -> We used:
    % syms x mu1 sigma1 mu2 sigma2
    % solve(exp(-(x-mu1)^2/(2*sigma1^2))/(sigma1*sqrt(2*pi)) == exp(-(x-mu2)^2/(2*sigma2^2))/(sigma2*sqrt(2*pi)), x)
    if sigma1 == sigma2  % Only one intersection point
        xSol = (mu1^2 - mu2^2)/(2*mu1 - 2*mu2);
        if isnan(xSol), xSol = mu1; end  % Only happens if mu1==mu2 -> A=1 regardless of xSol
        xEv = xSol-1;  % Evaluate the pdf on the left (lower) side of the intersection
        if normpdf(xEv, mu1, sigma1) < normpdf(xEv, mu2, sigma2)
            A = normcdf(xSol, mu1, sigma1) + normcdf(xSol, mu2, sigma2, 'upper');
        else
            A = normcdf(xSol, mu2, sigma2) + normcdf(xSol, mu1, sigma1, 'upper');
        end
    else  % 2 intersection points
        xSol = [ (mu2*sigma1^2 - mu1*sigma2^2 + sigma1*sigma2*(2*sigma2^2*log(sigma2/sigma1) - 2*sigma1^2*log(sigma2/sigma1) - 2*mu1*mu2 + mu1^2 + mu2^2)^(1/2))/(sigma1^2 - sigma2^2);
                -(mu1*sigma2^2 - mu2*sigma1^2 + sigma1*sigma2*(2*sigma2^2*log(sigma2/sigma1) - 2*sigma1^2*log(sigma2/sigma1) - 2*mu1*mu2 + mu1^2 + mu2^2)^(1/2))/(sigma1^2 - sigma2^2)];
        if xSol(1) > xSol(2), xSol = flip(xSol); end  % Sort xSol -> [xSolLow xSolHigh]
        xEv = sum(xSol)/2;  % Evaluate the pdf in between the 2 intersection points
        if normpdf(xEv, mu1, sigma1) > normpdf(xEv, mu2, sigma2)
            muEnds = mu1; sigmaEnds = sigma1; muMiddle = mu2; sigmaMiddle = sigma2;
        else
            muEnds = mu2; sigmaEnds = sigma2; muMiddle = mu1; sigmaMiddle = sigma1;
        end
        A = normcdf(xSol(1), muEnds, sigmaEnds) + diff(normcdf(xSol, muMiddle, sigmaMiddle)) + normcdf(xSol(2), muEnds, sigmaEnds, 'upper');
    end
    if A > 1, A=0; end  % Due to loss in precission, sometimes normpdf(xEv, mu1, sigma1) = normpdf(xEv, mu2, sigma2) = 0. In that case, A would be (wrongly) computed as 2, but should be 0
    
    if ~isempty(plotLims)
        disp(calc_overlap_twonormal(mu1, sigma1, mu2, sigma2, plotLims(1):0.05:plotLims(2)));
    end
end

function A = calc_overlap_twonormal(mu1, s1, mu2, s2, xRange)
    y = [normpdf(xRange,mu1,s1); normpdf(xRange,mu2,s2)]';
    yMin = min(y, [], 2);
    plot(xRange, y); hold on;
    area(xRange, yMin);
    overlap = cumtrapz(xRange, yMin);
    A = overlap(end);
end
