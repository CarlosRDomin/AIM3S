function f = plotExperimentWeights(weightsPerShelf, events_detected_and_gt, systemParams, tExitStore, plotVar, tOffs)
    if nargin<3 || isempty(systemParams), systemParams = struct('epsVar',500); end
    if nargin<4 || isempty(tExitStore), tExitStore = []; end
    if nargin<5 || isempty(plotVar), plotVar = false; end
    if nargin<6 || isempty(tOffs), tOffs = weightsPerShelf.t(1); end  % Set tOffs = seconds(0); to display actual datetime of the experiment (e.g. 19:50:26)
    fSize = 18;
    
    f = figure('Position',[0 0 700 400], 'DefaultAxesFontSize',fSize);
    needLegend = true;
    for iS = 1:size(weightsPerShelf.w, 2)
        % Plot weight on the left axis
        subtightplot(size(weightsPerShelf.w, 2), 1, size(weightsPerShelf.w, 2)-iS+1, 0.066, 0.119, 0.1); hold on;
        if plotVar
            l = plot(weightsPerShelf.t-tOffs, weightsPerShelf.wVar(:,iS), 'LineWidth', 2, 'DisplayName','Shelf weight (movVar)');
            plot(weightsPerShelf.t([1 end])-tOffs, repmat(systemParams.epsVar, 1,2), 'k--');
            yTitle = 'Shelf weight variance (g^2)';
            yL = ylim(); yL(2) = max(yL(2), 1e4); ylim(yL);
        else
            l = [
                plot(weightsPerShelf.t-tOffs, weightsPerShelf.w(:,iS), 'Color',[0.3 0.75 0.93], 'DisplayName','Shelf weight (raw)');
                plot(weightsPerShelf.t-tOffs, weightsPerShelf.wMean(:,iS), 'LineWidth', 2, 'Color',[0 0.32 0.49], 'DisplayName','Shelf weight (movMean)');
            ];
            yTitle = 'Shelf weight (g)';
        end
        if iS > 1, set(gca, 'XTickLabel',''); end  % No xLabels (only on the bottom row)
        
        % Create a right axis and plot tExitStore
        cOrder = get(gca, 'ColorOrder'); cOrder(3,:) = [0.995,0.615,0.423];
        yyaxis right; set(gca, 'ColorOrder',circshift(cOrder, -2));
        ax = get(gca, 'YAxis'); set(ax(2), 'Color','k', 'TickValues',[0 1], 'TickLabels',{'No event', 'Event'}, 'Limits',[0, 1]);
        %if ~isempty(tExitStore), plot(repmat(tExitStore-tOffs, 1,2), [0 1], 'k--', 'LineWidth', 2, 'DisplayName','tExitStore'); end
        
        % Also plot "bounding boxes" for any event in this shelf
        for iEventType = 1:length(events_detected_and_gt)
            evts = events_detected_and_gt{iEventType};
            eventIndsInShelf = find([evts.shelf] == iS);
            if iEventType == 1
                dispName = 'Detected event(s)';
                lineType = '-';
            else
                dispName = 'Ground truth event(s)';
                lineType = '-.';
            end
            for iEv = 1:length(eventIndsInShelf)
                ev = evts(eventIndsInShelf(iEv));
                s = stairs([weightsPerShelf.t(1)-seconds(1) ev.tB ev.tE weightsPerShelf.t(end)+seconds(1)]-tOffs, ... % Add seconds(1) so markers don't show up at t=0 and t=end
                    [0 1 0 0], [lineType getMarker(iEv)], 'Color',cOrder(2+iEventType,:), 'LineWidth', 1.75, 'DisplayName',dispName);
                if iEv == 1, l = [l; s]; end  % Add the first event to the legend
            end
        end
        if ~isempty(tExitStore), tLast = tExitStore; else tLast = weightsPerShelf.t(end); end
        set(gca, 'YGrid',~plotVar, 'XLim',[weightsPerShelf.t(1) tLast]-tOffs);
        if isdatetime(tOffs), set(get(gca, 'XAxis'), 'TickLabelFormat','mm:ss'); end  % Remove hours if xAxis is Duration
        title(sprintf('Shelf %d', iS));
        if needLegend && length(l) == 3+(~plotVar)
            L = legend(l, 'Orientation','horizontal', 'FontSize',fSize-4);
            p = get(L, 'Position'); p(2) = 0.99 - p(4); p(1) = (1-p(3))/2; set(L, 'Position',p);  % Centered up top
            needLegend = false;
        end
    end
    suplabel(sprintf('Time (%s)', get(get(gca,'XAxis'),'TickLabelFormat')), 'x', [], -0.015); %, 'FontSize',fSize);
    suplabel(yTitle, 'y', [], 0.019); %, 'FontSize',fSize);
end

function marker = getMarker(iEv)
    markers = ' o^v><+*xsdph';
    marker = markers(1 + mod(iEv-1,length(markers)));
end
