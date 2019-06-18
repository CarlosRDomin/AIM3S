function f = visualizeEventsTiming(events, weights, gt, systemParams, tStart, tEnd)
    if nargin<4 || isempty(systemParams), systemParams = []; end
    if nargin<5 || isempty(tStart), tStart = parseStrDatetime(gt.weight_to_cam_t_offset); end
    if nargin<6 || isempty(tEnd), tEnd = gt.t_exit_store; end
    events_gt = gt2events(gt, weights, 1, systemParams);
    f = figure;
    
    for iEventType = 1:2
        if iEventType == 1
            evts = events;
            dispName = 'Detected events';
        else
            evts = events_gt;
            dispName = 'Ground truth events';
        end
        
        subplot(2,1,iEventType); title(dispName); hold on;
        for iEv = 1:length(evts)
            plotEvent(evts(iEv).tB, evts(iEv).tE, tStart, tEnd);
        end
        xlim([tStart tEnd]);
    end
end

function plotEvent(tS, tE, tStart, tEnd)
    if tS < tEnd && tE <= tEnd
        stairs([tStart tS tE tEnd], [0 1 0 0]);
    end
end