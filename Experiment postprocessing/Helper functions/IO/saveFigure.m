function saveFigure(f, figName, FIG_FOLDER)
    prefix = sprintf('%s/%s', FIG_FOLDER, figName);
    
    % Set PaperSize to match width and height from PaperPosition (since PaperPositionMode='auto', it matches figure dimensions in inches)
    p = get(f, 'PaperPosition'); set(f, 'PaperSize',p(3:4));
    
    % Save figure in pdf and fig formats
    saveas(f, [prefix '.pdf']);
    savefig(f, [prefix '.fig']);
    fprintf('Saved figure as %s\n', prefix);
end
