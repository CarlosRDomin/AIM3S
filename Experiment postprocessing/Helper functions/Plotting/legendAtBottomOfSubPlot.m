%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper function to add a legend at the bottom of the current subplot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function l = legendAtBottomOfSubPlot(legendStr, ax, margin)
	if nargin<2 || isempty(ax)
		ax = gca;
	end
	if nargin<3 || isempty(margin)
		margin = [0.03 0.02];
	end
	if length(margin)<2, margin = repmat(margin, 1,2); end
	if length(margin)<3, margin = [margin, margin(1)]; end
	
	l = legend(ax, legendStr, 'Location','SouthOutside', 'Orientation','horizontal');
	set(l, 'FontSize',15, 'Units','Normalized');
	p = 0.03*ones(1,4);
	p(1:2) = margin(1:2);
	p(3) = 1 - margin(1) - margin(3);
	h = l.Position(4);
	p(4) = h;
	set(l, 'Position',p);
end
