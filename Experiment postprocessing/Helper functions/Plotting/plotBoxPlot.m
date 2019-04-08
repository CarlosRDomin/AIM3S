function [h, hLgnd] = plotBoxPlot(score, xTickNames, legendNames, colors, boxWidth, xTickWidth)
	if nargin<4 || isempty(colors), colors = get(gca, 'ColorOrder'); end
	if nargin<5 || isempty(boxWidth), boxWidth = 0.2; end
	if nargin<6 || isempty(xTickWidth), xTickWidth = 0.2; end
	if isnumeric(xTickNames), meanPos = xTickNames; else, meanPos = 1:size(score,1); end  % Decide where the boxes will go on the x axis
	boxWidth = boxWidth.*min(abs(diff(meanPos))); xTickWidth = xTickWidth.*min(abs(diff(meanPos)));  % And rescale boxWidth and xTickWidth by the shortest x-axis offset

	% Merge all data points in a row and use the variable group to indicate which points belong to which box (1, 2, 3....)
	data = [];
	group = [];
	n = 1;
	for i = 1:size(score,1)
		for j = 1:size(score,2)
			data = [data score{i,j}];
			group = [group n*ones(1,length(score{i,j}))];
			n = n+1;
		end
	end
	
	% Plot the actual boxplot
	pos = reshape(repmat(meanPos, size(score,2),1) + xTickWidth.*linspace(-1,1, size(score,2))', 1,[]);
	colGroup = repmat(1:size(score,2), 1,size(score,1));
	h = boxplot(data, group, 'positions',pos, 'colorgroup',colGroup, 'colors',colors, 'Widths',boxWidth);
 	set(gca, 'xTick',meanPos, 'xTickLabel',xTickNames);
	set(h, {'LineWidth'},{1.5});
	y = ylim; y(1) = 0; ylim(y);  % Set lower bound to 0

	% Fill in the boxes
	b = flip(findobj(gca, 'Tag','Box'));
	for i = 1:length(b)
		c = colors(1+mod(i-1, size(score,2)),:);
	   patch(get(b(i),'XData'), get(b(i),'YData'), c, 'FaceAlpha',.65, 'EdgeColor',c, 'LineWidth',1.5);
	end

	% Add legend
	c = get(gca, 'Children');
	hLgnd = legend(flip(c(1:size(score,2))), legendNames);
end
