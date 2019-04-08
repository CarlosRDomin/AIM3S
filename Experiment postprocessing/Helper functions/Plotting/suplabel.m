function [ax, h] = suplabel(text, whichLabel, supAxes, axMargin, varargin)
% Places text as a title, xlabel, or ylabel on a group of subplots.
% Returns a handle to the label and a handle to the axis.
%  [ax,h]=suplabel(text,whichLabel,supAxes)
% returns handles to both the axis and the label.
%  ax=suplabel(text,whichLabel,supAxes)
% returns a handle to the axis only.
%  suplabel(text) with one input argument assumes whichLabel='x'
%
% whichLabel is any of 'x', 'y', 'yy', or 't', specifying whether the
% text is to be the xlable, ylabel, right side y-label,
% or title respectively.
%
% supAxes is an optional argument specifying the Position of the
%  "super" axes surrounding the subplots.
%  supAxes defaults to [.08 .08 .84 .84]
%  specify supAxes if labels get chopped or overlay subplots
%
% EXAMPLE:
%  subplot(2,2,1);ylabel('ylabel1');title('title1')
%  subplot(2,2,2);ylabel('ylabel2');title('title2')
%  subplot(2,2,3);ylabel('ylabel3');xlabel('xlabel3')
%  subplot(2,2,4);ylabel('ylabel4');xlabel('xlabel4')
%  [ax1,h1]=suplabel('super X label');
%  [ax2,h2]=suplabel('super Y label','y');
%  [ax3,h2]=suplabel('super Y label (right)','yy');
%  [ax4,h3]=suplabel('super Title'  ,'t');
%  set(h3,'FontSize',30)
%
% SEE ALSO: text, title, xlabel, ylabel, zlabel, subplot,
%           suptitle (Matlab Central)

% Author: Ben Barrowes <barrowes@alum.mit.edu>

%modified 3/16/2010 by IJW to make axis behavior re "zoom" on exit same as
%at beginning. Requires adding tag to the invisible axes


	% currAx = findobj(gcf, 'type','axes', '-not','tag','suplabel');	% Speedup: No need to bring all other axes to front, just this one to back

	if nargin < 3, supAxes = []; end
	if nargin < 4 || isempty(axMargin), axMargin=0.01; end
	if length(axMargin) < 2, axMargin = repmat(axMargin, 1,2); end
	if length(axMargin) < 4, axMargin = repmat(axMargin(1:2), 1,2); end
	if isempty(supAxes) || ~isnumeric(supAxes(1))	% supAxes indicates the axes position if 1x4 double, but the set of axes the suplabel applies to otherwise
		if ~isempty(supAxes), ah = supAxes(:); else, ah = findall(gcf, 'type','axes', 'Visible','on'); end
		if ~isempty(ah)
			set(ah(~strcmpi(get(ah, 'Units'), 'Normalized')), 'Units','Normalized');	% Speedup: only set units to axes that don't have Normalized units yet (it's a slow operation)
			thisPos = cell2mat(get(ah, 'Position'));
			currAxBounds = [min(thisPos(:,1:2)) max(thisPos(:,1:2)+thisPos(:,3:4))];
% 			currAxBounds = NaN(1,4);
% 			for ii=1:length(ah)
% 				thisPos=get(ah(ii),'Position');
% 				currAxBounds(1:2) = min(currAxBounds(1:2), thisPos(1:2));
% 				currAxBounds(3:4) = max(currAxBounds(3:4), thisPos(1:2)+thisPos(3:4));
% 			end
% 			supAxes=[leftMin-axMargin,bottomMin-axMargin,leftMax-leftMin+axMargin*2,bottomMax-bottomMin+axMargin*2];
			supAxes = [currAxBounds(1:2)-axMargin(1:2), currAxBounds(3:4)-currAxBounds(1:2)+axMargin(1:2)+axMargin(3:4)];
		else
			supAxes=[.13, .11, .775, .815] + axMargin.*[-1 -1 1 1];
		end
	end
	if nargin < 2, whichLabel = 'x';  end
	if nargin < 1, help(mfilename); return; end

	if ((iscell(text) && ~ischar([text{:}])) || (~iscell(text) && ~ischar(text))) || ~ischar(whichLabel)
		error('text and whichLabel must be strings');
	end
	whichLabel=lower(whichLabel);

	ax=axes('Units','Normal', 'Position',supAxes, 'Visible','off', 'tag','suplabel');
	if strcmp('t',whichLabel)
		title(text, 'Visible','on', varargin{:});
		% set(get(ax,'Title'), 'Visible','on');
	elseif strcmp('x',whichLabel)
		xlabel(text, 'Visible','on', varargin{:});
		% set(get(ax,'XLabel'), 'Visible','on');
	elseif strcmp('y',whichLabel)
		ylabel(text, 'Visible','on', varargin{:});
		% set(get(ax,'YLabel'), 'Visible','on');
	elseif strcmp('yy',whichLabel)
		ylabel(text, 'Visible','on', varargin{:});
		set(ax,'YAxisLocation','right');
		%set(get(ax,'YLabel'), 'Visible','on');
	end

	% for k=length(currAx):-1:1, axes(currAx(k)); end	% axes(currAx(1));	% Restore gca	% restore all other axes
	f=gcf; f.Children = [f.Children(2:end); f.Children(1)];	% Send the axes to the back (so user can interact with everything else: zoom, pan, etc.)
	
	if (nargout < 2)
		return;
	elseif strcmp('t', whichLabel)
		h=get(ax,'Title');
	elseif strcmp('x', whichLabel)
		h=get(ax,'XLabel');
	elseif strcmp('y', whichLabel) || strcmp('yy', whichLabel)
		h=get(ax,'YLabel');
	end
end