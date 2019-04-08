function dispImproved(TXT,varargin)
% Prints overwritable message to the command line. If you dont want to keep
% this message, call dispImproved function with option 'keepthis'. If you want to
% keep the previous message, use option 'keepprev'. First argument must be
% the message.
% IMPORTANT! In the firt call, option 'init' must be used for initialization purposes.
% Options:
%     'init'      this must be called in the begining. Otherwise, it can overwrite the previous outputs on the command line.
%     'keepthis'    the message will be persistent, wont be overwritable,
%     'keepprev'  the previous message wont be overwritten. New message will start from next line,
%     'timestamp' current time hh:mm:ss will be appended to the begining of the message.
% Example:
%   clc;
%   fprintf('12345677890\n');
%   dispImproved('','init')      %Initialization. Does not print anything.
%   dispImproved('Time stamp will be written over this text.'); % First output
%   dispImproved('is current time.','timestamp','keepthis'); % Overwrites the previous output but this output wont be overwritten.
%   dispImproved(sprintf('*********\nDeveloped by %s\n*********','Kasim')); % does not overwrites the previous output
%   dispImproved('','timestamp','keepprev','keepthis'); % does not overwrites the previous output
%   dispImproved('this wont be overwriten','keepthis');
%   dispImproved('dummy dummy dummy');
%   dispImproved('final stat');
% % Output:
%     12345677890
%     15:15:34 is current time.
%     *********
%     Developed by Kasim
%     *********
%     15:15:34 
%     this wont be overwriten
%     final stat

% **********
% **** Options
	keepthis = false; % option for not overwriting
	keepprev = false;
	timestamp = false; % time stamp option
	init = 0; % is it initialization step?
	if ~isstr(TXT)
		return
	end
	persistent prevCharCnt;
	if isempty(prevCharCnt)
		prevCharCnt = 0;
	end
	if nargin == 0
		return
	elseif nargin > 1
		for i = 2:nargin
			eval([varargin{i-1} '=1;']);
		end
	end
	if init == 1
		prevCharCnt = 0;
		return;
	end
	if isempty(TXT) && timestamp == 0
		return
	end
	if timestamp == 1
		c = clock; % [year month day hour minute seconds]
		txtTimeStamp = sprintf('%02d:%02d:%02d ',c(4),c(5),round(c(6)));
	else
		txtTimeStamp = '';
	end
	if keepprev == 1
		prevCharCnt = 0;
	end
% *************** Make safe for fprintf, replace control charachters
	TXT = strrep(TXT,'%','%%');
	TXT = strrep(TXT,'\','\\');
% *************** Print
	TXT = [txtTimeStamp TXT];
	fprintf([repmat('\b',1, prevCharCnt) TXT]);
	nof_extra = length(strfind(TXT,'%%'));
	nof_extra = nof_extra + length(strfind(TXT,'\\'));
	nof_extra = nof_extra + length(strfind(TXT,'\n'));
	prevCharCnt = length(TXT) - nof_extra; %-1 is for \n
	if keepthis == 1
		prevCharCnt = 0;
	end
end