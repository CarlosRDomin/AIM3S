function data = readNPZ(filename)
	[baseFolder, name, ~] = fileparts(filename);
	zipFolder = [baseFolder '/' name]; zipFolder = 'temp';
	unzip(filename, zipFolder);
	outFiles = dir(zipFolder); outFiles = outFiles(3:end);	% Ignore '.' and '..'
    data = cell(length(outFiles), 2);
% 	data = struct();

	warnV = warning('query', 'verbose'); warning('off','verbose');	% Adjust warning settings to be less annoying (settings will be reset later on)
	warnB = warning('query', 'backtrace'); warning('off','backtrace');
	for i = 1:length(outFiles)
		f = outFiles(i).name;
		[~,magnitude,~] = fileparts(f);
		try
			y = readNPY([zipFolder '/' f]);
		catch
			warning(['Couldn''t load magnitude ''' magnitude ''', make sure its format is valid (eg, datetime is not allowed, should convert to float first)']);
			y = '<Error loading>';
		end
% 		data.(magnitude) = y;
        data(i,:) = {magnitude, y};
	end
	rmdir(zipFolder, 's'); % Remove folder and its contents ('s' argument)
	warning(warnV); warning(warnB);	% Restore warning settings
	data = cell2struct(data(:,2), data(:,1));
end
