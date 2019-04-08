function cdToThisScriptsDirectory()
    % Set [path to] the folder containing the file calling this script
    %
    % DESCRIPTION:
    %   Changes the directory to the path of the latest file being executed.

    try
        stack = dbstack('-completenames');
		if length(stack) < 2, return; end % Avoid errors when script is called from "Run section"
		file = stack(2).file; % stack(1) is this file, stack(2) is whoever calling
        path = fileparts(file);
        cd(path);
		addpath(genpath(path));	% Add all subdirectories to the path
    catch e
        error('cdToThisScriptsDirectory failed.\n\n%s', e.message);
    end
end