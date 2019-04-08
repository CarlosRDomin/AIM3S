function cdToWorkingDirectory()
    % Set [path to] Current Location
    %
    % DESCRIPTION:
    %   Changes the directory to the path of the current file in editor.

    try
        editor_service = com.mathworks.mlservices.MLEditorServices;
        editor_app = editor_service.getEditorApplication;
        active_editor = editor_app.getActiveEditor;
        storage_location = active_editor.getStorageLocation;
        file = char(storage_location.getFile);
        path = fileparts(file);
        cd(path);
		addpath(genpath(path));	% Add all subdirectories to the path
    catch e
        error('cdToWorkingDirectory failed.\n\n%s', e.message);
    end
end