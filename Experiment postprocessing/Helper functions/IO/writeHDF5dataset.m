function writeHDF5dataset(fileName, datasetName, data)
    try
        h5create(fileName, datasetName, size(data));
    catch me
        % Ignore errors if dataset already existed
        if ~strcmp(me.identifier,'MATLAB:imagesci:h5create:datasetAlreadyExists')
            rethrow(me);
        end
    end
    h5write(fileName, datasetName, data);
end
