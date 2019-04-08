function data = readHDF5(filename, root)
	if nargin<2 || isempty(root)
		root = '/';
	end

	info = h5info(filename, root);
	data = struct();  % Initialize output struct
	
	% Fill in data from attributes
	for i = 1:length(info.Attributes)
		data.(info.Attributes(i).Name) = info.Attributes(i).Value;
		if iscell(data.(info.Attributes(i).Name)) && length(data.(info.Attributes(i).Name))==1
			data.(info.Attributes(i).Name) = data.(info.Attributes(i).Name){:};
		end
	end
	
	% Fill in data from datasets
	for i = 1:length(info.Datasets)
		fieldName = info.Datasets(i).Name;
		data.(fieldName) = h5read(filename, [root '/' fieldName]);
	end
	
	% Fill in data from groups, recursively
	for i = 1:length(info.Groups)
		groupName = info.Groups(i).Name;
		groupPath = strsplit(groupName, '/');
		data.(groupPath{end}) = readHDF5(filename, groupName);
	end
end
