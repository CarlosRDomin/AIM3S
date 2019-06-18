function t = parseStrDatetime(t_str, fmt)
    if nargin<2 || isempty(fmt), fmt = 'yyyy-MM-dd HH:mm:ss.SSSSSS'; end
    
    t = datetime(t_str(1:length(fmt)), 'InputFormat',fmt);
end
