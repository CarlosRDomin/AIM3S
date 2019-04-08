function out = movingAvgFilter(windowSize, in)
    a = 1;
    b = (1/windowSize)*ones(1,windowSize);
    out = filtfilt(b, a, in);
end