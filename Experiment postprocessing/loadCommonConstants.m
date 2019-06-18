addpath(genpath('.'));	% Make sure all folders and subfolders are added to the path
cdToThisScriptsDirectory();	% Change directory to the folder containing this script
FIG_FOLDER = 'figures';
DATA_FOLDER = '../Dataset';
experimentType = 'Evaluation';
systemParams = struct('movAvgWindowInSamples',60, 'epsVar',500, 'epsMeanShelf',10, 'epsMeanPlate',5, 'N_high',30, 'N_low',30, 'Nvision',seconds(3), 'radHandSq',([100 100 100 200]/2).^2);
