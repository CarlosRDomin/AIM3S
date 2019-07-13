function cams = loadCamsData(tStr, experimentType, DATA_FOLDER, loadBgndMask)
    if nargin<2 || isempty(experimentType), experimentType = 'Evaluation'; end
	if nargin<3 || isempty(DATA_FOLDER), DATA_FOLDER = '../Dataset'; end
	if nargin<4 || isempty(loadBgndMask), loadBgndMask = false; end
    EXP_PREFIX = [DATA_FOLDER '/' experimentType '/' tStr '/'];
    fprintf('Loading camera(s) from experiment t=%s...\n', tStr);
	
	% Load multicam info (how many cams and time of each frame)
	multicam = readHDF5([EXP_PREFIX 'multicam_' tStr '.h5']);
    t = (parseStrDatetime(multicam.t_start):seconds(1/double(multicam.fps)):parseStrDatetime(multicam.t_end))';
    aux = {cell(size(multicam.frame_nums))};  % Every field in cams will have a cell entry per frame (row) and cam (col)
    cams = struct('t',t, 'frameNums',multicam.frame_nums, 'hands',aux, 'products',aux, 'bgndMask',aux);
    
    % Fill in info for each cam
    for iCam = 1:size(multicam.frame_nums,2)
        fprintf('\tLoading cam %d data (pose, products, background subtraction masks)\n', iCam);
        CAM_SUFFIX = ['cam' num2str(iCam) '_' tStr];
        camInfo = readHDF5([EXP_PREFIX CAM_SUFFIX '.h5']);
        prodInfo = readHDF5([EXP_PREFIX CAM_SUFFIX '_mask_objdet' '.h5']);
        framesWithProds = fieldnames(prodInfo);
        for iFrame = 1:length(multicam.frame_nums)
            frameNum = multicam.frame_nums(iFrame, iCam)+1;  % Add 1 since indexing is 0-based
            frameStr = sprintf('frame%05d', frameNum);
            
            % Hands position: 4x(# hands found), where rows indicate:
            % xHand, yHand, personId, jointType (=4 for RHand, 7 for LHand)
            cams.hands{iFrame,iCam} = double(camInfo.hands.(frameStr));
            
            % Products found: (4+Nproducts)x(# of products found in that frame) matrix
            if ismember(frameStr, framesWithProds)
                cams.products{iFrame,iCam} = double(prodInfo.(frameStr));
            end
            
            % Background mask (since it uses 1280x720 = ~1MB per frame,
            % don't load it unless we really need it -> Save filename only)
            bgndMaskFileName = [EXP_PREFIX 'background_masks/' CAM_SUFFIX '_mask_' frameStr '.png'];
            if loadBgndMask
                cams.bgndMask{iFrame,iCam} = imread(bgndMaskFileName) > 127;
            else
                cams.bgndMask{iFrame,iCam} = bgndMaskFileName;
            end
        end
    end	
end
