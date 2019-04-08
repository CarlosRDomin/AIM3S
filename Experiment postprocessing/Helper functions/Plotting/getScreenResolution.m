function s = getScreenResolution(accountForDock)
	if true
		set(0, 'Units','pixels');
		s = get(0, 'ScreenSize');

		if nargin<1 || accountForDock
			dockH = 80;
		else
			dockH = 0;
		end
		s = [1, dockH+1, s(3), s(4)-dockH];
	else
		screenW = java.awt.Toolkit.getDefaultToolkit.getScreenSize.getWidth;
        screenH = java.awt.Toolkit.getDefaultToolkit.getScreenSize.getHeight;
		
		f = figure('Units','pixels'); 	% jFrame seems to need some time to initialize (otherwise it throws NullPointerException), keep trying until we succeed (or maxTries is reached)
		warnOrigState = warning('query', 'MATLAB:HandleGraphics:ObsoletedProperty:JavaFrame');
		warning('off', 'MATLAB:HandleGraphics:ObsoletedProperty:JavaFrame');				% Disable warning message
		jFrame = get(f, 'JavaFrame');
		warning(warnOrigState.state, 'MATLAB:HandleGraphics:ObsoletedProperty:JavaFrame');	% Reset warning state
		pause(0.00001);
		jFrame.setMaximized(true);
		pause(2);
		s = get(f, 'Position');
		close(f);
	end
end
