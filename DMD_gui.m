function DMD_gui

% Connect and initialize DMD
D=DMD('debug',1);

D.fwVersion

% Stop any existing sequences
D.patternControl(0);

% Set display mode to 3 (Pattern-on-the-fly);
D.setMode(3);


% Stop any existing sequences
D.patternControl(0);

% Disconnect from DMD
D.delete;



end

