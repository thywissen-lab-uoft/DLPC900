function DMD_gui
% Add all subdirectories for this m file
curpath = fileparts(mfilename('fullpath'));
dmdDir=fullfile(curpath,'DMDConnect');
addpath(dmdDir);addpath(genpath(dmdDir))   

% Connect and initialize DMD
D=DMD('debug',0);

% Display firmware version
D.fwVersion

% Stop any existing sequences
D.patternControl(0);

% Set display mode to 3 (Pattern-on-the-fly);
D.setMode(3);

%%%%% Define Pattern Settings
pat=struct;
pat.idx             = 0;        % pattern index
pat.exposureTime    = 1000E3;     % exposure time in us
pat.clearAfter      = 1;        % clear pattern after exposure
pat.bitDepth        = 1;        % desired bit depth (1 corresponds to bitdepth of 1)
pat.leds            = 0;        % select which color to use
pat.triggerIn       = 0;        % wait for trigger or cuntinue
pat.darkTime        = 0;        % dark time after exposure in us
pat.triggerOut      = 0;        % use trigger2 as output
pat.patternIdx      = 0;        % image pattern index
pat.bitPosition     = 0;        % bit position in image pattern

% Update pattern definition
D.definePattern(pat);

% Set images to be one
D.numOfImages;

%%%%% Define  Image

% Initialize to zero
I=ones(1080,1920);

BMP=prepBMP(I);

%%%%% Upload Image

% Prepare the DMD
D.initPatternLoad(0,size(BMP,1));

% Do the upload
D.uploadPattern(BMP);

%%%%%

% Start the 
D.patternControl(2);

%D.displayImage(ones(1080,1920));

pause(1);

% Stop any existing sequences
D.patternControl(0);

% Disconnect from DMD
D.delete;
end

