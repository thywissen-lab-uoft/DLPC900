function D=DMD_gui(D)
% Add all subdirectories for this m file
curpath = fileparts(mfilename('fullpath'));
dmdDir=fullfile(curpath,'DMDConnect');
addpath(dmdDir);addpath(genpath(dmdDir))   

%% Initialize DMD

if nargin~=1
    % Connect and initialize DMD
    D=DMD('debug',0);
end

% Display firmware version
D.fwVersion

% Stop any existing sequences
disp('Stopping any running sequences');
D.patternControl(0);

% Set display mode to 3 (Pattern-on-the-fly);
disp('Setting to Pattern-on-the-fly mode');
D.setMode(3);

%% Define  Image

% Initialize to ones
I=ones(1080,1920);

% Initialize to zeros
% I=zeros(1080,1920);

% Convert image to BMP
fprintf('Converting image matrix to compressed BMP ... ');
BMP=prepBMP(I);
disp('done');

%% Upload Image

% Prepare the DMD
D.initPatternLoad(0,size(BMP,1));

% Do the upload
fprintf('Uploading BMP ... ');
D.uploadPattern(BMP);
disp('done');

%% Define Pattern Settings
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
fprintf('Sending pattern settings ... ');
D.definePattern(pat);
disp('done');

% Set images to be one, no
numImages=1;
numRepititions=0;
D.numOfImages(numImages,numRepititions);

%% Start Sequence

% Start the sequence
D.patternControl(2);

% Disconnect from DMD
D.delete;
end

