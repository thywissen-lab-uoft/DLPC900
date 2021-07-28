
%% Dependencies
curpath = fileparts(mfilename('fullpath'));
dmdDir=fullfile(curpath,'DMDConnect');
addpath(dmdDir);addpath(genpath(dmdDir))   
%% Define  Image

% Initialize to ones
I=ones(1080,1920);

% Initialize to zeros
%I=zeros(1080,1920);

R=50;
[xx,yy]=meshgrid(1:1920,1:1080);
inds=sqrt((xx-1920/2).^2+(yy-1080/2).^2)<R;
I(inds)=0;

% inds=sqrt((xx-1920/2).^2+(yy-1080/2-200).^2)<R;
% I(inds)=0;
% 
% inds=sqrt((xx-1920/2).^2+(yy-(1080/2+200)).^2)<R;
% I(inds)=0;

% inds=sqrt((xx-(1920/2-200)).^2+(yy-1080/2).^2)<R;
% I(inds)=0;
% 
% inds=sqrt((xx-(1920/2+200)).^2+(yy-1080/2).^2)<R;
% I(inds)=0;


% Deconvole with gaussian profile of beam
doGrayScale=1;
if doGrayScale
    I=I*.2;

    str='DCC_2021-07-28_14-26-10.mat'; 
    Zbeam = getGaussEnvelope(str);
    
    % Divide by envelope
    I1=I./Zbeam;
    I1=real(I1);
    
    % Threshold the data
    I1(I1>1)=1;
    
    % Account for non-linearity
    I2=I1.^(1/2);    
    
    % Dither
    I3a=dither(I2);
    I3b=stucki(I2);
    
    % Plot results
    figure(101)
    subplot(321)
    imagesc(I)
    caxis([0 1]);
    colorbar;
    axis equal tight
    title('target');

    subplot(322)
    imagesc(Zbeam)
    caxis([0 1]);
    colorbar;
    axis equal tight
    title('gauss');

    subplot(323)
    imagesc(I1);
    caxis([0 1]);
    colorbar;
    axis equal tight
    title('grayscale');
    
    subplot(324)
    imagesc(I2);
    caxis([0 1]);
    colorbar;
    axis equal tight
    title('grayscale + non linearity');  
    
    subplot(325)
    imagesc(I3a);
    caxis([0 1]);
    colorbar;
    axis equal tight
    title('dither 1');  
    
    subplot(326)
    imagesc(I3b);
    caxis([0 1]);
    colorbar;
    axis equal tight
    title('dither 2');  
    
    I=I3b;
    
end


% Convert to zeros and ones if not already done
I=logical(I);
I=~I;

% Flip vertically due to imaging system
Iflip=flip(I,1);

hFI=figure(103);
imagesc(I);

% Convert image to BMP
fprintf('Converting image matrix to compressed BMP ... ');
BMP=prepBMP(Iflip);
disp('done');



%% Define Pattern

%Define Pattern Settings
pat=struct;
pat.idx             = 0;        % pattern index
pat.exposureTime    = 1000E3;   % exposure time in us
pat.clearAfter      = 1;        % clear pattern after exposure
pat.bitDepth        = 1;        % desired bit depth (1 corresponds to bitdepth of 1)
pat.leds            = 0;        % select which color to use
pat.triggerIn       = 1;        % wait for trigger or cuntinue
pat.darkTime        = 0;        % dark time after exposure in us
pat.triggerOut      = 0;        % use trigger2 as output
pat.patternIdx      = 0;        % image pattern index
pat.bitPosition     = 0;        % bit position in image pattern

% Set images to be one, no
numImages=1;
numRepititions=0; % 0 means forever?



%% Initialize DMD

if ~exist('dmd_handle','var') || ~isvalid(dmd_handle)
    % Connect and initialize DMD
    disp('connecting to dmd');
    dmd_handle=DMD('debug',0);
end


if dmd_handle.sleeping
    dmd_handle.wakeup;
end


if dmd_handle.isidle
    dmd_handle.active;
end

% Display firmware version
dmd_handle.fwVersion

% Stop any existing sequences
disp('Stopping any running sequences');
dmd_handle.patternControl(0);

% Set display mode to 3 (Pattern-on-the-fly);
disp('Setting to Pattern-on-the-fly mode');
dmd_handle.setMode(3);

%% Upload Image

% Prepare the DMD
dmd_handle.initPatternLoad(0,size(BMP,1));

% Do the upload
fprintf('Uploading BMP ... ');
dmd_handle.uploadPattern(BMP);
disp('done');

%% Send Pattern settings

% Update pattern definition
fprintf('Sending pattern settings ... ');
dmd_handle.definePattern(pat);
disp('done');


dmd_handle.numOfImages(numImages,numRepititions);

%% Start Sequence

% Start the sequence
dmd_handle.patternControl(2); % 0 : stop, 1: pause, 2: start

% Disconnect from DMD
%dmd_handle.delete;
%delete(dmd_handle);
%clear('dmd_handle');