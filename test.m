refImg='DCC_2021-07-28_14-26-10.mat'; 
load(refImg)
Z=data.Data;

% Crop ROI
ROI=[110 1163 220 810];
% Background ROI
bgROI=[2 50 2 50];

hF=figure(101);
clf
    
% Crop and subtract background
Zbg=Z(bgROI(3):bgROI(4),bgROI(1):bgROI(2));
nBg=sum(sum(Zbg))/(size(Zbg,1)*size(Zbg,2));
Z=Z(ROI(3):ROI(4),ROI(1):ROI(2))-nBg;

% Resizes image to be 1920 x 1080
Z=imresize(Z,[1080 1920]);

% Apply small filter to get rid of pixel noise
Z=imgaussfilt(Z,2);

imagesc(Z)
axis equal tight
