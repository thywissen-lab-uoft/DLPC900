function zOut = makeUoT

data=load('UoT.mat');
data=data.zUoT;
img=zeros(1080,1920);

s=251;
data2=imresize(data,[2*s 2*s]);


yC=1080/2;
xC=1920/2;

img((yC-s):(yC+s-1),(xC-s):(xC+s-1))=data2;


img=double(img);
img=img/max(max(img));

img=round(img);

zOut=img;


end

