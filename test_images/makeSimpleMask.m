function makeSimpleMask

fname='simple.tif';

X=1:1920;
Y=1:1080;

[xx,yy]=meshgrid(X,Y);
mask=zeros(1080,1920);

% Circle Radius
R=50;

Yc=1080/2;
Xc=1920/2;

pos=[...
    0 -2;
    0 -1
    0 0;
    0 1;
    0 2;
    -2 0;
    -1 0;
    1 0;
    2 0];
pos=pos*200;

foo=@(x,y,xC,yC) ((x-xC).^2+(y-yC).^2)<=R.^2;

for kk=1:size(pos,1)
   mask=mask+foo(xx,yy,Xc+pos(kk,1),Yc+pos(kk,2))   ;
    
end
mask_L=logical(mask);
imwrite(mask_L, 'test.tif','tif','Compression','none');
% imwrite(mask, 'test_rle.tif','tif','Compression','rle');

    
    

end

