
%%
%Iterative method of threshold division
figure();
 
BGT = Basic_Global_Threshold_func(img);
imshow(img > BGT,'InitialMagnification','fit');   % Basic global threshold image
title(['Iterative Thresholding at: ', num2str(BGT)]);
%%
%Adaptive threshold segmentation
I = original_img;
T=zeros(7);
for i=0:7
    for j=0:7
        img = imcrop(I,[1+i*4 1+j*4 7 7]); 
        t=255*graythresh(img);
        T(j+1,i+1)=t;
    end
end
T=uint8(T);
T1=imresize(T,[256 256],'bilinear');%bilinear interpolation

for i=1:64
    for j=1:64
        if (I(i,j)<T1(i,j))
            BW(i,j)=0;
        else
            BW(i,j)=1;
        end
    end
end

imshow(BW,);
title('Adaptive threshold segmentation')

