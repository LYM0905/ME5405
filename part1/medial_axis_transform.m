
%---------------------------------------------------------------------------------
% task 3
% medial axis transform
% Skeletonisation using the bwskel function
BW3  = img < Otsu;

BW4 = bwskel(BW3);

BW4 = imcomplement(BW4);
%Show image after skeletonisation
figure;
imshow(BW4,'InitialMagnification','fit');
title('medial axis transform')