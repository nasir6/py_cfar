
%% Close all windows, clear all variables, clear output screen
close all;
clear all;
clc;

%% Compile Average Image C++ file 
mex avgRegionImage.cpp

%% Load image (use png for simplicity's sake)
image = imread('vessels.png');

% if image colour, make it grayscale
% make sure image is uint8 (pixel values 0 - 255)
[r,c,p] = size(image);
if p > 1,
    image = uint8(rgb2gray(image));
end
clear inprof outprof C

%% Set up parameters for CFAR processing
% Create mask (useful to speed up processing in images with only specific areas
% that should be processed)
% for the moment just make a mask that works for all pixels
paramStruct.mask = uint8(ones(r,c).*255);

% Set guard/background/padding (edges)
paramStruct.guardSize = 5;
paramStruct.backgroundSize = 7;
paramStruct.padSize = 2;
paramStruct.threshold = 2.5;

if paramStruct.guardSize > paramStruct.backgroundSize
disp('Please ensure that the guard size is smaller than the background size')
end
% Apply average (mean) image processing. Each pixel's mean background value
% u_b is calculated using the specified guard/background window sizes. The
% CFAR algorithm only requires the mean background calculation only once
% because the threshold value $T$ is the dependant variable. 
% 
[avgImage] = avgRegionImage(image, paramStruct.mask, paramStruct);
avgImage = scaledata(avgImage, 0, 1);

% Create test image by multiplying the mean image by the selected threshold
% value.
testImage = double(avgImage.*paramStruct.threshold);

% Compare each pixel in the input image to the test image to determine
% which pixels are bright pixels (according to CFAR algorithm). Scale data
% so both are double images between 0, 1
imageD = double(scaledata(image,0,1));
outputImage = imageD >= testImage;

%% Show input and CFAR images linked so zooming on one does the same on the other
figure('renderer', 'zbuffer')
a = subplot(2,1,1);
imshow(image);
b = subplot(2,1,2);
imshow(outputImage);
linkaxes([a,b],'xy');