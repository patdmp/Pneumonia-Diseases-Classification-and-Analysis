function [dataOut,info] = imresizePD(data, info, imgSize)
%imresizePD resize image data

data = im2gray(data);
data = imresize(data, imgSize);

%data = im2double(data);
dataOut = {data, info.Label};
end