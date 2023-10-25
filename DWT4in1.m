function [dataOut,info] = DWT4in1(data,info,imgSize)
%DTCWT4in1 : resize data to imgSize, dwt2 the image, 
%            then concat approximaate and 3 detailed 
%            images into 1

    img = data;
    resizedImg = imresize(im2gray(img), imgSize);

    level = 6;
    [c,s] = wavedec2(resizedImg, level, 'haar');
    [H6,V6,D6] = detcoef2('all',c,s,6);
    A6 = appcoef2(c,s,'haar',6);

    imgOut = [A6 H6; V6 D6];
    dataOut = {imgOut, info.Label};
end