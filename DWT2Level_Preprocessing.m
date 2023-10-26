%LOAD DATA
current_path = pwd;
path2data = fullfile(pwd, 'Data_Covid', 'train');
imds = imageDatastore(path2data, 'IncludeSubfolders',true,'LabelSource','foldernames');

std_size = [256 256];

%2-LEVEL DWT
for i = 1:4
    if i == 1
        class = 'covid';
    elseif i == 2
        class = 'normal';
    elseif i == 3
        class = 'pneumonia_bacterial';
    else
        class = 'pneumonia_viral';
    end
    
    imds = imageDatastore(fullfile(path2data, location));
    numImages = length(imds.Files);
    
    parfor j = 1:numImages
        [X, fileinfo] = readimage(imds, j);
        [filepath, name, ext] = fileparts()

        is3d = length(size(X)) == 3;
        for color = 0:3 
            if is3d
                if color == 0
                    % 0 is gray
                    newX = im2gray(X);
                else
                    % 1 to 3 are red, green, blue respectively
                    newX = X(:,:,color);
                end
            else
                newX = X;
            end

            %2-Level DWT
            [c,s] = wavedec2(newX,2,'haar');
            A2 = appcoef2(c,s,'haar',2);

            splitImages{j}{color+1}.A2 = A2;
            
            %break if image is 2d
            if not(is3d)
                break
            end

        end %end color loop
    end %end parforloop
    
    %SAVE IMAGE
    for k = 1:length(splitImages)
        for channel = 1:length(splitImages{k})
            
        end
    end
    path2preprocessedImg = fullfile(current_path, ...
            'Data_Covid', 'train_dwt', class, file_name);
        save(path2preprocessedImg, "A2", "H2", "V2", "D2");
end