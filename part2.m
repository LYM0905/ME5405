clc;

% Problem 1:
% Show the original image (enlarged)
originalImage = imread('hello_world.jpg');
figure(1);
imshow(originalImage);
hold on;

% Problem 2:
% Create an image which is a sub-image of the original image comprising the middle line – HELLO, WORLD.
[height, width, ~] = size(originalImage);

% Calculate the height of each part
thirdHeight = height / 3;

% [X Y WIDTH HEIGHT]
cropRect = [1, thirdHeight, width, thirdHeight];

% Crop image
textImage = imcrop(originalImage, cropRect);

figure(2);
% Display the cropped image
imshow(textImage);
hold on;

% Problem 3:
% Create a binary image from Step 2 using thresholding.
if size(textImage, 3) == 3 % Check if the image is RGB
     grayImage = rgb2gray(textImage);
else
     grayImage = textImage; % If already grayscale, no need to convert
end
% Apply Otsu's thresholding
thresholdValue = graythresh(grayImage); % use Otsu method
binaryImage = imbinarize(grayImage, thresholdValue); % Convert to binary image

% Display the binary image
figure (3);
imshow(binaryImage); % Display binary image
hold on;

% Problem 4:
% Determine one-pixel image
% Refinement operation
thinImage = bwmorph(binaryImage, 'thin', Inf);

% Or, skeletonization operation
% skeletonImage = bwmorph(binaryImage, 'skel', Inf);

% Display the refined image
figure(4)
imshow(thinImage);
hold on;

% Or, display the skeletonized image
% imshow(skeletonImage);

% Problem 5
% Find the outlines of the characters
outlines = bwperim(binaryImage);

% Display the outlines
figure (5);
imshow(outlines);
hold on;

% Problem 6:
% Segment and label characters in the binary image

% Find connected components (individual characters) in the binary image
cc = bwconncomp(binaryImage);
stats = regionprops(cc, 'BoundingBox', 'Centroid');

% Create a new figure (Figure 6) for displaying the segmented characters
figure(6);
imshow(binaryImage); % Display the binary image

hold on; % Enable overlaying on the image

% Initialize a counter for numeric labels
labelCounter = 1;

for i = 1:length(stats)
    % Get the bounding box and centroid of each connected component
    bbox = stats(i).BoundingBox;
    centroid = stats(i).Centroid;
    
    % Calculate the width and height of the bounding box
    width = bbox(3);
    height = bbox(4);
    
    % Check if the region has a width-to-height ratio greater than 1
    if width / height > 1
        % Split the region into two equal parts horizontally
        halfWidth = width / 2;
        
        % Create two new bounding boxes for the split regions
        bbox1 = [bbox(1), bbox(2), halfWidth, height];
        bbox2 = [bbox(1) + halfWidth, bbox(2), halfWidth, height];
        
        % Draw red boxes around the split regions and label them with numeric labels
        rectangle('Position', bbox1, 'EdgeColor', 'r', 'LineWidth', 2);
        text(centroid(1) - 5, centroid(2), num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
        
        % Increment the numeric label counter
        labelCounter = labelCounter + 1;
        
        rectangle('Position', bbox2, 'EdgeColor', 'r', 'LineWidth', 2);
        text(centroid(1) + halfWidth - 5, centroid(2), num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
        
        % Increment the numeric label counter again for the second part
        labelCounter = labelCounter + 1;
    else
        % Draw a red box around the non-split region and label it with a numeric label
        rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
        text(centroid(1) - 5, centroid(2), num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
        
        % Increment the numeric label counter
        labelCounter = labelCounter + 1;
    end
end

hold off; % Disable overlaying on the image




% Problem 7
% Using the training dataset provided on LumiNUS (p_dataset_26.zip), train the
% (conventional) unsupervised classification method of your choice (i.e., self-ordered
% maps (SOM), k-nearest neighbors (kNN), or support vector machine (SVM)) to
% recognize the different characters


% % KNN

% pixel value method (0.95107)
% Main script part
% Step 1: Load the dataset
% Specify the folder path containing the H letter picture
folderPathH = 'D:\Graduate\5405\p_dataset_26\p_dataset_26\SampleH';

% Specify the folder path containing the E letter picture
folderPathE = 'D:\Graduate\5405\p_dataset_26\p_dataset_26\SampleE';

% Specify the folder path containing the D letter picture
folderPathD = 'D:\Graduate\5405\p_dataset_26\p_dataset_26\SampleD';

% Specify the folder path containing the L letter picture
folderPathL = 'D:\Graduate\5405\p_dataset_26\p_dataset_26\SampleL';

% Specify the folder path containing the O letter picture
folderPathO = 'D:\Graduate\5405\p_dataset_26\p_dataset_26\SampleO';

% Specify the folder path containing the R letter picture
folderPathR = 'D:\Graduate\5405\p_dataset_26\p_dataset_26\SampleR';

% Specify the folder path containing the W letter picture
folderPathW = 'D:\Graduate\5405\p_dataset_26\p_dataset_26\SampleW';

% Use imageDatastore to load the dataset, including image data in subfolders, and assign new labels
imdsH = imageDatastore(folderPathH, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsE = imageDatastore(folderPathE, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsD = imageDatastore(folderPathD, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsL = imageDatastore(folderPathL, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsO = imageDatastore(folderPathO, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsR = imageDatastore(folderPathR, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsW = imageDatastore(folderPathW, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Merge all data into one data set
imdsCombined = imageDatastore(cat(1, imdsH.Files, imdsE.Files, imdsD.Files, imdsL.Files, imdsO.Files, imdsR.Files, imdsW.Files), 'LabelSource', 'foldernames');


% Shuffle the data set
imdsCombined = shuffle(imdsCombined);


% Display some information from the data set
disp(numel(imdsCombined.Files)); % Display the total number of samples
disp(unique(imdsCombined.Labels)); % Display new category labels


% Step 2: Split the dataset
[trainingSet, testSet] = splitEachLabel(imdsCombined, 0.75, 'randomize');

% Step 3: Feature extraction
trainingFeatures = extractFeaturesFromImages(trainingSet, @extractFeaturesUsingRawPixels);
trainingLabels = trainingSet.Labels;
testFeatures = extractFeaturesFromImages(testSet, @extractFeaturesUsingRawPixels);

% Step 4: Train kNN classifier
k = 3; % k value can be adjusted as needed,in this example:values = 1:2:20;
knnClassifier = fitcknn(trainingFeatures, trainingLabels, 'NumNeighbors', k);

% Step 5: Evaluate the classifier
predictedLabels = predict(knnClassifier, testFeatures);
accuracy = sum(predictedLabels == testSet.Labels) / numel(testSet.Labels);
disp(['Accuracy: ', num2str(accuracy)]);



% Create a cell array that stores the extracted character images
extractedCharacters = cell(0);

% Initialize a counter for numeric labels
labelCounter = 1;
binaryImageInverted = ~binaryImage;

for i = 1:length(stats)
    % Get the bounding box of the current character
    bbox = stats(i).BoundingBox;

    % Rounds the coordinates of the bounding box to the nearest integer value
    x = round(bbox(1));
    y = round(bbox(2));
    width = round(bbox(3));
    height = round(bbox(4));

    % Check if the width of the bounding box is greater than the height
    if width / height > 1
        % The width is greater than the height, and the divided area is two characters

        % Compute new two bounding boxes bbox1 and bbox2
        halfWidth = width / 2;
        bbox1 = [x, y, halfWidth, height];
        bbox2 = [x + halfWidth, y, halfWidth, height];

        % Use imcrop function to extract character parts from binary images and store them separately
        charImage1 = imcrop(binaryImageInverted, bbox1);
        charImage2 = imcrop(binaryImageInverted, bbox2);

        % Add extracted character image to cell array and assign new numeric label
        extractedCharacters{end+1} = charImage1;
        extractedCharacters{end+1} = charImage2;

        % Draw red bounding boxes and numerical labels to identify the two characters
        hold on;
        rectangle('Position', bbox1, 'EdgeColor', 'r', 'LineWidth', 2);
        text(bbox1(1) + 5, bbox1(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
        rectangle('Position', bbox2, 'EdgeColor', 'r', 'LineWidth', 2);
        text(bbox2(1) + 5, bbox2(2) - 5, num2str(labelCounter + 1), 'Color', 'r', 'FontSize', 12);
        hold off;

        % Increment labelCounter to assign a new numeric label to the next character
        labelCounter = labelCounter + 2;
    else
        % Width not greater than height, processing single characters

        % Use imcrop function to extract character parts from binary images
        charImage = imcrop(binaryImageInverted, [x, y, width, height]);

        % Add extracted character image to cell array
        extractedCharacters{end+1} = charImage;

        % Draw red bounding boxes and numerical labels to identify individual characters
        hold on;
        rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
        text(bbox(1) + 5, bbox(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
        hold off;

        % Increment labelCounter to assign a new numeric label to the next character
        labelCounter = labelCounter + 1;
    end
end

% Display or save extracted character image (optional)
for i = 1:length(extractedCharacters)
    figure;
    imshow(extractedCharacters{i});
    title(['Extracted Character ', num2str(i)]);
    % Optional: Save the extracted character image to a file
    % imwrite(extractedCharacters{i}, ['char_', num2str(i), '.png']);
end




% Now, the extractedCharacters array contains an image of each individual character
% Create a cell array to store character classification results
characterPredictions = cell(size(extractedCharacters));


% Loop through each extracted character image
for i = 1:length(extractedCharacters)
    % Extract current character image
    charImage = extractedCharacters{i};


    % Resize character images to match training data
    charImage = imresize(charImage, [32 32]);

    % Get features of character images
    charFeatures = extractFeaturesUsingRawPixels(charImage);

    % Classify characters using KNN classifier
    predictedLabel = predict(knnClassifier, charFeatures);

    % Store classification results
    characterPredictions{i} = predictedLabel;

    % Print the classification results for each character
    fprintf('Classification results for character %d：%s\n', i, predictedLabel);
end





% local function definition
function features = extractFeaturesFromImages(imageSet, featureExtractor)
    numImages = numel(imageSet.Files);
    features = [];
    for i = 1:numImages
        img = readimage(imageSet, i);
        imgFeatures = featureExtractor(img);
        features = [features; imgFeatures];
    end
end

function features = extractFeaturesUsingRawPixels(image)
    % If the image is not binary, resize and convert to grayscale image
    if ~islogical(image)
        resizedImage = imresize(image, [32 32]);
        if size(resizedImage, 3) == 3
            grayImage = rgb2gray(resizedImage);
        else
            grayImage = resizedImage;
        end
        features = double(grayImage(:)');
    else
        % If the image is already binary, resize it directly
        resizedImage = imresize(image, [32 32]);
        features = double(resizedImage(:)');
    end
end




% % Histogram（0.50731）
% % Main script part
% % Step 1: Load the dataset
% dataFolderPath = 'D:\Graduate\5405\p_dataset_26\p_dataset_26';
% imds = imageDatastore(dataFolderPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%
% % Step 2: Split the dataset
% [trainingSet, testSet] = splitEachLabel(imds, 0.75, 'randomize');
%
% % Step 3: Feature extraction (using grayscale histogram method)
% trainingFeatures = extractFeaturesFromImages(trainingSet, @extractFeaturesUsingHistogram);
% trainingLabels = trainingSet.Labels;
% testFeatures = extractFeaturesFromImages(testSet, @extractFeaturesUsingHistogram);
%
% % Step 4: Train kNN classifier
% k = 100; % k value can be adjusted as needed
% knnClassifier = fitcknn(trainingFeatures, trainingLabels, 'NumNeighbors', k);
%
% % Step 5: Evaluate the classifier
% predictedLabels = predict(knnClassifier, testFeatures);
% accuracy = sum(predictedLabels == testSet.Labels) / numel(testSet.Labels);
% disp(['Accuracy: ', num2str(accuracy)]);
%
%
% % Create a cell array that stores the extracted character images
% extractedCharacters = cell(0);
% binaryImageInverted = ~binaryImage;
% % Initialize a counter for numeric labels
% labelCounter = 1;
% 
% for i = 1:length(stats)
%     % Get the bounding box of the current character
%     bbox = stats(i).BoundingBox;
% 
%     % Round the coordinates of the bounding box to the nearest integer value
%     x = round(bbox(1));
%     y = round(bbox(2));
%     width = round(bbox(3));
%     height = round(bbox(4));
% 
%     % Check if the width of the bounding box is greater than the height
%     if width / height > 1
%         % The width is greater than the height, and the divided area is two characters
% 
%         % Compute new two bounding boxes bbox1 and bbox2
%         halfWidth = width / 2;
%         bbox1 = [x, y, halfWidth, height];
%         bbox2 = [x + halfWidth, y, halfWidth, height];
% 
%         % Use imcrop function to extract character parts from binary images and store them separately
%         charImage1 = imcrop(binaryImageInverted , bbox1);
%         charImage2 = imcrop(binaryImageInverted , bbox2);
% 
%         % Add extracted character image to cell array and assign new numeric label
%         extractedCharacters{end+1} = charImage1;
%         extractedCharacters{end+1} = charImage2;
% 
%         % Draw red bounding boxes and numerical labels to identify the two characters
%         hold on;
%         rectangle('Position', bbox1, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox1(1) + 5, bbox1(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         rectangle('Position', bbox2, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox2(1) + 5, bbox2(2) - 5, num2str(labelCounter + 1), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter to assign a new numeric label to the next character
%         labelCounter = labelCounter + 2;
%     else
%         % Width not greater than height, processing single characters
% 
%         % Use imcrop function to extract character parts from binary images
%         charImage = imcrop(binaryImageInverted , [x, y, width, height]);
% 
%         % Add extracted character image to cell array
%         extractedCharacters{end+1} = charImage;
% 
%         % Draw red bounding boxes and numerical labels to identify individual characters
%         hold on;
%         rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox(1) + 5, bbox(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter to assign a new numeric label to the next character
%         labelCounter = labelCounter + 1;
%     end
% end
% 
% % Display or save extracted character image (optional)
% for i = 1:length(extractedCharacters)
%     figure;
%     imshow(extractedCharacters{i});
%     title(['Extracted Character ', num2str(i)]);
%     % Optional: Save the extracted character image to a file
%     % imwrite(extractedCharacters{i}, ['char_', num2str(i), '.png']);
% end

% % Connected component analysis
% cc = bwconncomp(binaryImage);
% stats = regionprops(cc, 'Image');
% 
% %Create a cell array that stores character images
% extractedCharacters = cell(cc.NumObjects, 1);
% 
% % Iterate over each connected component (character)
% for i = 1:cc.NumObjects
%     % Extract the image of the current character
%     charImage = stats(i).Image;
% 
%     % Store character image in cell array
%     extractedCharacters{i} = charImage;
% end
% 
% % Display or save the extracted character image (optional）
% for i = 1:length(extractedCharacters)
%     figure;
%     imshow(extractedCharacters{i});
%     title(['Extracted Character ', num2str(i)]);
%     % Optional: Save the extracted character image to a file
%     % imwrite(extractedCharacters{i}, ['char_', num2str(i), '.png']);
% end



% Now, the extractedCharacters array contains an image of each individual character
% Create a cell array to store character classification results
% characterPredictions = cell(size(extractedCharacters));

% % Iterate through each extracted character image
% for i = 1:length(extractedCharacters)
%     % Extract current character image
%     charImage = extractedCharacters{i};
% 
%     % Resize character images to match training data
%     charImage = imresize(charImage, [32 32]);
% 
%     % Extract features of character images
%     charFeatures = extractFeaturesUsingHistogram(charImage);
% 
%     % Classify characters using KNN classifier
%     predictedLabel = predict(knnClassifier, charFeatures);
% 
%     % Store classification results
%     characterPredictions{i} = predictedLabel;
% 
%     % Print the classification results for each character
%     fprintf('Classification results for character %d：%s\n', i, predictedLabel);
% end
% 
% % local function definition
% function features = extractFeaturesUsingHistogram(image)
%     % Convert image to grayscale (if it is in color)
%     if size(image, 3) == 3
%         grayImage = rgb2gray(image);
%     else
%         grayImage = image;
%     end
% 
%       % Compute grayscale histogram as feature
%     % Assume 2 bins are used since the logical image only has two values
%     features = imhist(grayImage, 2)'; 
%     features = features / sum(features); % introduce normalized histogram
% end
% function features = extractFeaturesFromImages(imageSet, featureExtractor)
%     numImages = numel(imageSet.Files);
%     features = []; % Initialize an empty matrix to store features
% 
%     for i = 1:numImages
%         img = readimage(imageSet, i); % Read each image from the data store
%         imgFeatures = featureExtractor(img); % Apply feature extraction function
%         features = [features; imgFeatures]; % Add features to matrix
%     end
% end



% % Edge detection method (0.14286)
% % Main script part
% % Step 1: Load the dataset
% dataFolderPath = 'D:\Graduate\5405\p_dataset_26\p_dataset_26';
% imds = imageDatastore(dataFolderPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%
% % Step 2: Split the dataset
% [trainingSet, testSet] = splitEachLabel(imds, 0.75, 'randomize');
%
% % Step 3: Feature extraction (using edge detection method)
% trainingFeatures = extractFeaturesFromImages(trainingSet, @extractFeaturesUsingEdgeDetection);
% trainingLabels = trainingSet.Labels;
%
% % Calculate PCA coefficient matrix
% [coeff, ~, ~, ~, ~] = pca(trainingFeatures);
% pcaCoeff = coeff(:, 1:256); % Save the first 256 principal components
%
% % Training features after dimensionality reduction
% reducedTrainingFeatures = trainingFeatures * pcaCoeff;
%
% % Step 4: Train kNN classifier
% k = 10; % k value can be adjusted as needed
% knnClassifier = fitcknn(reducedTrainingFeatures, trainingLabels, 'NumNeighbors', k);
%
% % Extract and dimensionally reduce test features
% testFeatures = extractFeaturesFromImages(testSet, @extractFeaturesUsingEdgeDetection);
% reducedTestFeatures = testFeatures * pcaCoeff;
%
% % Step 5: Evaluate the classifier
% predictedLabels = predict(knnClassifier, reducedTestFeatures);
% accuracy = sum(predictedLabels == testSet.Labels) / numel(testSet.Labels);
% disp(['Accuracy: ', num2str(accuracy)]);
%
% % Create a cell array that stores the extracted character images
% extractedCharacters = cell(0);
% binaryImageInverted = ~binaryImage;
% % Initialize a counter for numeric labels
% labelCounter = 1;
% 
% for-loop to process each character region in the image
% for i = 1:length(stats)
%     % Get the bounding box of the current character
%     bbox = stats(i).BoundingBox;
% 
%     % Round the coordinates of the bounding box to the nearest integers
%     x = round(bbox(1));
%     y = round(bbox(2));
%     width = round(bbox(3));
%     height = round(bbox(4));
% 
%     % Check if the width of the bounding box is greater than the height
%     if width / height > 1
%         % If width is greater than height, split the region into two characters
% 
%         % Calculate new bounding boxes bbox1 and bbox2
%         halfWidth = width / 2;
%         bbox1 = [x, y, halfWidth, height];
%         bbox2 = [x + halfWidth, y, halfWidth, height];
% 
%         % Use imcrop to extract character parts from the binary image and store them separately
%         charImage1 = imcrop(binaryImageInverted , bbox1);
%         charImage2 = imcrop(binaryImageInverted , bbox2);
% 
%         % Add the extracted character images to a cell array and assign new numeric labels
%         extractedCharacters{end+1} = charImage1;
%         extractedCharacters{end+1} = charImage2;
% 
%         % Draw red bounding boxes and numeric labels to identify the two characters
%         hold on;
%         rectangle('Position', bbox1, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox1(1) + 5, bbox1(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         rectangle('Position', bbox2, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox2(1) + 5, bbox2(2) - 5, num2str(labelCounter + 1), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter for assigning new numeric labels to the next character
%         labelCounter = labelCounter + 2;
%     else
%         % If width is not greater than height, process a single character
% 
%         % Use imcrop to extract the character part from the binary image
%         charImage = imcrop(binaryImageInverted , [x, y, width, height]);
% 
%         % Add the extracted character image to a cell array
%         extractedCharacters{end+1} = charImage;
% 
%         % Draw a red bounding box and numeric label to identify the single character
%         hold on;
%         rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox(1) + 5, bbox(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter for assigning new numeric labels to the next character
%         labelCounter = labelCounter + 1;
%     end
% end
% 
% % Display or save the extracted character images (optional)
% for i = 1:length(extractedCharacters)
%     figure;
%     imshow(extractedCharacters{i});
%     title(['Extracted Character ', num2str(i)]);
%     % Optional: Save the extracted character images to files
%     % imwrite(extractedCharacters{i}, ['char_', num2str(i), '.png']);
% end
% 
% % Connected component analysis
% cc = bwconncomp(binaryImage);
% stats = regionprops(cc, 'Image');
% 
% % Create a cell array to store character images
% extractedCharacters = cell(cc.NumObjects, 1);
% 
% % Iterate over each connected component (character)
% for i = 1:cc.NumObjects
%     % Extract the image of the current character
%     charImage = stats(i).Image;
% 
%     % Store the character image in the cell array
%     extractedCharacters{i} = charImage;
% end
% 
% % Display or save the extracted character images (optional)
% for i = 1:length(extractedCharacters)
%     figure;
%     imshow(extractedCharacters{i});
%     title(['Extracted Character ', num2str(i)]);
%     % Optional: Save the extracted character images to files
%     % imwrite(extractedCharacters{i}, ['char_', num2str(i), '.png']);
% end
% 
% % Now, the extractedCharacters array contains images of each individual character
% % Create a cell array to store character classification results
% characterPredictions = cell(size(extractedCharacters));
% 
% % Iterate over each extracted character image
% for i = 1:length(extractedCharacters)
%     % Extract the current character image
%     charImage = extractedCharacters{i};
% 
%     % Resize the character image to match the training data
%     charImage = imresize(charImage, [128 128]);
% 
%     % Extract features from the character image
%     charFeatures = extractFeaturesUsingEdgeDetection(charImage);
% 
%     % Classify the character using a KNN classifier
%     predictedLabel = predict(knnClassifier, charFeatures);
% 
%     % Store the classification result
%     characterPredictions{i} = predictedLabel;
% 
%     % Print the classification result for each character
%     fprintf('Classification result for character %d: %s\n', i, predictedLabel);
% end
% 
% % Helper function to extract features using edge detection
% function features = extractFeaturesUsingEdgeDetection(image)
%     % Convert the image to grayscale if it's in color
%     if size(image, 3) == 3
%         grayImage = rgb2gray(image);
%     else
%         grayImage = image;
%     end
% 
%     % Resize the image for consistency
%     resizedImage = imresize(grayImage, [128, 128]);
% 
%     % Apply edge detection
%     edgeImage = edge(resizedImage, 'Canny');
% 
%     % Convert the edge image to a one-dimensional feature vector
%     edgeFeatures = edgeImage(:)';
% 
%     % Choose the number of bins for the grayscale histogram based on the image type
%     if islogical(resizedImage)
%         numBins = 2; % Use 2 bins for logical images
%     else
%         numBins = 256; % Use 256 bins for non-logical images
%     end
% 
%     % Calculate the grayscale histogram as a feature
%     histogramFeatures = imhist(resizedImage, numBins)';
%     histogramFeatures = histogramFeatures / sum(histogramFeatures); % Normalize the histogram
% 
%     % Combine edge features and histogram features
%     combinedFeatures = [edgeFeatures, histogramFeatures];
% 
%     % Return the features
%     features = combinedFeatures(1:256); % Ensure the feature vector length is 256
% end
% 
% % Helper function to extract features from a set of images using a feature extractor
% function features = extractFeaturesFromImages(imageSet, featureExtractor)
%     numImages = numel(imageSet.Files);
%     features = []; % Initialize an empty matrix to store features
% 
%     for i = 1:numImages
%         img = readimage(imageSet, i); % Read each image from the data store
%         imgFeatures = featureExtractor(img); % Apply the feature extraction function
%         features = [features; imgFeatures]; % Add the features to the matrix
%     end
% end




% % Gabor Features ()
% % Main Script Section
% % Step 1: Load the dataset
% dataFolderPath = 'D:\Graduate\5405\p_dataset_26\p_dataset_26';
% imds = imageDatastore(dataFolderPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% 
% % Step 2: Split the dataset
% [trainingSet, testSet] = splitEachLabel(imds, 0.75, 'randomize');
% 
% % Step 3: Feature extraction (using Gabor feature extraction)
% trainingFeatures = extractFeaturesFromImages(trainingSet, @extractFeaturesUsingGabor);
% trainingLabels = trainingSet.Labels;
% testFeatures = extractFeaturesFromImages(testSet, @extractFeaturesUsingGabor);
% 
% % Step 4: Train a kNN classifier
% k = 3; % k value can be adjusted as needed
% knnClassifier = fitcknn(trainingFeatures, trainingLabels, 'NumNeighbors', k);
% 
% % Step 5: Evaluate the classifier
% predictedLabels = predict(knnClassifier, testFeatures);
% accuracy = sum(predictedLabels == testSet.Labels) / numel(testSet.Labels);
% disp(['Accuracy: ', num2str(accuracy)]);
% 
% % Create a cell array to store extracted character images
% extractedCharacters = cell(0);
% binaryImageInverted = ~binaryImage;
% % Initialize a counter for numeric labels
% labelCounter = 1;
% 
% for i = 1:length(stats)
%     % Get the bounding box of the current character
%     bbox = stats(i).BoundingBox;
% 
%     % Round the coordinates of the bounding box to the nearest integers
%     x = round(bbox(1));
%     y = round(bbox(2));
%     width = round(bbox(3));
%     height = round(bbox(4));
% 
%     % Check if the width of the bounding box is greater than the height
%     if width / height > 1
%         % If width is greater than height, split the region into two characters
% 
%         % Calculate new bounding boxes bbox1 and bbox2
%         halfWidth = width / 2;
%         bbox1 = [x, y, halfWidth, height];
%         bbox2 = [x + halfWidth, y, halfWidth, height];
% 
%         % Use imcrop to extract character parts from the binary image and store them separately
%         charImage1 = imcrop(binaryImageInverted , bbox1);
%         charImage2 = imcrop(binaryImageInverted , bbox2);
% 
%         % Add the extracted character images to a cell array and assign new numeric labels
%         extractedCharacters{end+1} = charImage1;
%         extractedCharacters{end+1} = charImage2;
% 
%         % Draw red bounding boxes and numeric labels to identify the two characters
%         hold on;
%         rectangle('Position', bbox1, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox1(1) + 5, bbox1(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         rectangle('Position', bbox2, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox2(1) + 5, bbox2(2) - 5, num2str(labelCounter + 1), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter to assign new numeric labels to the next character
%         labelCounter = labelCounter + 2;
%     else
%         % If width is not greater than height, process a single character
% 
%         % Use imcrop to extract the character part from the binary image
%         charImage = imcrop(binaryImageInverted , [x, y, width, height]);
% 
%         % Add the extracted character image to a cell array
%         extractedCharacters{end+1} = charImage;
% 
%         % Draw a red bounding box and numeric label to identify the single character
%         hold on;
%         rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox(1) + 5, bbox(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter to assign new numeric labels to the next character
%         labelCounter = labelCounter + 1;
%     end
% end
% 
% % Display or save the extracted character images (optional)
% for i = 1:length(extractedCharacters)
%     figure;
%     imshow(extractedCharacters{i});
%     title(['Extracted Character ', num2str(i)]);
%     % Optional: Save the extracted character images to files
%     % imwrite(extractedCharacters{i}, ['char_', num2str(i), '.png']);
% end
% 
% % Connected component analysis
% cc = bwconncomp(binaryImage);
% stats = regionprops(cc, 'Image');
% 
% % Create a cell array to store character images
% extractedCharacters = cell(cc.NumObjects, 1);
% 
% % Iterate over each connected component (character)
% for i = 1:cc.NumObjects
%     % Extract the image of the current character
%     charImage = stats(i).Image;
% 
%     % Store the character image in a cell array
%     extractedCharacters{i} = charImage;
% end
% 
% % Display or save the extracted character images (optional)
% for i = 1:length(extractedCharacters)
%     figure;
%     imshow(extractedCharacters{i});
%     title(['Extracted Character ', num2str(i)]);
%     % Optional: Save the extracted character images to files
%     % imwrite(extractedCharacters{i}, ['char_', num2str(i), '.png']);
% end
% 
% % Now, the extractedCharacters array contains images of each individual character
% % Create a cell array to store character classification results
% characterPredictions = cell(size(extractedCharacters));
% 
% % Iterate over each extracted character image
% for i = 1:length(extractedCharacters)
%     % Extract the current character image
%     charImage = extractedCharacters{i};
% 
%     % Resize the character image to match the training data
%     charImage = imresize(charImage, [128 128]);
% 
%     % Extract features from the character image
%     charFeatures = extractFeaturesUsingGabor(charImage);
% 
%     % Classify the character using the kNN classifier
%     predictedLabel = predict(knnClassifier, charFeatures);
% 
%     % Store the classification result
%     characterPredictions{i} = predictedLabel;
% 
%     % Print the classification result for each character
%     fprintf('Classification result for character %d: %s\n', i, predictedLabel);
% end
% 
% % Local Function Definitions
% function features = extractFeaturesFromImages(imageSet, featureExtractor)
%     numImages = numel(imageSet.Files);
%     features = [];
%     for i = 1:numImages
%         img = readimage(imageSet, i);
%         imgFeatures = featureExtractor(img);
%         features = [features; imgFeatures];
%     end
% end
% 
% function features = extractFeaturesUsingGabor(image)
%     % Convert the image to grayscale if it's in color
%     if size(image, 3) == 3
%         grayImage = rgb2gray(image);
%     else
%         grayImage = image;
%     end
% 
%     % Define Gabor filter parameters
%     wavelength = 2.^(0:4) * 3; % Example wavelengths
%     orientation = 0:45:135;    % Example orientations
% 
%     % Create a Gabor filter array
%     gaborArray = gabor(wavelength, orientation);
% 
%     % Apply Gabor filters
%     gaborMag = imgaborfilt(grayImage, gaborArray);
% 
%     % Convert Gabor response images into a one-dimensional vector
%     features = reshape(gaborMag, 1, []); 
% end


% 
% 
% % Hough transform method
% % Main script part
% % Step 1: Load the dataset
% dataFolderPath = 'D:\Graduate\5405\p_dataset_26\p_dataset_26';
% imds = imageDatastore(dataFolderPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% 
% % Step 2: Split the dataset
% [trainingSet, testSet] = splitEachLabel(imds, 0.75, 'randomize');
% 
% % Step 3: Feature extraction (using Hough Transform)
% trainingFeatures = extractFeaturesFromImages(trainingSet);
% trainingLabels = trainingSet.Labels;
% testFeatures = extractFeaturesFromImages(testSet);
% 
% % Step 4: Train the kNN classifier
% k = 3; % Adjust the value of k as needed
% knnClassifier = fitcknn(trainingFeatures, trainingLabels, 'NumNeighbors', k);
% 
% % Step 5: Evaluate the classifier
% predictedLabels = predict(knnClassifier, testFeatures);
% accuracy = sum(predictedLabels == testSet.Labels) / numel(testSet.Labels);
% disp(['Accuracy: ', num2str(accuracy)]);
% 
% % Create a cell array to store extracted character images
% extractedCharacters = cell(0);
% 
% % Initialize a counter for numeric labels
% labelCounter = 1;
% binaryImageInverted = ~binaryImage;
% 
% for i = 1:length(stats)
%     % Get the bounding box of the current character
%     bbox = stats(i).BoundingBox;
% 
%     % Rounds the coordinates of the bounding box to the nearest integer value
%     x = round(bbox(1));
%     y = round(bbox(2));
%     width = round(bbox(3));
%     height = round(bbox(4));
% 
%     % Check if the width of the bounding box is greater than the height
%     if width / height > 1
%         % The width is greater than the height, and the divided area is two characters
% 
%         % Compute new bounding boxes bbox1 and bbox2
%         halfWidth = width / 2;
%         bbox1 = [x, y, halfWidth, height];
%         bbox2 = [x + halfWidth, y, halfWidth, height];
% 
%         % Use imcrop function to extract character parts from binary images and store them separately
%         charImage1 = imcrop(binaryImageInverted, bbox1);
%         charImage2 = imcrop(binaryImageInverted, bbox2);
% 
%         % Add extracted character image to cell array and assign new numeric label
%         extractedCharacters{end+1} = charImage1;
%         extractedCharacters{end+1} = charImage2;
% 
%         % Draw red bounding boxes and numerical labels to identify the two characters
%         hold on;
%         rectangle('Position', bbox1, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox1(1) + 5, bbox1(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         rectangle('Position', bbox2, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox2(1) + 5, bbox2(2) - 5, num2str(labelCounter + 1), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter to assign a new numeric label to the next character
%         labelCounter = labelCounter + 2;
%     else
%         % Width not greater than height, processing single characters
% 
%         % Use imcrop function to extract character parts from binary images
%         charImage = imcrop(binaryImageInverted, [x, y, width, height]);
% 
%         % Add extracted character image to cell array
%         extractedCharacters{end+1} = charImage;
% 
%         % Draw red bounding boxes and numerical labels to identify individual characters
%         hold on;
%         rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox(1) + 5, bbox(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter to assign a new numeric label to the next character
%         labelCounter = labelCounter + 1;
%     end
% end
% 
% % Display or save extracted character image (optional)
% for i = 1:length(extractedCharacters)
%     figure;
%     imshow(extractedCharacters{i});
%     title(['Extracted Character ', num2str(i)]);
%     % Optional: Save the extracted character image to a file
%     % imwrite(extractedCharacters{i}, ['char_', num2str(i), '.png']);
% end
% 
% % Now, the extractedCharacters array contains the image of each individual character
% % Create a cell array to store character classification results
% characterPredictions = cell(size(extractedCharacters));
% 
% % Iterate through each extracted character image
% for i = 1:length(extractedCharacters)
%     % Extract the current character image
%     charImage = extractedCharacters{i};
% 
%     % Adjust the size of the character image to match the training data
%     charImage = imresize(charImage, [32 32]);
% 
%     % Extract features from the character image
%     charFeatures = extractFeaturesUsingHough(charImage);
% 
%     % Ensure the feature vector matches the training data
%     numFeatures = 10; % This should be the number of features used in the training data
%     charFeatures = charFeatures(:)';
%     if length(charFeatures) > numFeatures
%         charFeatures = charFeatures(1:numFeatures);
%     elseif length(charFeatures) < numFeatures
%         charFeatures = [charFeatures zeros(1, numFeatures - length(charFeatures))];
%     end
% 
%     % Predict using the KNN classifier
%     predictedLabel = predict(knnClassifier, charFeatures);
% 
%     % Store the classification result
%     characterPredictions{i} = predictedLabel;
% 
%     % Print the classification result for each character
%     fprintf('Classification result for character %d: %s\n', i, predictedLabel);
% end
% 
% % Local function definitions
% function features = extractFeaturesFromImages(imageSet)
%     numImages = numel(imageSet.Files);
%     numFeatures = 10; % Assume each feature vector has 10 elements
%     features = zeros(numImages, numFeatures); % Pre-allocate feature matrix
%     for i = 1:numImages
%         img = readimage(imageSet, i);
%         % Extract features
%         imgFeatures = extractFeaturesUsingHough(img);
%         % Ensure consistent feature vector length
%         imgFeatures = imgFeatures(:)';
%         if length(imgFeatures) > numFeatures
%             imgFeatures = imgFeatures(1:numFeatures); % Truncate if too long
%         elseif length(imgFeatures) < numFeatures
%             imgFeatures = [imgFeatures zeros(1, numFeatures - length(imgFeatures))]; % Fill with zeros if too short
%         end
%         features(i, :) = imgFeatures;
%     end
% end
% 
% function features = extractFeaturesUsingHough(image)
%     % Convert to grayscale if the image is not a binary image
%     if size(image, 3) == 3
%         grayImage = rgb2gray(image);
%     else
%         grayImage = image;
%     end
% 
%     % Extract features using Hough Transform (add your Hough Transform code here)
%     % Example: Detect lines using Hough Transform
%     [H, T, R] = hough(grayImage);
%     P = houghpeaks(H, 5); % Select the strongest Hough peaks
%     lines = houghlines(grayImage, T, R, P);
% 
%     % Extract features from lines and convert them into a one-dimensional vector
%     % Example: Extract the number of lines, angles, etc.
%     numLines = length(lines);
%     angles = [lines.theta]; % Extract angles of the lines
%     % ...
% 
%     % Convert features into a one-dimensional vector
%     features = [numLines, angles];
% end





% % SVM

% % pixel value method()
% % Main script part
% % Step 1: Load the dataset
% % Specify the folder path containing the H letter picture
% folderPathH = 'D:\Graduate\5405\p_dataset_26\p_dataset_26\SampleH';
% 
% % Specify the folder path containing the E letter picture
% folderPathE = 'D:\Graduate\5405\p_dataset_26\p_dataset_26\SampleE';
% 
% % Specify the folder path containing the D letter picture
% folderPathD = 'D:\Graduate\5405\p_dataset_26\p_dataset_26\SampleD';
% 
% % Specify the folder path containing the L letter picture
% folderPathL = 'D:\Graduate\5405\p_dataset_26\p_dataset_26\SampleL';
% 
% % Specify the folder path containing the O letter picture
% folderPathO = 'D:\Graduate\5405\p_dataset_26\p_dataset_26\SampleO';
% 
% % Specify the folder path containing the R letter picture
% folderPathR = 'D:\Graduate\5405\p_dataset_26\p_dataset_26\SampleR';
% 
% % Specify the folder path containing the W letter picture
% folderPathW = 'D:\Graduate\5405\p_dataset_26\p_dataset_26\SampleW';
% 
% % Use imageDatastore to load the dataset, including image data in subfolders, and assign new labels
% imdsH = imageDatastore(folderPathH, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% imdsE = imageDatastore(folderPathE, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% imdsD = imageDatastore(folderPathD, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% imdsL = imageDatastore(folderPathL, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% imdsO = imageDatastore(folderPathO, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% imdsR = imageDatastore(folderPathR, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% imdsW = imageDatastore(folderPathW, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% 
% % Merge all data into one data set
% imdsCombined = imageDatastore(cat(1, imdsH.Files, imdsE.Files, imdsD.Files, imdsL.Files, imdsO.Files, imdsR.Files, imdsW.Files), 'LabelSource', 'foldernames');
% 
% 
% % Shuffle the data set
% imdsCombined = shuffle(imdsCombined);
% 
% 
% % Display some information from the data set
% disp(numel(imdsCombined.Files)); % Display the total number of samples
% disp(unique(imdsCombined.Labels)); % Display new category labels
% 
% 
% % Combine all datasets
% imdsCombined = imageDatastore(cat(1, imdsH.Files, imdsE.Files, imdsD.Files, imdsL.Files, imdsO.Files, imdsR.Files, imdsW.Files), 'LabelSource', 'foldernames');
% 
% % Shuffle the combined dataset
% imdsCombined = shuffle(imdsCombined);
% 
% % Display dataset information
% disp(numel(imdsCombined.Files)); % Total number of samples
% disp(unique(imdsCombined.Labels)); % Category labels
% 
% % Step 2: Split the dataset
% [trainingSet, testSet] = splitEachLabel(imdsCombined, 0.75, 'randomize');
% 
% % Step 3: Feature extraction
% trainingFeatures = extractFeaturesFromImages(trainingSet, @extractFeaturesUsingRawPixels);
% trainingLabels = trainingSet.Labels;
% testFeatures = extractFeaturesFromImages(testSet, @extractFeaturesUsingRawPixels);
% 
% % Step 4: Train SVM classifier for multi-class classification
% SVMClassifier = fitcecoc(trainingFeatures, trainingLabels);
% 
% % Step 5: Evaluate the classifier
% predictedLabels = predict(SVMClassifier, testFeatures);
% accuracy = sum(predictedLabels == testSet.Labels) / numel(testSet.Labels);
% disp(['Accuracy: ', num2str(accuracy)]);
% 
% 
% 
% % Create a cell array that stores the extracted character images
% extractedCharacters = cell(0);
% 
% % Initialize a counter for numeric labels
% labelCounter = 1;
% binaryImageInverted = ~binaryImage;
% 
% for i = 1:length(stats)
%     % Get the bounding box of the current character
%     bbox = stats(i).BoundingBox;
% 
%     % Rounds the coordinates of the bounding box to the nearest integer value
%     x = round(bbox(1));
%     y = round(bbox(2));
%     width = round(bbox(3));
%     height = round(bbox(4));
% 
%     % Check if the width of the bounding box is greater than the height
%     if width / height > 1
%         % The width is greater than the height, and the divided area is two characters
% 
%         % Compute new two bounding boxes bbox1 and bbox2
%         halfWidth = width / 2;
%         bbox1 = [x, y, halfWidth, height];
%         bbox2 = [x + halfWidth, y, halfWidth, height];
% 
%         % Use imcrop function to extract character parts from binary images and store them separately
%         charImage1 = imcrop(binaryImageInverted, bbox1);
%         charImage2 = imcrop(binaryImageInverted, bbox2);
% 
%         % Add extracted character image to cell array and assign new numeric label
%         extractedCharacters{end+1} = charImage1;
%         extractedCharacters{end+1} = charImage2;
% 
%         % Draw red bounding boxes and numerical labels to identify the two characters
%         hold on;
%         rectangle('Position', bbox1, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox1(1) + 5, bbox1(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         rectangle('Position', bbox2, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox2(1) + 5, bbox2(2) - 5, num2str(labelCounter + 1), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter to assign a new numeric label to the next character
%         labelCounter = labelCounter + 2;
%     else
%         % Width not greater than height, processing single characters
% 
%         % Use imcrop function to extract character parts from binary images
%         charImage = imcrop(binaryImageInverted, [x, y, width, height]);
% 
%         % Add extracted character image to cell array
%         extractedCharacters{end+1} = charImage;
% 
%         % Draw red bounding boxes and numerical labels to identify individual characters
%         hold on;
%         rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox(1) + 5, bbox(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter to assign a new numeric label to the next character
%         labelCounter = labelCounter + 1;
%     end
% end
% 
% % Display or save extracted character image (optional)
% for i = 1:length(extractedCharacters)
%     figure;
%     imshow(extractedCharacters{i});
%     title(['Extracted Character ', num2str(i)]);
%     % Optional: Save the extracted character image to a file
%     % imwrite(extractedCharacters{i}, ['char_', num2str(i), '.png']);
% end
% 
% 
% 
% 
% % Now, the extractedCharacters array contains an image of each individual character
% % Create a cell array to store character classification results
% % Character classification
% characterPredictions = cell(size(extractedCharacters));
% 
% for i = 1:length(extractedCharacters)
%     charImage = extractedCharacters{i};
%     charImage = imresize(charImage, [32 32]);
%     charFeatures = extractFeaturesUsingRawPixels(charImage);
%     predictedLabel = predict(SVMClassifier, charFeatures);
%     characterPredictions{i} = predictedLabel;
%     fprintf('Classification results for character %d：%s\n', i, predictedLabel);
% end
% 
% 
% 
% 
% % local function definition
% function features = extractFeaturesFromImages(imageSet, featureExtractor)
%     numImages = numel(imageSet.Files);
%     features = [];
%     for i = 1:numImages
%         img = readimage(imageSet, i);
%         imgFeatures = featureExtractor(img);
%         features = [features; imgFeatures];
%     end
% end
% 
% function features = extractFeaturesUsingRawPixels(image)
%     % If the image is not binary, resize and convert to grayscale image
%     if ~islogical(image)
%         resizedImage = imresize(image, [32 32]);
%         if size(resizedImage, 3) == 3
%             grayImage = rgb2gray(resizedImage);
%         else
%             grayImage = resizedImage;
%         end
%         features = double(grayImage(:)');
%     else
%         % If the image is already binary, resize it directly
%         resizedImage = imresize(image, [32 32]);
%         features = double(resizedImage(:)');
%     end
% end


% % Histogram
% % Main script part
% % % Step 1: Load the dataset
% dataFolderPath = 'D:\Graduate\5405\p_dataset_26\p_dataset_26';
% imds = imageDatastore(dataFolderPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% 
% % % Step 2: Split the dataset
% [trainingSet, testSet] = splitEachLabel(imds, 0.75, 'randomize');
% 
% % % Step 3: Feature extraction (using grayscale histogram method)
% trainingFeatures = extractFeaturesFromImages(trainingSet, @extractFeaturesUsingHistogram);
% trainingLabels = trainingSet.Labels;
% testFeatures = extractFeaturesFromImages(testSet, @extractFeaturesUsingHistogram);
% 
% % % Step 4: Train SVM classifier for multi-class problem
% svmClassifier = fitcecoc(trainingFeatures, trainingLabels);
% 
% % % Step 5: Evaluate the classifier
% predictedLabels = predict(svmClassifier, testFeatures);
% accuracy = sum(predictedLabels == testSet.Labels) / numel(testSet.Labels);
% disp(['Accuracy: ', num2str(accuracy)]);
% 
% 
% % Create a cell array that stores the extracted character images
% extractedCharacters = cell(0);
% binaryImageInverted = ~binaryImage;
% % Initialize a counter for numeric labels
% labelCounter = 1;
% 
% for i = 1:length(stats)
%     % Get the bounding box of the current character
%     bbox = stats(i).BoundingBox;
% 
%     % Round the coordinates of the bounding box to the nearest integer value
%     x = round(bbox(1));
%     y = round(bbox(2));
%     width = round(bbox(3));
%     height = round(bbox(4));
% 
%     % Check if the width of the bounding box is greater than the height
%     if width / height > 1
%         % The width is greater than the height, and the divided area is two characters
% 
%         % Compute new two bounding boxes bbox1 and bbox2
%         halfWidth = width / 2;
%         bbox1 = [x, y, halfWidth, height];
%         bbox2 = [x + halfWidth, y, halfWidth, height];
% 
%         % Use imcrop function to extract character parts from binary images and store them separately
%         charImage1 = imcrop(binaryImageInverted , bbox1);
%         charImage2 = imcrop(binaryImageInverted , bbox2);
% 
%         % Add extracted character image to cell array and assign new numeric label
%         extractedCharacters{end+1} = charImage1;
%         extractedCharacters{end+1} = charImage2;
% 
%         % Draw red bounding boxes and numerical labels to identify the two characters
%         hold on;
%         rectangle('Position', bbox1, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox1(1) + 5, bbox1(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         rectangle('Position', bbox2, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox2(1) + 5, bbox2(2) - 5, num2str(labelCounter + 1), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter to assign a new numeric label to the next character
%         labelCounter = labelCounter + 2;
%     else
%         % Width not greater than height, processing single characters
% 
%         % Use imcrop function to extract character parts from binary images
%         charImage = imcrop(binaryImageInverted , [x, y, width, height]);
% 
%         % Add extracted character image to cell array
%         extractedCharacters{end+1} = charImage;
% 
%         % Draw red bounding boxes and numerical labels to identify individual characters
%         hold on;
%         rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox(1) + 5, bbox(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter to assign a new numeric label to the next character
%         labelCounter = labelCounter + 1;
%     end
% end
% 
% % Display or save extracted character image (optional)
% for i = 1:length(extractedCharacters)
%     figure;
%     imshow(extractedCharacters{i});
%     title(['Extracted Character ', num2str(i)]);
%     % Optional: Save the extracted character image to a file
%     % imwrite(extractedCharacters{i}, ['char_', num2str(i), '.png']);
% end
% 
% % Connected component analysis
% cc = bwconncomp(binaryImage);
% stats = regionprops(cc, 'Image');
% 
% %Create a cell array that stores character images
% extractedCharacters = cell(cc.NumObjects, 1);
% 
% % Iterate over each connected component (character)
% for i = 1:cc.NumObjects
%     % Extract the image of the current character
%     charImage = stats(i).Image;
% 
%     % Store character image in cell array
%     extractedCharacters{i} = charImage;
% end
% 
% % Display or save the extracted character image (optional）
% for i = 1:length(extractedCharacters)
%     figure;
%     imshow(extractedCharacters{i});
%     title(['Extracted Character ', num2str(i)]);
%     % Optional: Save the extracted character image to a file
%     % imwrite(extractedCharacters{i}, ['char_', num2str(i), '.png']);
% end
% 
% 
% 
% % Now, the extractedCharacters array contains an image of each individual character
% % Create a cell array to store character classification results
% characterPredictions = cell(size(extractedCharacters));
% 
% % Iterate through each extracted character image
% for i = 1:length(extractedCharacters)
%     % Extract current character image
%     charImage = extractedCharacters{i};
% 
%     % Resize character images to match training data
%     charImage = imresize(charImage, [32 32]);
% 
%     % Extract features of character images
%     charFeatures = extractFeaturesUsingHistogram(charImage);
% 
%     % Classify characters using KNN classifier
%     predictedLabel = predict(svmClassifier, charFeatures);
% 
%     % Store classification results
%     characterPredictions{i} = predictedLabel;
% 
%     % Print the classification results for each character
%     fprintf('Classification results for character %d：%s\n', i, predictedLabel);
% end
% 
% % local function definition
% function features = extractFeaturesUsingHistogram(image)
%     % Convert image to grayscale (if it is in color)
%     if size(image, 3) == 3
%         grayImage = rgb2gray(image);
%     else
%         grayImage = image;
%     end
% 
%       % Compute grayscale histogram as feature
%     % Assume 2 bins are used since the logical image only has two values
%     features = imhist(grayImage, 2)'; 
%     features = features / sum(features); % normalized histogram
% end
% function features = extractFeaturesFromImages(imageSet, featureExtractor)
%     numImages = numel(imageSet.Files);
%     features = []; % Initialize an empty matrix to store features
% 
%     for i = 1:numImages
%         img = readimage(imageSet, i); % Read each image from the data store
%         imgFeatures = featureExtractor(img); % Apply feature extraction function
%         features = [features; imgFeatures]; % Add features to matrix
%     end
% end


% % Edges
% % Main script part
% % Step 1: Load the dataset
% dataFolderPath = 'D:\Graduate\5405\p_dataset_26\p_dataset_26';
% imds = imageDatastore(dataFolderPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% 
% % Step 2: Split the dataset
% [trainingSet, testSet] = splitEachLabel(imds, 0.75, 'randomize');
% 
% % Step 3: Feature extraction (using edge detection method)
% trainingFeatures = extractFeaturesFromImages(trainingSet, @extractFeaturesUsingEdges);
% trainingLabels = trainingSet.Labels;
% testFeatures = extractFeaturesFromImages(testSet, @extractFeaturesUsingEdges);
% 
% % Step 4: Train SVM classifier for multi-class problem
% svmClassifier = fitcecoc(trainingFeatures, trainingLabels);
% 
% % Step 5: Evaluate the classifier
% predictedLabels = predict(svmClassifier, testFeatures);
% accuracy = sum(predictedLabels == testSet.Labels) / numel(testSet.Labels);
% disp(['Accuracy: ', num2str(accuracy)]);
% 
% 
% % Create a cell array that stores the extracted character images
% extractedCharacters = cell(0);
% binaryImageInverted = ~binaryImage;
% % Initialize a counter for numeric labels
% labelCounter = 1;
% 
% for i = 1:length(stats)
%     % Get the bounding box of the current character
%     bbox = stats(i).BoundingBox;
% 
%     % Round the coordinates of the bounding box to the nearest integer value
%     x = round(bbox(1));
%     y = round(bbox(2));
%     width = round(bbox(3));
%     height = round(bbox(4));
% 
%     % Check if the width of the bounding box is greater than the height
%     if width / height > 1
%         % The width is greater than the height, and the divided area is two characters
% 
%         % Compute new two bounding boxes bbox1 and bbox2
%         halfWidth = width / 2;
%         bbox1 = [x, y, halfWidth, height];
%         bbox2 = [x + halfWidth, y, halfWidth, height];
% 
%         % Use imcrop function to extract character parts from binary images and store them separately
%         charImage1 = imcrop(binaryImageInverted , bbox1);
%         charImage2 = imcrop(binaryImageInverted , bbox2);
% 
%         % Add extracted character image to cell array and assign new numeric label
%         extractedCharacters{end+1} = charImage1;
%         extractedCharacters{end+1} = charImage2;
% 
%         % Draw red bounding boxes and numerical labels to identify the two characters
%         hold on;
%         rectangle('Position', bbox1, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox1(1) + 5, bbox1(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         rectangle('Position', bbox2, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox2(1) + 5, bbox2(2) - 5, num2str(labelCounter + 1), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter to assign a new numeric label to the next character
%         labelCounter = labelCounter + 2;
%     else
%         % Width not greater than height, processing single characters
% 
%         % Use imcrop function to extract character parts from binary images
%         charImage = imcrop(binaryImageInverted , [x, y, width, height]);
% 
%         % Add extracted character image to cell array
%         extractedCharacters{end+1} = charImage;
% 
%         % Draw red bounding boxes and numerical labels to identify individual characters
%         hold on;
%         rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox(1) + 5, bbox(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter to assign a new numeric label to the next character
%         labelCounter = labelCounter + 1;
%     end
% end
% 
% % Display or save extracted character image (optional)
% for i = 1:length(extractedCharacters)
%     figure;
%     imshow(extractedCharacters{i});
%     title(['Extracted Character ', num2str(i)]);
%     % Optional: Save the extracted character image to a file
%     % imwrite(extractedCharacters{i}, ['char_', num2str(i), '.png']);
% end
% 
% % Connected component analysis
% cc = bwconncomp(binaryImage);
% stats = regionprops(cc, 'Image');
% 
% %Create a cell array that stores character images
% extractedCharacters = cell(cc.NumObjects, 1);
% 
% % Iterate over each connected component (character)
% for i = 1:cc.NumObjects
%     % Extract the image of the current character
%     charImage = stats(i).Image;
% 
%     % Store character image in cell array
%     extractedCharacters{i} = charImage;
% end
% 
% % Display or save the extracted character image (optional）
% for i = 1:length(extractedCharacters)
%     figure;
%     imshow(extractedCharacters{i});
%     title(['Extracted Character ', num2str(i)]);
%     % Optional: Save the extracted character image to a file
%     % imwrite(extractedCharacters{i}, ['char_', num2str(i), '.png']);
% end
% 
% 
% 
% % Now, the extractedCharacters array contains an image of each individual character
% % Create a cell array to store character classification results
% characterPredictions = cell(size(extractedCharacters));
% 
% % Iterate through each extracted character image
% for i = 1:length(extractedCharacters)
%     % Extract current character image
%     charImage = extractedCharacters{i};
% 
%     % Resize character images to match training data
%     charImage = imresize(charImage, [32 32]);
% 
%     % Extract features of character images
%     charFeatures = extractFeaturesUsingHistogram(charImage);
% 
%     % Classify characters using KNN classifier
%     predictedLabel = predict(svmClassifier, charFeatures);
% 
%     % Store classification results
%     characterPredictions{i} = predictedLabel;
% 
%     % Print the classification results for each character
%     fprintf('Classification results for character %d：%s\n', i, predictedLabel);
% end
% 
% % local function definition
% % Function for edge-based feature extraction
% function features = extractFeaturesUsingEdges(image)
%     % Convert image to grayscale if it is not already
%     if size(image, 3) == 3
%         grayImage = rgb2gray(image);
%     else
%         grayImage = image;
%     end
% 
%     % Apply edge detection
%     edges = edge(grayImage, 'Canny');
% 
%     % Convert edges to a feature vector
%     features = edges(:)';
% end
% 
% % Function to extract features from image set
% function features = extractFeaturesFromImages(imageSet, featureExtractor)
%     numImages = numel(imageSet.Files);
%     features = []; % Initialize an empty matrix to store features
% 
%     for i = 1:numImages
%         img = readimage(imageSet, i); % Read each image from the data store
%         imgFeatures = featureExtractor(img); % Apply feature extraction function
%         features = [features; imgFeatures]; % Add features to matrix
%     end
% end

% % Gabor
% % Main script part
% % Step 1: Load the dataset
% dataFolderPath = 'D:\Graduate\5405\p_dataset_26\p_dataset_26';
% imds = imageDatastore(dataFolderPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% 
% % Step 2: Split the dataset
% [trainingSet, testSet] = splitEachLabel(imds, 0.75, 'randomize');
% 
% % Step 3: Feature extraction (using Gabor method)
% trainingFeatures = extractFeaturesFromImages(trainingSet, @extractFeaturesUsingGabor);
% trainingLabels = trainingSet.Labels;
% testFeatures = extractFeaturesFromImages(testSet, @extractFeaturesUsingGabor);
% 
% % Step 4: Train SVM classifier for multi-class problem
% svmClassifier = fitcecoc(trainingFeatures, trainingLabels);
% 
% % Step 5: Evaluate the classifier
% predictedLabels = predict(svmClassifier, testFeatures);
% accuracy = sum(predictedLabels == testSet.Labels) / numel(testSet.Labels);
% disp(['Accuracy: ', num2str(accuracy)]);
% 
% 
% % Create a cell array that stores the extracted character images
% extractedCharacters = cell(0);
% binaryImageInverted = ~binaryImage;
% % Initialize a counter for numeric labels
% labelCounter = 1;
% 
% for i = 1:length(stats)
%     % Get the bounding box of the current character
%     bbox = stats(i).BoundingBox;
% 
%     % Round the coordinates of the bounding box to the nearest integer value
%     x = round(bbox(1));
%     y = round(bbox(2));
%     width = round(bbox(3));
%     height = round(bbox(4));
% 
%     % Check if the width of the bounding box is greater than the height
%     if width / height > 1
%         % The width is greater than the height, and the divided area is two characters
% 
%         % Compute new two bounding boxes bbox1 and bbox2
%         halfWidth = width / 2;
%         bbox1 = [x, y, halfWidth, height];
%         bbox2 = [x + halfWidth, y, halfWidth, height];
% 
%         % Use imcrop function to extract character parts from binary images and store them separately
%         charImage1 = imcrop(binaryImageInverted , bbox1);
%         charImage2 = imcrop(binaryImageInverted , bbox2);
% 
%         % Add extracted character image to cell array and assign new numeric label
%         extractedCharacters{end+1} = charImage1;
%         extractedCharacters{end+1} = charImage2;
% 
%         % Draw red bounding boxes and numerical labels to identify the two characters
%         hold on;
%         rectangle('Position', bbox1, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox1(1) + 5, bbox1(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         rectangle('Position', bbox2, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox2(1) + 5, bbox2(2) - 5, num2str(labelCounter + 1), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter to assign a new numeric label to the next character
%         labelCounter = labelCounter + 2;
%     else
%         % Width not greater than height, processing single characters
% 
%         % Use imcrop function to extract character parts from binary images
%         charImage = imcrop(binaryImageInverted , [x, y, width, height]);
% 
%         % Add extracted character image to cell array
%         extractedCharacters{end+1} = charImage;
% 
%         % Draw red bounding boxes and numerical labels to identify individual characters
%         hold on;
%         rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox(1) + 5, bbox(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter to assign a new numeric label to the next character
%         labelCounter = labelCounter + 1;
%     end
% end
% 
% % Display or save extracted character image (optional)
% for i = 1:length(extractedCharacters)
%     figure;
%     imshow(extractedCharacters{i});
%     title(['Extracted Character ', num2str(i)]);
%     % Optional: Save the extracted character image to a file
%     % imwrite(extractedCharacters{i}, ['char_', num2str(i), '.png']);
% end
% 
% % Connected component analysis
% cc = bwconncomp(binaryImage);
% stats = regionprops(cc, 'Image');
% 
% %Create a cell array that stores character images
% extractedCharacters = cell(cc.NumObjects, 1);
% 
% % Iterate over each connected component (character)
% for i = 1:cc.NumObjects
%     % Extract the image of the current character
%     charImage = stats(i).Image;
% 
%     % Store character image in cell array
%     extractedCharacters{i} = charImage;
% end
% 
% % Display or save the extracted character image (optional）
% for i = 1:length(extractedCharacters)
%     figure;
%     imshow(extractedCharacters{i});
%     title(['Extracted Character ', num2str(i)]);
%     % Optional: Save the extracted character image to a file
%     % imwrite(extractedCharacters{i}, ['char_', num2str(i), '.png']);
% end
% 
% 
% 
% % Now, the extractedCharacters array contains an image of each individual character
% % Create a cell array to store character classification results
% characterPredictions = cell(size(extractedCharacters));
% 
% % Iterate through each extracted character image
% for i = 1:length(extractedCharacters)
%     % Extract current character image
%     charImage = extractedCharacters{i};
% 
%     % Resize character images to match training data
%     charImage = imresize(charImage, [32 32]);
% 
%     % Extract features of character images
%     charFeatures = extractFeaturesUsingHistogram(charImage);
% 
%     % Classify characters using KNN classifier
%     predictedLabel = predict(svmClassifier, charFeatures);
% 
%     % Store classification results
%     characterPredictions{i} = predictedLabel;
% 
%     % Print the classification results for each character
%     fprintf('Classification results for character %d：%s\n', i, predictedLabel);
% end
% 
% % local function definition
% % Function for edge-based feature extraction
% function features = extractFeaturesUsingGabor(image)
%     if size(image, 3) == 3
%         grayImage = rgb2gray(image);
%     else
%         grayImage = image;
%     end
% 
%     % Gabor Filter parameters
%     wavelength = 2.^(0:5) * 3; % A range of wavelengths
%     orientation = 0:45:135;    % Four orientations (0, 45, 90, 135 degrees)
% 
%     gaborArray = gabor(wavelength, orientation);
%     gaborMag = imgaborfilt(grayImage, gaborArray);
% 
%     % Feature extraction
%     numFilters = length(gaborArray);
%     features = zeros(1, numFilters);
%     for i = 1:numFilters
%         features(i) = mean2(gaborMag(:,:,i));
%     end
% end
% 
% 
% % Function to extract features from image set
% function features = extractFeaturesFromImages(imageSet, featureExtractor)
%     numImages = numel(imageSet.Files);
%     features = []; % Initialize an empty matrix to store features
% 
%     for i = 1:numImages
%         img = readimage(imageSet, i); % Read each image from the data store
%         imgFeatures = featureExtractor(img); % Apply feature extraction function
%         features = [features; imgFeatures]; % Add features to matrix
%     end
% end

% % HoughTransform
% % Main script part
% % Define the expected length of the feature vector
% expectedFeatureLength = 2; % Adjust this value based on your feature extraction method
% 
% % Step 1: Load the dataset
% dataFolderPath = 'D:\Graduate\5405\p_dataset_26\p_dataset_26';
% imds = imageDatastore(dataFolderPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% 
% % Step 2: Split the dataset
% [trainingSet, testSet] = splitEachLabel(imds, 0.75, 'randomize');
% 
% % Expected length of the feature vector
% expectedFeatureLength = 256; % Assuming the use of a normalized histogram of length 256 as features
% 
% % Feature extraction
% trainingFeatures = extractFeaturesFromImages(trainingSet, @extractFeaturesUsingHoughTransform, expectedFeatureLength);
% trainingLabels = trainingSet.Labels;
% testFeatures = extractFeaturesFromImages(testSet, @extractFeaturesUsingHoughTransform, expectedFeatureLength);
% 
% % Check the size of the extracted features
% disp(['Size of trainingFeatures: ', num2str(size(trainingFeatures))]);
% disp(['Length of trainingLabels: ', num2str(length(trainingLabels))]);
% assert(size(trainingFeatures, 1) == length(trainingLabels), 'Mismatch between the number of features and labels');
% 
% % Train SVM classifier
% svmClassifier = fitcecoc(trainingFeatures, trainingLabels);
% 
% % Evaluate the classifier
% predictedLabels = predict(svmClassifier, testFeatures);
% accuracy = sum(predictedLabels == testSet.Labels) / numel(testSet.Labels);
% disp(['Accuracy: ', num2str(accuracy)]);
% 
% % Character extraction and processing
% % ... (Your existing code for character extraction and processing here)
% 
% % Connected component analysis
% cc = bwconncomp(binaryImage);
% stats = regionprops(cc, 'Image');
% 
% % Create a cell array that stores character images
% extractedCharacters = cell(cc.NumObjects, 1);
% 
% % Iterate over each connected component (character)
% for i = 1:cc.NumObjects
%     % Extract the image of the current character
%     charImage = stats(i).Image;
% 
%     % Store character image in cell array
%     extractedCharacters{i] = charImage;
% end
% 
% % Display or save the extracted character image (optional)
% for i = 1:length(extractedCharacters)
%     figure;
%     imshow(extractedCharacters{i});
%     title(['Extracted Character ', num2str(i)]);
%     % Optional: Save the extracted character image to a file
%     % imwrite(extractedCharacters{i}, ['char_', num2str(i), '.png']);
% end
% 
% % Now, the extractedCharacters array contains an image of each individual character
% % Create a cell array to store character classification results
% characterPredictions = cell(size(extractedCharacters));
% 
% % Iterate through each extracted character image
% for i = 1:length(extractedCharacters)
%     % Extract the current character image
%     charImage = extractedCharacters{i};
% 
%     % Resize character image to match training data
%     charImage = imresize(charImage, [32 32]); % Ensure this size is consistent with training
% 
%     % Extract features of the character image
%     charFeatures = extractFeaturesUsingHoughTransform(charImage, expectedFeatureLength);
% 
%     % Ensure the length of the extracted features matches the training feature length
%     if length(charFeatures) ~= expectedFeatureLength
%         error('The length of the extracted features does not match the training feature length');
%     end
% 
%     % Reshape feature vector to match the SVM classifier's input format
%     charFeatures = reshape(charFeatures, 1, []); % Ensure the feature vector is a row vector
% 
%     % Classify characters using SVM classifier
%     predictedLabel = predict(svmClassifier, charFeatures);
% 
%     % Store classification results
%     characterPredictions{i} = predictedLabel;
% 
%     % Print classification results for each character
%     fprintf('Classification results for character %d: %s\n', i, predictedLabel);
% end
% 
% % Local function definitions
% % Function for edge-based feature extraction
% function features = extractFeaturesFromImages(imageSet, featureExtractor, featureLength)
%     numImages = numel(imageSet.Files);
%     features = zeros(numImages, featureLength); % Initialize feature matrix
% 
%     for i = 1:numImages
%         img = readimage(imageSet, i); % Read each image from the datastore
%         imgFeatures = featureExtractor(img, featureLength); % Apply feature extraction function
% 
%         % Ensure successful feature extraction for each image
%         if length(imgFeatures) ~= featureLength
%             warning(['Feature extraction failed for image index: ', num2str(i)]);
%             continue;
%         end
% 
%         features(i, :) = imgFeatures; % Add features to the matrix
%     end
% end
% 
% function features = extractFeaturesUsingHoughTransform(image, featureLength)
%     % Convert image to grayscale if not already
%     if size(image, 3) == 3
%         grayImage = rgb2gray(image);
%     else
%         grayImage = image;
%     end
% 
%     % Choose appropriate histogram settings based on image type
%     if islogical(grayImage)
%         % For binary images, use 2 histogram bins
%         histogramValues = imhist(grayImage, 2);
%     else
%         % For non-binary images, use 256 histogram bins
%         histogramValues = imhist(grayImage, 256);
%     end
% 
%     % Normalize the histogram
%     normalizedHistogram = histogramValues / sum(histogramValues);
% 
%     % Ensure histogram length is at least featureLength
%     if length(normalizedHistogram) < featureLength
%         normalizedHistogram(end+1:featureLength) = 0; % Pad with zeros to featureLength
%     end
% 
%     % Return the first featureLength features
%     features = normalizedHistogram(1:featureLength);
% end


% % SOM
%%Rawpixel
% % Main script part
% % Step 1: Load the dataset
% % Specify the folder path containing the H letter picture
% folderPathH = 'D:\Graduate\5405\p_dataset_26\p_dataset_26\SampleH';
% 
% % Specify the folder path containing the E letter picture
% folderPathE = 'D:\Graduate\5405\p_dataset_26\p_dataset_26\SampleE';
% 
% % Specify the folder path containing the D letter picture
% folderPathD = 'D:\Graduate\5405\p_dataset_26\p_dataset_26\SampleD';
% 
% % Specify the folder path containing the L letter picture
% folderPathL = 'D:\Graduate\5405\p_dataset_26\p_dataset_26\SampleL';
% 
% % Specify the folder path containing the O letter picture
% folderPathO = 'D:\Graduate\5405\p_dataset_26\p_dataset_26\SampleO';
% 
% % Specify the folder path containing the R letter picture
% folderPathR = 'D:\Graduate\5405\p_dataset_26\p_dataset_26\SampleR';
% 
% % Specify the folder path containing the W letter picture
% folderPathW = 'D:\Graduate\5405\p_dataset_26\p_dataset_26\SampleW';
% 
% % Use imageDatastore to load the dataset, including image data in subfolders, and assign new labels
% imdsH = imageDatastore(folderPathH, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% imdsE = imageDatastore(folderPathE, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% imdsD = imageDatastore(folderPathD, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% imdsL = imageDatastore(folderPathL, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% imdsO = imageDatastore(folderPathO, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% imdsR = imageDatastore(folderPathR, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% imdsW = imageDatastore(folderPathW, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% 
% % Merge all data into one dataset
% imdsCombined = imageDatastore(cat(1, imdsH.Files, imdsE.Files, imdsD.Files, imdsL.Files, imdsO.Files, imdsR.Files, imdsW.Files), 'LabelSource', 'foldernames');
% 
% % Shuffle the dataset
% imdsCombined = shuffle(imdsCombined);
% 
% % Display some information from the dataset
% disp(numel(imdsCombined.Files)); % Display the total number of samples
% disp(unique(imdsCombined.Labels)); % Display new category labels
% 
% % Step 2: Split the dataset
% [trainingSet, testSet] = splitEachLabel(imdsCombined, 0.75, 'randomize');
% trainingFeatures = extractFeaturesFromImages(trainingSet, @extractFeaturesUsingRawPixels);
% trainingLabels = trainingSet.Labels;
% 
% % Step 3: Create and train SOM network
% % Define grid size and training parameters
% gridSize = [10, 10]; % Grid size can be adjusted as needed
% somNet = selforgmap(gridSize);
% 
% % Train SOM network
% [somNet, tr] = train(somNet, trainingFeatures');
% testFeatures = extractFeaturesFromImages(testSet, @extractFeaturesUsingRawPixels);
% 
% % Step 4: Classify using SOM
% neuronIndices = vec2ind(somNet(trainingFeatures'));
% neuronLabels = arrayfun(@(x) mode(trainingLabels(neuronIndices == x)), 1:prod(gridSize));
% 
% % Step 5: Evaluate the classifier
% testNeuronIndices = vec2ind(somNet(testFeatures'));
% predictedLabels = arrayfun(@(x) neuronLabels(x), testNeuronIndices);
% accuracy = sum(predictedLabels' == testSet.Labels) / numel(testSet.Labels);
% disp(['Accuracy: ', num2str(accuracy)]);
% 
% % Create a cell array that stores the extracted character images
% extractedCharacters = cell(0);
% 
% % Initialize a counter for numeric labels
% labelCounter = 1;
% binaryImageInverted = ~binaryImage;
% 
% for i = 1:length(stats)
%     % Get the bounding box of the current character
%     bbox = stats(i).BoundingBox;
% 
%     % Round the coordinates of the bounding box to the nearest integer value
%     x = round(bbox(1));
%     y = round(bbox(2));
%     width = round(bbox(3));
%     height = round(bbox(4));
% 
%     % Check if the width of the bounding box is greater than the height
%     if width / height > 1
%         % The width is greater than the height, and the divided area is two characters
% 
%         % Compute new two bounding boxes bbox1 and bbox2
%         halfWidth = width / 2;
%         bbox1 = [x, y, halfWidth, height];
%         bbox2 = [x + halfWidth, y, halfWidth, height];
% 
%         % Use imcrop function to extract character parts from binary images and store them separately
%         charImage1 = imcrop(binaryImageInverted, bbox1);
%         charImage2 = imcrop(binaryImageInverted, bbox2);
% 
%         % Add extracted character image to cell array and assign new numeric label
%         extractedCharacters{end+1} = charImage1;
%         extractedCharacters{end+1} = charImage2;
% 
%         % Draw red bounding boxes and numerical labels to identify the two characters
%         hold on;
%         rectangle('Position', bbox1, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox1(1) + 5, bbox1(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         rectangle('Position', bbox2, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox2(1) + 5, bbox2(2) - 5, num2str(labelCounter + 1), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter to assign a new numeric label to the next character
%         labelCounter = labelCounter + 2;
%     else
%         % Width not greater than height, processing single characters
% 
%         % Use imcrop function to extract character parts from binary images
%         charImage = imcrop(binaryImageInverted, [x, y, width, height]);
% 
%         % Add extracted character image to cell array
%         extractedCharacters{end+1} = charImage;
% 
%         % Draw red bounding boxes and numerical labels to identify individual characters
%         hold on;
%         rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox(1) + 5, bbox(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter to assign a new numeric label to the next character
%         labelCounter = labelCounter + 1;
%     end
% end
% 
% % Display or save extracted character image (optional)
% characterPredictions = cell(size(extractedCharacters));
% for i = 1:length(extractedCharacters)
%     charImage = extractedCharacters{i};
%     charImage = imresize(charImage, [32 32]);
%     charFeatures = extractFeaturesUsingRawPixels(charImage);
%     neuronIndex = vec2ind(somNet(charFeatures'));
%     predictedLabel = neuronLabels(neuronIndex);
%     characterPredictions{i} = predictedLabel;
%     fprintf('Character %d classification result：%s\n', i, predictedLabel);
% end
% 
% % Local function definition
% function features = extractFeaturesFromImages(imageSet, featureExtractor)
%     numImages = numel(imageSet.Files);
%     features = [];
%     for i = 1:numImages
%         img = readimage(imageSet, i);
%         imgFeatures = featureExtractor(img);
%         features = [features; imgFeatures];
%     end
% end
% 
% function features = extractFeaturesUsingRawPixels(image)
%     % If the image is not binary, resize and convert to grayscale image
%     if ~islogical(image)
%         resizedImage = imresize(image, [32 32]);
%         if size(resizedImage, 3) == 3
%             grayImage = rgb2gray(resizedImage);
%         else
%             grayImage = resizedImage;
%         end
%         features = double(grayImage(:)');
%     else
%         % If the image is already binary, resize it directly
%         resizedImage = imresize(image, [32 32]);
%         features = double(resizedImage(:)');
%     end
% end


% % Histogram
% % Main script part
% % Step 1: Load the dataset
% % Load the dataset
% dataFolderPath = 'D:\Graduate\5405\p_dataset_26\p_dataset_26';
% imds = imageDatastore(dataFolderPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% 
% % Step 2: Split the dataset
% % Divide the dataset
% [trainingSet, testSet] = splitEachLabel(imds, 0.75, 'randomize');
% 
% % Step 3: Feature extraction (using grayscale histogram method)
% % Feature extraction (using grayscale histogram method)
% % (Create and train SOM network first to determine expectedInputSize)
% 
% % Step 4: Train SOM classifier
% % Train SOM classifier
% somNet = selforgmap([8 8]); % Create an 8x8 SOM network
% somNet = train(somNet, trainingFeatures');
% expectedInputSize = size(somNet.IW{1,1},2); % Determine the input size of SOM network
% 
% % Set the number of bins before extracting features
% numBins = expectedInputSize; % Set the number of bins based on the input size of SOM network
% trainingFeatures = extractFeaturesFromImages(trainingSet, @extractHistogramFeatures, expectedInputSize);
% testFeatures = extractFeaturesFromImages(testSet, @extractHistogramFeatures, expectedInputSize);
% 
% % Step 5: Evaluate the classifier
% % Evaluate the classifier
% % Transpose predictedLabels into a column vector
% predictedLabels = predictedLabels';
% 
% % Now the sizes of predictedLabels and trueLabels should match
% % Calculate accuracy
% accuracy = sum(predictedLabels == trueLabels) / numel(trueLabels);
% disp(['Accuracy: ', num2str(accuracy)]);
% 
% % Create a cell array that stores the extracted character images
% extractedCharacters = cell(0);
% 
% % Initialize a counter for numeric labels
% labelCounter = 1;
% binaryImageInverted = ~binaryImage;
% 
% for i = 1:length(stats)
%     % Get the bounding box of the current character
%     bbox = stats(i).BoundingBox;
% 
%     % Round the coordinates of the bounding box to the nearest integer value
%     x = round(bbox(1));
%     y = round(bbox(2));
%     width = round(bbox(3));
%     height = round(bbox(4));
% 
%     % Check if the width of the bounding box is greater than the height
%     if width / height > 1
%         % The width is greater than the height, and the divided area is two characters
% 
%         % Compute new two bounding boxes bbox1 and bbox2
%         halfWidth = width / 2;
%         bbox1 = [x, y, halfWidth, height];
%         bbox2 = [x + halfWidth, y, halfWidth, height];
% 
%         % Use imcrop function to extract character parts from binary images and store them separately
%         charImage1 = imcrop(binaryImageInverted, bbox1);
%         charImage2 = imcrop(binaryImageInverted, bbox2);
% 
%         % Add extracted character image to cell array and assign new numeric label
%         extractedCharacters{end+1} = charImage1;
%         extractedCharacters{end+1} = charImage2;
% 
%         % Draw red bounding boxes and numerical labels to identify the two characters
%         hold on;
%         rectangle('Position', bbox1, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox1(1) + 5, bbox1(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         rectangle('Position', bbox2, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox2(1) + 5, bbox2(2) - 5, num2str(labelCounter + 1), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter to assign a new numeric label to the next character
%         labelCounter = labelCounter + 2;
%     else
%         % Width not greater than height, processing single characters
% 
%         % Use imcrop function to extract character parts from binary images
%         charImage = imcrop(binaryImageInverted, [x, y, width, height]);
% 
%         % Add extracted character image to cell array
%         extractedCharacters{end+1} = charImage;
% 
%         % Draw red bounding boxes and numerical labels to identify individual characters
%         hold on;
%         rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox(1) + 5, bbox(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter to assign a new numeric label to the next character
%         labelCounter = labelCounter + 1;
%     end
% end
% 
% % Display or save extracted character image (optional)
% characterPredictions = cell(size(extractedCharacters));
% for i = 1:length(extractedCharacters)
%     charImage = extractedCharacters{i};
%     charImage = imresize(charImage, [32 32]); % Adjust image size
%     charFeatures = extractHistogramFeatures(charImage, numBins);
% 
%     % Ensure the feature vector size matches the SOM network's input size
%     if numel(charFeatures) ~= expectedSize
%         error('Feature size does not match the SOM network’s input size.');
%     end
% 
%     neuronIndex = vec2ind(somNet(charFeatures')); % Note the transpose of the feature vector
%     predictedLabel = neuronLabels(neuronIndex);
%     characterPredictions{i} = predictedLabel;
%     fprintf('Character %d classification result: %s\n', i, predictedLabel);
% end
% 
% % Local function definitions
% function features = extractFeaturesFromImages(imageSet, featureExtractor, numBins)
%     numImages = numel(imageSet.Files);
%     features = [];
%     for i = 1:numImages
%         img = readimage(imageSet, i);
%         imgFeatures = featureExtractor(img, numBins);
%         features = [features; imgFeatures'];
%     end
% end
% 
% function features = extractHistogramFeatures(image, expectedInputSize)
%     % Convert image to grayscale if it is in color
%     if size(image, 3) == 3
%         grayImage = rgb2gray(image);
%     else
%         grayImage = image;
%     end
% 
%     % Check if the image is logical (binary); if so, use 2 bins for histogram
%     if islogical(grayImage)
%         histogram = imhist(grayImage, 2)';
%     else
%         histogram = imhist(grayImage, 256)'; % Use more bins for non-binary images
%     end
% 
%     histogram = histogram / sum(histogram); % Normalize the histogram
% 
%     % Adjust feature vector to match expected input size of SOM network
%     if numel(histogram) > expectedInputSize
%         features = histogram(1:expectedInputSize); % Truncate the feature vector
%     elseif numel(histogram) < expectedInputSize
%         features = [histogram, zeros(1, expectedInputSize - numel(histogram))]; % Extend the feature vector
%     else
%         features = histogram;
%     end
% end


% % Edge
% % Main script part
% % Step 1: Load the dataset
% dataFolderPath = 'D:\Graduate\5405\p_dataset_26\p_dataset_26';
% imds = imageDatastore(dataFolderPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% 
% % Step 2: Split the dataset
% [trainingSet, testSet] = splitEachLabel(imds, 0.75, 'randomize');
% 
% % Step 3: Feature extraction (using edge detection method)
% % (First, create and train the SOM network to determine expectedInputSize)
% 
% % Step 4: Train SOM classifier
% somNet = selforgmap([8 8]); % Create an 8x8 SOM network
% somNet = train(somNet, trainingFeatures');
% expectedInputSize = size(somNet.IW{1,1},2); % Determine the input size of SOM network
% 
% % Set the number of bins before extracting features
% numBins = expectedInputSize; 
% trainingFeatures = extractFeaturesFromImages(trainingSet, @extractEdgeFeatures, expectedInputSize);
% testFeatures = extractFeaturesFromImages(testSet, @extractEdgeFeatures, expectedInputSize);
% 
% % Step 5: Evaluate the classifier
% predictedLabels = vec2ind(somNet(testFeatures'));
% predictedLabels = predictedLabels'; % Transpose predictedLabels into a column vector
% 
% % Now the sizes of predictedLabels and trueLabels should match
% accuracy = sum(predictedLabels == trueLabels) / numel(trueLabels);
% disp(['Accuracy: ', num2str(accuracy)]);
% 
% % Create a cell array to store the extracted character images
% extractedCharacters = cell(0);
% 
% % Initialize a counter for numeric labels
% labelCounter = 1;
% binaryImageInverted = ~binaryImage;
% 
% for i = 1:length(stats)
%     % Get the bounding box of the current character
%     bbox = stats(i).BoundingBox;
% 
%     % Round the coordinates of the bounding box to the nearest integer value
%     x = round(bbox(1));
%     y = round(bbox(2));
%     width = round(bbox(3));
%     height = round(bbox(4));
% 
%     % Check if the width of the bounding box is greater than the height
%     if width / height > 1
%         % The width is greater than the height, and the divided area is two characters
% 
%         % Compute new two bounding boxes bbox1 and bbox2
%         halfWidth = width / 2;
%         bbox1 = [x, y, halfWidth, height];
%         bbox2 = [x + halfWidth, y, halfWidth, height];
% 
%         % Use imcrop function to extract character parts from binary images and store them separately
%         charImage1 = imcrop(binaryImageInverted, bbox1);
%         charImage2 = imcrop(binaryImageInverted, bbox2);
% 
%         % Add extracted character image to cell array and assign new numeric label
%         extractedCharacters{end+1} = charImage1;
%         extractedCharacters{end+1} = charImage2;
% 
%         % Draw red bounding boxes and numerical labels to identify the two characters
%         hold on;
%         rectangle('Position', bbox1, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox1(1) + 5, bbox1(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         rectangle('Position', bbox2, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox2(1) + 5, bbox2(2) - 5, num2str(labelCounter + 1), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter to assign a new numeric label to the next character
%         labelCounter = labelCounter + 2;
%     else
%         % Width not greater than height, processing single characters
% 
%         % Use imcrop function to extract character parts from binary images
%         charImage = imcrop(binaryImageInverted, [x, y, width, height]);
% 
%         % Add extracted character image to cell array
%         extractedCharacters{end+1} = charImage;
% 
%         % Draw red bounding boxes and numerical labels to identify individual characters
%         hold on;
%         rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox(1) + 5, bbox(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter to assign a new numeric label to the next character
%         labelCounter = labelCounter + 1;
%     end
% end
% 
% % Display or save extracted character image (optional)
% characterPredictions = cell(size(extractedCharacters));
% for i = 1:length(extractedCharacters)
%     charImage = extractedCharacters{i};
%     charImage = imresize(charImage, [32 32]); % Adjust image size
%     charFeatures = extractEdgeFeatures(charImage, numBins);
% 
%     % Ensure the feature vector size matches the SOM network's input size
%     if numel(charFeatures) ~= expectedSize
%         error('Feature size does not match the SOM network’s input size.');
%     end
% 
%     neuronIndex = vec2ind(somNet(charFeatures')); % Note the transpose of the feature vector
%     predictedLabel = neuronLabels(neuronIndex);
%     characterPredictions{i} = predictedLabel;
%     fprintf('Character %d classification result: %s\n', i, predictedLabel);
% end
% 
% % Local function definitions
% function features = extractFeaturesFromImages(imageSet, featureExtractor, expectedInputSize)
%     numImages = numel(imageSet.Files);
%     features = [];
%     for i = 1:numImages
%         img = readimage(imageSet, i);
%         imgFeatures = featureExtractor(img, expectedInputSize);
%         features = [features; imgFeatures'];
%     end
% end
% 
% function features = extractEdgeFeatures(image, expectedInputSize)
%     % Convert image to grayscale if it is in color
%     if size(image, 3) == 3
%         image = rgb2gray(image);
%     end
% 
%     % Apply edge detection
%     edges = edge(image, 'Canny');
% 
%     % Flatten the edge matrix to a vector
%     edgeFeatures = edges(:)';
% 
%     % Adjust the feature vector length to match the SOM network's expected input size
%     featureLength = length(edgeFeatures);
%     if featureLength > expectedInputSize
%         features = edgeFeatures(1:expectedInputSize);
%     elseif featureLength < expectedInputSize
%         features = [edgeFeatures, zeros(1, expectedInputSize - featureLength)];
%     else
%         features = edgeFeatures;
%     end
% end

% % Gabor
% % Main script part
% % Step 1: Load the dataset
% dataFolderPath = 'D:\Graduate\5405\p_dataset_26\p_dataset_26';
% imds = imageDatastore(dataFolderPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% 
% % Step 2: Split the dataset
% [trainingSet, testSet] = splitEachLabel(imds, 0.75, 'randomize');
% 
% % Step 3: Feature extraction (using edge detection method)
% % (First, create and train the SOM network to determine expectedInputSize)
% 
% % Step 4: Train SOM classifier
% somNet = selforgmap([8 8]); % Create an 8x8 SOM network
% somNet = train(somNet, trainingFeatures');
% expectedInputSize = size(somNet.IW{1,1},2); % Determine the input size of the SOM network
% 
% % Set the number of bins before extracting features
% numBins = expectedInputSize; 
% % Use Gabor feature extraction method
% trainingFeatures = extractFeaturesFromImages(trainingSet, @extractGaborFeatures, expectedInputSize);
% testFeatures = extractFeaturesFromImages(testSet, @extractGaborFeatures, expectedInputSize);
% 
% % Step 5: Evaluate the classifier
% predictedLabels = vec2ind(somNet(testFeatures'));
% predictedLabels = predictedLabels'; % Transpose predictedLabels into a column vector
% 
% % Now the sizes of predictedLabels and trueLabels should match
% accuracy = sum(predictedLabels == trueLabels) / numel(trueLabels);
% disp(['Accuracy: ', num2str(accuracy)]);
% 
% % Create a cell array to store the extracted character images
% extractedCharacters = cell(0);
% 
% % Initialize a counter for numeric labels
% labelCounter = 1;
% binaryImageInverted = ~binaryImage;
% 
% for i = 1:length(stats)
%     % Get the bounding box of the current character
%     bbox = stats(i).BoundingBox;
% 
%     % Round the coordinates of the bounding box to the nearest integer value
%     x = round(bbox(1));
%     y = round(bbox(2));
%     width = round(bbox(3));
%     height = round(bbox(4));
% 
%     % Check if the width of the bounding box is greater than the height
%     if width / height > 1
%         % The width is greater than the height, and the divided area is two characters
% 
%         % Compute new two bounding boxes bbox1 and bbox2
%         halfWidth = width / 2;
%         bbox1 = [x, y, halfWidth, height];
%         bbox2 = [x + halfWidth, y, halfWidth, height];
% 
%         % Use imcrop function to extract character parts from binary images and store them separately
%         charImage1 = imcrop(binaryImageInverted, bbox1);
%         charImage2 = imcrop(binaryImageInverted, bbox2);
% 
%         % Add extracted character image to cell array and assign new numeric label
%         extractedCharacters{end+1} = charImage1;
%         extractedCharacters{end+1} = charImage2;
% 
%         % Draw red bounding boxes and numerical labels to identify the two characters
%         hold on;
%         rectangle('Position', bbox1, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox1(1) + 5, bbox1(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         rectangle('Position', bbox2, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox2(1) + 5, bbox2(2) - 5, num2str(labelCounter + 1), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter to assign a new numeric label to the next character
%         labelCounter = labelCounter + 2;
%     else
%         % Width not greater than height, processing single characters
% 
%         % Use imcrop function to extract character parts from binary images
%         charImage = imcrop(binaryImageInverted, [x, y, width, height]);
% 
%         % Add extracted character image to cell array
%         extractedCharacters{end+1} = charImage;
% 
%         % Draw red bounding boxes and numerical labels to identify individual characters
%         hold on;
%         rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox(1) + 5, bbox(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter to assign a new numeric label to the next character
%         labelCounter = labelCounter + 1;
%     end
% end
% 
% % Display or save extracted character image (optional)
% characterPredictions = cell(size(extractedCharacters));
% for i = 1:length(extractedCharacters)
%     charImage = extractedCharacters{i};
%     charImage = imresize(charImage, [32 32]); % Adjust image size
%     charFeatures = extractGaborFeatures(charImage, numBins);
% 
%     % Ensure the feature vector size matches the SOM network's input size
%     if numel(charFeatures) ~= expectedSize
%         error('Feature size does not match the SOM network’s input size.');
%     end
% 
%     neuronIndex = vec2ind(somNet(charFeatures')); % Note the transpose of the feature vector
%     predictedLabel = neuronLabels(neuronIndex);
%     characterPredictions{i} = predictedLabel;
%     fprintf('Character %d classification result: %s\n', i, predictedLabel);
% end
% 
% % Local function definitions
% function features = extractFeaturesFromImages(imageSet, featureExtractor, expectedInputSize)
%     numImages = numel(imageSet.Files);
%     features = [];
%     for i = 1:numImages
%         img = readimage(imageSet, i);
%         imgFeatures = featureExtractor(img, expectedInputSize);
%         features = [features; imgFeatures'];
%     end
% end
% 
% function features = extractGaborFeatures(image, expectedInputSize)
%     % Convert image to grayscale if it is in color
%     if size(image, 3) == 3
%         image = rgb2gray(image);
%     end
% 
%     % Define Gabor filter parameters
%     wavelength = 2.^(0:5) * 3; % Wavelength
%     orientation = 0:45:135;  % Orientation
%     gaborArray = gabor(wavelength, orientation);
% 
%     % Apply Gabor filter
%     gaborMag = imgaborfilt(image, gaborArray);
% 
%     % Extract and organize features
%     numFilters = length(gaborArray);
%     gaborFeatures = zeros(1, numFilters);
%     for i = 1:numFilters
%         gaborFeatures(i) = mean2(gaborMag(:,:,i));
%     end
% 
%     % Adjust the feature vector length to match the expected input size of the SOM network
%     featureLength = length(gaborFeatures);
%     if featureLength > expectedInputSize
%         features = gaborFeatures(1:expectedInputSize);
%     elseif featureLength < expectedInputSize
%         features = [gaborFeatures, zeros(1, expectedInputSize - featureLength)];
%     else
%         features = gaborFeatures;
%     end
% end


% % Hough
% % Main script part
% % Step 1: Load the dataset
% dataFolderPath = 'D:\Graduate\5405\p_dataset_26\p_dataset_26';
% imds = imageDatastore(dataFolderPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% 
% % Step 2: Split the dataset
% [trainingSet, testSet] = splitEachLabel(imds, 0.75, 'randomize');
% 
% % Step 3: Feature extraction (using edge detection method)
% % (First, create and train the SOM network to determine expectedInputSize)
% 
% % Step 4: Train SOM classifier
% somNet = selforgmap([8 8]); % Create an 8x8 SOM network
% somNet = train(somNet, trainingFeatures');
% expectedInputSize = size(somNet.IW{1,1},2); % Determine the input size of the SOM network
% 
% % Set the number of bins before extracting features
% numBins = expectedInputSize; 
% % Use Hough Transform feature extraction method
% trainingFeatures = extractFeaturesFromImages(trainingSet, @extractHoughFeatures, expectedInputSize);
% testFeatures = extractFeaturesFromImages(testSet, @extractHoughFeatures, expectedInputSize);
% 
% % Step 5: Evaluate the classifier
% predictedLabels = vec2ind(somNet(testFeatures'));
% predictedLabels = predictedLabels'; % Transpose predictedLabels into a column vector
% 
% % Now the sizes of predictedLabels and trueLabels should match
% accuracy = sum(predictedLabels == trueLabels) / numel(trueLabels);
% disp(['Accuracy: ', num2str(accuracy)]);
% 
% % Create a cell array to store the extracted character images
% extractedCharacters = cell(0);
% 
% % Initialize a counter for numeric labels
% labelCounter = 1;
% binaryImageInverted = ~binaryImage;
% 
% for i = 1:length(stats)
%     % Get the bounding box of the current character
%     bbox = stats(i).BoundingBox;
% 
%     % Round the coordinates of the bounding box to the nearest integer value
%     x = round(bbox(1));
%     y = round(bbox(2));
%     width = round(bbox(3));
%     height = round(bbox(4));
% 
%     % Check if the width of the bounding box is greater than the height
%     if width / height > 1
%         % The width is greater than the height, and the divided area is two characters
% 
%         % Compute new two bounding boxes bbox1 and bbox2
%         halfWidth = width / 2;
%         bbox1 = [x, y, halfWidth, height];
%         bbox2 = [x + halfWidth, y, halfWidth, height];
% 
%         % Use imcrop function to extract character parts from binary images and store them separately
%         charImage1 = imcrop(binaryImageInverted, bbox1);
%         charImage2 = imcrop(binaryImageInverted, bbox2);
% 
%         % Add extracted character image to cell array and assign new numeric label
%         extractedCharacters{end+1} = charImage1;
%         extractedCharacters{end+1} = charImage2;
% 
%         % Draw red bounding boxes and numerical labels to identify the two characters
%         hold on;
%         rectangle('Position', bbox1, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox1(1) + 5, bbox1(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         rectangle('Position', bbox2, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox2(1) + 5, bbox2(2) - 5, num2str(labelCounter + 1), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter to assign a new numeric label to the next character
%         labelCounter = labelCounter + 2;
%     else
%         % Width not greater than height, processing single characters
% 
%         % Use imcrop function to extract character parts from binary images
%         charImage = imcrop(binaryImageInverted, [x, y, width, height]);
% 
%         % Add extracted character image to cell array
%         extractedCharacters{end+1} = charImage;
% 
%         % Draw red bounding boxes and numerical labels to identify individual characters
%         hold on;
%         rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
%         text(bbox(1) + 5, bbox(2) - 5, num2str(labelCounter), 'Color', 'r', 'FontSize', 12);
%         hold off;
% 
%         % Increment labelCounter to assign a new numeric label to the next character
%         labelCounter = labelCounter + 1;
%     end
% end
% 
% % Display or save extracted character image (optional)
% characterPredictions = cell(size(extractedCharacters));
% for i = 1:length(extractedCharacters)
%     charImage = extractedCharacters{i};
%     charImage = imresize(charImage, [32 32]); % Adjust image size
%     charFeatures = extractHoughFeatures(charImage, numBins);
% 
%     % Ensure the feature vector size matches the SOM network's input size
%     if numel(charFeatures) ~= expectedSize
%         error('Feature size does not match the SOM network’s input size.');
%     end
% 
%     neuronIndex = vec2ind(somNet(charFeatures')); % Note the transpose of the feature vector
%     predictedLabel = neuronLabels(neuronIndex);
%     characterPredictions{i} = predictedLabel;
%     fprintf('Character %d classification result: %s\n', i, predictedLabel);
% end
% 
% % Local function definitions
% function features = extractFeaturesFromImages(imageSet, featureExtractor, expectedInputSize)
%     numImages = numel(imageSet.Files);
%     features = [];
%     for i = 1:numImages
%         img = readimage(imageSet, i);
%         imgFeatures = featureExtractor(img, expectedInputSize);
%         features = [features; imgFeatures'];
%     end
% end
% 
% function features = extractHoughFeatures(image, expectedInputSize)
%     % Convert image to grayscale if it is in color
%     if size(image, 3) == 3
%         image = rgb2gray(image);
%     end
% 
%     % Apply edge detection
%     edges = edge(image, 'Canny');
% 
%     % Apply Hough Transform
%     [H, theta, rho] = hough(edges);
% 
%     % Extract features from the Hough Transform (statistical features of H)
%     houghFeatures = [mean2(H), std2(H)];
% 
%     % Adjust the feature vector length to match the expected input size of the SOM network
%     featureLength = length(houghFeatures);
%     if featureLength > expectedInputSize
%         features = houghFeatures(1:expectedInputSize);
%     elseif featureLength < expectedInputSize
%         features = [houghFeatures, zeros(1, expectedInputSize - featureLength)];
%     else
%         features = houghFeatures;
%     end
% end




