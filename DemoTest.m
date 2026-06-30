 


clc; clear; close all;

disp('=== Fake Image Detector Demo ===');
disp('Accuracy: 97.17% (4 epochs trained)');

%% Load the Final Model
load('myFakeDetector_4Epochs_Final.mat', 'trainedNet');
disp('✅ Model loaded successfully.');

%% Reload Dataset (Important - fixes imdsTest error)
imds = imageDatastore('dataset', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[imdsTrain, imdsVal, imdsTest] = splitEachLabel(imds, 0.7, 0.15, 0.15, 'randomized');

disp('Dataset reloaded.');

while true
    choice = questdlg('Choose image for demo:', 'Demo Testing', ...
        'Test Set Image (Safest)', 'Personal Photo', 'Exit');

    if strcmp(choice, 'Exit') || isempty(choice)
        break;
    end

    if strcmp(choice, 'Test Set Image (Safest)')
        testIdx = randi(numel(imdsTest.Files));
        testImage = imread(imdsTest.Files{testIdx});
        source = 'Test Set Image';
    else
        [filename, pathname] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp','Image Files'}, 'Select Image');
        if filename == 0, continue; end
        testImage = imread(fullfile(pathname, filename));
        source = filename;
    end

    % Analyze the image
    inputImg = imresize(testImage, [224 224]);
    predictedLabel = classify(trainedNet, inputImg);
    confidence = max(predict(trainedNet, inputImg)) * 100;

    % Show Result
    figure('Name', 'Detection Result', 'Position', [300 150 950 650]);
    imshow(testImage);
    
    if strcmp(char(predictedLabel), 'FAKE')
        titleColor = 'r';
        resultText = 'FAKE (AI-Generated)';
    else
        titleColor = 'g';
        resultText = 'REAL Photograph';
    end
    
    title(sprintf('%s\nConfidence: %.1f%%', resultText, confidence), ...
        'FontSize', 20, 'FontWeight', 'bold', 'Color', titleColor);
    
    sgtitle(['Source: ' source ' | Accuracy: 97.17%'], 'FontSize', 14);

    % Show explainable features
    if questdlg('Show explainable features (ELA, FFT, Grad-CAM)?', 'Demo', 'Yes', 'No', 'Yes') == "Yes"
        showExtraFeatures(trainedNet, testImage);
    end
end

disp('Demo testing completed.');