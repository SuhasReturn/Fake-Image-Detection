%% ================================================
%% Fresh Training - 4 Epochs with Auto Save
%% Clean & Safe Version for Overnight Training
%% ================================================

clc; clear; close all;

disp('=== Fresh Training - 4 Epochs with Auto-Save ===');
disp('Starting clean training...');

%% 1. Load Dataset
imds = imageDatastore('dataset', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[imdsTrain, imdsVal, imdsTest] = splitEachLabel(imds, 0.7, 0.15, 0.15, 'randomized');

augmenter = imageDataAugmenter('RandXReflection', true, 'RandRotation', [-20 20]);
augTrain = augmentedImageDatastore([224 224], imdsTrain, 'DataAugmentation', augmenter);
augVal   = augmentedImageDatastore([224 224], imdsVal);

disp(['Training images: ' num2str(numel(imdsTrain.Files))]);

%% 2. Build Fresh ResNet-50 Model
net = resnet50;
lgraph = layerGraph(net);

lgraph = replaceLayer(lgraph, 'fc1000', ...
    fullyConnectedLayer(2, 'Name', 'newFC', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10));

lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', classificationLayer('Name', 'newClass'));

disp('✅ Fresh model created.');

%% 3. Create Checkpoint Folder
checkpointPath = fullfile(pwd, 'Checkpoints_Fresh');
if ~exist(checkpointPath, 'dir')
    mkdir(checkpointPath);
end
disp(['✅ Checkpoints will be saved in: ' checkpointPath]);

%% 4. Training Options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 4, ...                    
    'MiniBatchSize', 8, ...                
    'ValidationData', augVal, ...
    'ValidationFrequency', 500, ...        
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 2, ...
    'ExecutionEnvironment', 'gpu', ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'CheckpointPath', checkpointPath);     % Auto-save after every epoch

disp('🚀 Starting fresh 4-epoch training...');
disp('Model will be automatically saved after each epoch.');

%% 5. Train
trainedNet = trainNetwork(augTrain, lgraph, options);

%% 6. Save Final Model
save('myFakeDetector_4Epochs_Final.mat', 'trainedNet');
disp('🎉 Training completed successfully!');
disp('Final model saved as myFakeDetector_4Epochs_Final.mat');

%% 7. Final Accuracy
[YPred, ~] = classify(trainedNet, augVal);
accuracy = mean(YPred == imdsVal.Labels) * 100;
disp(['Final Validation Accuracy: ' num2str(accuracy, '%.2f') '%']);