function showExtraFeatures(trainedNet, testImage)
% Enhanced Explainable Fake Image Detection Features
% Fixed for MATLAB compatibility

    if isempty(testImage)
        error('No image provided');
    end

    % Resize image
    testImage = imresize(testImage, [224 224]);
    if size(testImage, 3) == 1
        testImage = cat(3, testImage, testImage, testImage);
    end

    % Create nice figure
    figure('Name', 'Explainable Fake Image Detection Analysis', ...
           'Position', [150 100 1400 700], 'Color', [0.95 0.95 0.98]);

    % 1. Original Image
    subplot(2, 3, 1);
    imshow(testImage);
    title('1. Original Image', 'FontSize', 14, 'FontWeight', 'bold');

    % 2. Error Level Analysis (ELA)
    try
        tempfile = 'temp_ela.jpg';
        imwrite(testImage, tempfile, 'jpeg', 'Quality', 90);
        compressed = imread(tempfile);
        delete(tempfile);
        ela = uint8(abs(double(testImage) - double(compressed)) * 12);
        
        subplot(2, 3, 2);
        imshow(ela, []);
        title('2. Error Level Analysis (ELA)', 'FontSize', 14, 'FontWeight', 'bold');
        xlabel('Bright areas = likely fake');
    catch
        subplot(2, 3, 2);
        text(0.5, 0.5, 'ELA not available', 'HorizontalAlignment', 'center');
    end

    % 3. Frequency Analysis (FFT)
    try
        gray = rgb2gray(testImage);
        F = fftshift(fft2(gray));
        fftMag = log(abs(F) + 1);
        
        subplot(2, 3, 3);
        imshow(fftMag, []);
        title('3. Frequency Analysis (FFT)', 'FontSize', 14, 'FontWeight', 'bold');
    catch
        subplot(2, 3, 3);
        text(0.5, 0.5, 'FFT not available', 'HorizontalAlignment', 'center');
    end

    % 4. Prediction with Confidence
    predictedLabel = classify(trainedNet, testImage);
    scores = predict(trainedNet, testImage);
    [~, idx] = max(scores);
    confidence = scores(idx) * 100;
    
    subplot(2, 3, 4);
    imshow(testImage);
    if strcmp(char(predictedLabel), 'FAKE')
        titleColor = 'r';
    else
        titleColor = 'g';
    end
    title(sprintf('4. Prediction: %s\nConfidence: %.1f%%', ...
        char(predictedLabel), confidence), ...
        'FontSize', 14, 'FontWeight', 'bold', 'Color', titleColor);

    % 5. Grad-CAM
    try
        scoreMap = gradCAM(trainedNet, testImage, predictedLabel, ...
            'FeatureLayer', 'activation_49_relu');
        
        subplot(2, 3, 5);
        imshow(testImage);
        hold on;
        imagesc(scoreMap, 'AlphaData', 0.65);
        colormap('jet');
        colorbar;
        title('5. Grad-CAM: AI Attention Map', 'FontSize', 14, 'FontWeight', 'bold');
        xlabel('Red = regions model focused on');
        hold off;
    catch
        subplot(2, 3, 5);
        text(0.5, 0.5, 'Grad-CAM not available\nTry after full training', ...
            'HorizontalAlignment', 'center');
    end

    sgtitle('Hybrid Fake Image Detection - Explainable Features', ...
            'FontSize', 16, 'FontWeight', 'bold');

end