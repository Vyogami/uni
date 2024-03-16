% Read the color image
img = imread('balloons.png');

% Plot the red, green, and blue channels on separate plots using subplot
subplot(2, 2, 1);
imshow(img);
title('Original Image');

subplot(2, 2, 2);
imshow(img(:, :, 1));
title('Red Channel');

subplot(2, 2, 3);
imshow(img(:, :, 2));
title('Green Channel');

subplot(2, 2, 4);
imshow(img(:, :, 3));
title('Blue Channel');

% Select the green channel for thresholding and plot its histogram
green_channel = img(:, :, 2);
figure;
imhist(green_channel);
title('Green Channel Histogram');

% Threshold the image using the green channel and convert it to a binary image
threshold = 150;
binary_img = green_channel > threshold;

% Display the binary image
figure;
imshow(binary_img);
title('Binary Image');
