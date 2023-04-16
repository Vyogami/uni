% Read the image
img = imread('screws.tif');

% Plot the histogram
histogram(img);

% Determine threshold
threshold = 100;

% Threshold the image
thresholded_img = img > threshold;

% Display original and thresholded image
subplot(1,2,1), imshow(img), title('Original Image');
subplot(1,2,2), imshow(thresholded_img), title('Thresholded Image');
