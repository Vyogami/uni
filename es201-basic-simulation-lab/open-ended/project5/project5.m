% Read the image
img = imread('cameraman.tif');

% Get the negative of the image
neg_img = 255 - img;

% Brighten the image
bright_img = img + 50;

% Darken the image
dark_img = img - 50;

% Plot the original image, its negative, and the brightened/darkened images
subplot(2, 2, 1);
imshow(img);
title('Original Image');

subplot(2, 2, 2);
imshow(neg_img);
title('Negative Image');

subplot(2, 2, 3);
imshow(bright_img);
title('Brightened Image');

subplot(2, 2, 4);
imshow(dark_img);
title('Darkened Image');
