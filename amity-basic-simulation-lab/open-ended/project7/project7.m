% Read the grayscale image
img = imread('circuit.tif');

% Display the original image
subplot(2, 2, 1);
imshow(img);
title('Original Image');

% Add Gaussian noise to the image
noisy_img_gaussian = imnoise(img, 'gaussian', 0.05);
% Display the noisy image with Gaussian noise
subplot(2, 2, 2);
imshow(noisy_img_gaussian);
title('Noisy Image (Gaussian)');

% Add salt and pepper noise to the image
noisy_img_sp = imnoise(img, 'salt & pepper', 0.05);
% Display the noisy image with salt and pepper noise
subplot(2, 2, 3);
imshow(noisy_img_sp);
title('Noisy Image (Salt and Pepper)');

% Apply Gaussian filter to the noisy image
filtered_img_gaussian = imgaussfilt(noisy_img_gaussian, 2);
% Display the filtered image with Gaussian filter
subplot(2, 2, 4);
imshow(filtered_img_gaussian);
title('Filtered Image (Gaussian)');

