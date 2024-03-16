# Image Processing Project

<p>Read a gray scale image in Matlab/Octave. </p><p>a) Add noise (Gaussian, Salt and Pepper noise) to the image.  </p><p>b) Use filters (Gaussian, Median filters) to reduce noise in the image.</p><p>c) Display the original, noisy and smoothened images.</p><p></p>

## Problem Definition

The aim of this project is to demonstrate how to read a grayscale image in Matlab/Octave, and then add noise (Gaussian and Salt and Pepper noise) to the image. The project will also show how to use filters (Gaussian and Median filters) to reduce noise in the image. Finally, the original, noisy, and smoothened images will be displayed.

## Problem Scope

This project will require the Image Processing Toolbox in Matlab/Octave. The grayscale image "circuit.tif" will be used, which is available in the "Files" tab of this team.

## Technical Review

The project will involve the following technical steps:

1. Read the grayscale image using imread()
2. Add Gaussian and Salt and Pepper noise using imnoise()
3. Apply Gaussian and Median filters to reduce the noise using imgaussfilt() and medfilt2()
4. Display the original, noisy, and smoothened images using imshow()

## Design Requirements

The following design requirements must be met:

1. The code must be written in Matlab/Octave
2. The Image Processing Toolbox must be installed
3. The grayscale image "circuit.tif" must be used
4. The noisy image must have Gaussian and Salt and Pepper noise with a noise level of 0.05
5. The Gaussian filter must have a standard deviation of 2
6. The Median filter must have a window size of 3x3

## Design Description

### Overview

The program reads a grayscale image using Matlab/Octave's `imread()` function. The image is then subjected to two types of noise: Gaussian noise and salt and pepper noise using the `imnoise()` function. The program then applies two filters to reduce the noise in the image: a Gaussian filter and a median filter. The filtered images are displayed using the `imshow()` function.

### Detailed description

1. The image is read using the `imread()` function.

2. Gaussian noise is added to the image using the `imnoise()` function. The amount of noise added is controlled by the third argument passed to the function.

3. Salt and pepper noise is added to the image using the `imnoise()` function. The amount of noise added is controlled by the third argument passed to the function.

4. A Gaussian filter is applied to the noisy image using the `imgaussfilt()` function. The standard deviation of the filter is set to 2.

5. A median filter is applied to the noisy image using the `medfilt2()` function. The size of the filter is set to 3-by-3.

6. The original image, noisy images, and filtered images are displayed using the `imshow()` function.

### Use

To use this program, you need to have Matlab or Octave installed on your computer and have access to the `Image Processing Toolbox`. You can run the program by executing the `project7.m` script in Matlab or Octave. The output will be displayed in the form of four images: the original image, noisy image with Gaussian noise, noisy image with salt and pepper noise, and the filtered image with Gaussian filter.

## Evaluation

### Overview

The program was tested on a grayscale image and the results were evaluated.

### Prototype

The program was run using the `circuit.tif` image provided in the Files Tab of this team. The program successfully added Gaussian and salt and pepper noise to the image and applied Gaussian and median filters to reduce the noise in the image. The original image, noisy images, and filtered images were displayed correctly.

### Testing and results

The noisy images showed clear visible noise, and the filtered images showed a significant reduction in noise. The Gaussian filter performed better than the median filter in reducing the Gaussian noise. The median filter performed better than the Gaussian filter in reducing the salt and pepper noise. Overall, both filters improved the image quality significantly.

### Assessment

The program successfully implemented the desired functionality of adding noise and applying filters to reduce the noise in the image.

### Next Steps

The program could be improved by allowing the user to input the amount of noise to be added to the image and the filter size. Additionally, the program could be expanded to work with color images.
