# Image Thresholding in MATLAB/Octave

<p>Read a gray scale image in Matlab/Octave. </p><p>(Use screws.tif from <https://people.math.sc.edu/Burkardt/data/tif/tif.html>, and available in Files Tab of this team)</p><p>a) Plot its histogram and determine a suitable value for threshold.  </p><p>b) Display the original and thresholded image.</p><p></p>

## Problem Definition

This MATLAB/Octave code demonstrates image thresholding, a common image processing technique used to separate foreground and background pixels in an image. Specifically, the code reads in a grayscale image, plots its histogram, determines a suitable threshold value, and then applies thresholding to convert the image to a binary image.

The problem to be solved is to segment an image into foreground and background regions. Image thresholding is used to accomplish this task.

## Technical Review

Image thresholding is a technique used in image processing to separate pixels into two classes based on their intensity values. A threshold is set to divide the pixel intensity values into two regions, and any pixel with an intensity value above the threshold is classified as foreground (or object) and any pixel with an intensity value below the threshold is classified as background.

## Design Requirements

The requirements for this project are to:

- Read a grayscale image
- Plot its histogram
- Determine a suitable threshold value
- Apply thresholding to convert the image to a binary image
- Display the original and thresholded images

## Design Description

### Overview

The code reads in a grayscale image using the `imread` function. It then plots the histogram of the image using the `histogram` function to visualize the distribution of pixel intensities. Based on the histogram, a suitable threshold value is determined. Finally, the thresholding operation is applied to the image using the `>` operator, which compares each pixel's intensity value to the threshold and sets its value to either 0 or 1.

### Detailed Description

The following steps are followed in the code:

1. Read in the grayscale image using `imread` function
2. Plot the histogram of the image using `histogram` function
3. Determine a suitable threshold value
4. Apply thresholding to convert the image to a binary image using the `>` operator
5. Display the original and thresholded images side by side using `subplot` and `imshow` functions

### Use

To use this code, simply run the script in MATLAB or Octave after downloading the `screws.tif` file from [here](https://people.math.sc.edu/Burkardt/data/tif/tif.html) and placing it in the same directory as the script.

## Evaluation

### Overview

The code was evaluated by visually inspecting the original and thresholded images and comparing them to ensure the foreground and background regions were accurately separated.

### Prototype

The prototype consisted of the MATLAB/Octave script that read in the image and performed the thresholding operation.

### Testing and Results

The code was tested using the `screws.tif` file, and the threshold value was determined to be 100 based on the histogram. The original and thresholded images were displayed side by side using `subplot` and `imshow` functions.

### Assessment

The code was successful in separating the foreground and background regions of the image using thresholding. However, the threshold value was determined manually and may not be optimal for all images.

### Next Steps

Future improvements to the code could include automatically determining the threshold value using techniques such as Otsu's method or adaptive thresholding.
