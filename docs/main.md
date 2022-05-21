# Description of Models and Algorithms

## Convert images to signal
First step is preprocessing. An image is auto-rotated so that grid lines are orthogonal to the edges of the image. The algorithm searches for all lines in the image and calculates the angle of their inclination to rotate the picture. Then the RGB-format image is converted to grayscale for further work.

Second step is grid detection. Image is binarized and grid lines are separated from the signal using adaptive thresholding. After that, the algorithm evaluates the scale with the pixel-counting approach - calculating all spacings between grid lines. The scale is used to convert the signal from units of pixels to millimeters.

Third step is signal detection. The grayscale image is binarized using adaptive thresholding with Otsu's method. Then the binarized image is filtered with the median filter to remove noise (separate located pixels).

The final step is signal extraction - the list of all connected pixels of the signal is converted to the 1D array where pixels are scaled and represent amplitudes of the signal. 


## Detect risk markers
TBD

## Differensial diagnostics by risk markers
TBD

## Differensial diagnostics with a neural network
TBD