# Description of Models and Algorithms

## Convert images to signal
First step is preprocessing. An image is auto-rotated so that grid lines are orthogonal to the edges of the image. The algorithm searches for all lines in the image and calculates the angle of their inclination to rotate the picture. Then the RGB-format image is converted to grayscale for further work.

Second step is grid detection. Image is binarized and grid lines are separated from the signal using adaptive thresholding. After that, the algorithm evaluates the scale with the pixel-counting approach - calculating all spacings between grid lines. The scale is used to convert the signal from units of pixels to millimeters.

Third step is signal detection. The grayscale image is binarized using adaptive thresholding with Otsu's method. Then the binarized image is filtered with the median filter to remove noise (separate located pixels).

The final step is signal extraction - the list of all connected pixels of the signal is converted to the 1D array where pixels are scaled and represent amplitudes of the signal. 


## Detect risk markers
Risk markers include ST segment elevation in lead V3 (STE60) in millimeters, the corrected QT interval (QTc) in milliseconds and R-peak amplitude in lead V4 (RA) in millimeters. ST elevation and R-peak amplitude are measured from the PR segment, the P-wave offset point to be precise. ST elevation is measured at 60 ms after the J point (QRS offset). QTc is calculated using the Bazett formula [1]. All ECG features with the exception of the QRS onset and T-wave offset points are detected using the Neurokit2 library [2]. QRS onset points and T-wave offset points are detected using the gradient based approach [3].


[1] - BAZETT, H.C. (1997), AN ANALYSIS OF THE TIME-RELATIONS OF ELECTROCARDIOGRAMS.. 
Annals of Noninvasive Electrocardiology, 2: 177-194. https://doi.org/10.1111/j.1542-474X.1997.tb00325.x

[2] - Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H.,
Schölzel, C., & Chen, S. A. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing.
Behavior Research Methods, 53(4), 1689–1696. https://doi.org/10.3758/s13428-020-01516-y

[3] - Mazomenos, Evangelos & Chen, Taihai & Acharyya, Amit & Bhattacharya, A. & Rosengarten, James & Maharatna, Koushik. (2012). 
A Time-Domain Morphology and Gradient based algorithm for ECG feature extraction.
2012 IEEE International Conference on Industrial Technology, ICIT 2012, Proceedings. 10.1109/ICIT.2012.6209924. 

## Differensial diagnostics by risk markers
Diagnostic is performed via the modified criterion formula [1]. In the original paper if the value of the equation ([1.196 * ST segment elevation 60 ms after the J point in lead V3 in mm] + [0.059 * QTc in ms] – [0.326 * R-wave amplitude in lead V4 in mm]) is greater than 23.4 the diagnosis is STEMI and if less than or equal to 23.4, the diagnosis is early repolarization.

The original formula was modified by adding a maximal R-peak amplitude parameter and the coefficient values were optimized by a logistic regression model with SGD learning.

The modified formula is as follows: (2.9 * [STE60 V3 in mm]) + (0.3 * [QTc in ms]) + (-1.7 * np.minimum([RA V4 in mm], 19)) >= 126.9
 

[1] - Smith SW, Khalil A, Henry TD, Rosas M, Chang RJ, Heller K, Scharrer E, Ghorashi M, Pearce LA.
Electrocardiographic differentiation of early repolarization from subtle anterior ST-segment elevation myocardial infarction.
Ann Emerg Med. 2012 Jul;60(1):45-56.e2. doi: 10.1016/j.annemergmed.2012.02.015. Epub 2012 Apr 19. PMID: 22520989.

## Differensial diagnostics with a neural network
TBD