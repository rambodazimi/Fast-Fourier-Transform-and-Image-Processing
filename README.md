# Fast-Fourier-Transform-and-Image-Processing
Fast Fourier Transform and Image Processing

## Introduction
In this project you will implement two versions of the Discrete Fourier Transform (DFT). One of them will be a brute force approach that follows directly from the formula. The second one will be an implementation of the Fast Fourier Transform (FFT) which follows a divide-and-conquer approach to algorithm design. In particular, the FFT we will focus on is the Cooley-Tukey FFT.
At the end of this lab you should know how to:

Implement two types of DFT (naïve and FFT);

Understand why one version of the algorithm is better than the other;

Extend the FFT algorithm to multidimensional case (2-dimensions in this lab); Use FFT to denoise an image.

Use FFT to compress an image file.

## Background
The Fourier transform of a signal decomposes it into a sum of its various frequency components. However, in the real world a lot of the signals we deal with are either discrete or, even if they are continuous, can only be approximated via sampling which gives a discretized approximation of the original signal.

### The Cooley-Tukey FFT
In 1965 Cooley and Tukey came up with their own algorithm for Xk that takes advantage of the symmetry of the FT to simplify the algorithmic complexity. The algorithm fits into the divide-and- conquer design paradigm that you are familiar with from e.g. mergesort.

The proof works as follows:

• Split the sum in the even and odd indices which we sum separately and then put together.

• Factor out a constant by reducing the exponent by one in the second term. This makes the coefficients the same for both sums so we can have them saved and precomputed in an array.

• Observe that multiplying by 2 is the same as dividing by a half.

• Finally, notice that the two terms are just a DFT of the even and odd indices which can be computed as separate problems and then finally summed.

The algorithm works by successively splitting the problem in halves a shown above. Convince yourselves that this can be done as many times as we want by just applying the same technique to the even and odd halves to split the problem to 4-subproblems, then 8, then 16...

### 2D Fourier Transform
The 2D-DFT can be thought of as a composition of separate 1-dimensional DFTs. Looking at the equation we define the 2DDFT for a 2D sample of signal f with N rows and M columns.

## Project Description
Write a program using Python (version 3.5+, no Python 2) that:

• Performs DFT both with the naïve algorithm and the FFT algorithm discussed above.

• Performs the inverse operation. For the inverse just worry about the FFT implementation.

• Does the same thing for 2D Fourier Transforms (2d-FFT) and its inverse.

• Plots the resulting 2D DFT on a log scale plot.

• Saves the DFT on a csv or txt file.

For this lab you must use Python. You are not allowed to use any prebuilt FT algorithm, but you may use external libraries or functions that help you load and save data, manipulate images (cv2), plot (matplotlib or plotly) and carry out vectorized operations (numpy).

Your application should be named fft.py, and it should be invoked at the command line using the following syntax:

python fft.py [-m mode] [-i image]

where the argument is defined as follows:

mode (optional):

- [1] (Default) for fast mode where the image is converted into its FFT form and displayed

- [2] for denoising where the image is denoised by applying an FFT, truncating high frequencies and then displayed

- [3] for compressing and saving the image

- [4] for plotting the runtime graphs for the report

image (optional): filename of the image we wish to take the DFT of. (Default: the file name of the image given to you for the assignment)

## Output Behaviour
Your program’s output depends on the mode you are running.

For the first mode your program should simply perform the FFT and output a one by two subplot of the original image and next to it its Fourier transform. The Fourier transform should be log scaled. An easy way to do this with matplotlib is to import LogNorm from matplotlib.colors to produce a logarithmic colormap.

For the second mode you should also output a one by two subplot. In this subplot you should include the original image next to its denoised version. To denoise all you need to do is to take the FFT of the image and set all the high frequencies to zero before inverting to get back the filtered original. Where you choose to draw the distinction between a “high” and a “low” frequency is up to you to design and tune for to get the best result.
Note: The FFT plot you produce goes from 0 to 2π so any frequency close to zero can be considered low (even frequencies near 2π) since 2π is just zero shifted by a cycle. Your program should print in the command line the number of non-zeros you are using and the fraction they represent of the original Fourier coefficients.

For mode three you need to do two things. Firstly, you must take the FFT of the image to compress it. The compression comes from setting some Fourier coefficients to zero. There are a few ways of doing this, you can threshold the coefficients’ magnitude and take only the largest percentile of them. Alternatively, you can keep all the coefficients of very low frequencies as well as a fraction of the largest coefficients from higher frequencies to also filter the image at the same time. You should experiment with various schemes, decide what works best and justify it in your report.
Then your program should do the following: display a 2 by 3 subplot of the image at 6 different compression levels starting from the original image (0% compression) all the way to setting 95% of the coefficients to zero. To obtain the images just inverse transform the modified Fourier coefficients.

Finally, for the plotting mode your code should produce plots that summarize the runtime complexity of your algorithms. Your code should print in the command line the means and variances of the runtime of your algorithms versus the problem size. For more qualitative details about what you should plot see the report section.


