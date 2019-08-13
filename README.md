#Power Spectrum Code
The code in this repository contains a few useful functions for producing the power spectra of distributions and, conversely, producing distributions that follow the attributes designated by a given power spectrum. It also contains some other files with helpful/used functions

##-mock\_power\_spectrum.py:
-This file offers some hopefully intuitive examples of the power spectra that correspond to different mock-distributions. It takes as input a parameter file (found in the Parameter Files directory) and can be called, for example, with: "python3 mock\_power\_spectrum.py gaussian_test.txt". This specific example will produce a Gaussian noise distribution and its power spectrum and plot them side-by-side.

-The *Parameter Files* directory contains example .txt files for generating different types of distributions and their power spectra.

-The *Output Files* directory is where this program stores the power spectra it generates and their errors, hopefully with what are clear and indicative filenames.



##power\_spectrum\_to\_universe.py:
This file only contains one fucntion that takes as input an array specifying a power spectrum and an integer indicating a number of dimensions and produces a distribution that follows the attributes specified by the given power spectrum and which spans the specified number of dimensions. This is done by populating a Fourier space according to the given power spectrum and then performing an inverse Fourier transform on it, but there are some random elements to the population process. As such, the produced distribution's power spectrum won't be an *exact* match to the one input, but it will haveroughly the same features.

##p\_spec\_tools.py:
File with some useful functions for generating and plotting power spectra.

##get\_array\_from\_txt.py:
The name sort of says most of it. Retrieves an array stored in a .txt file. Can return it either as a python list, a numpy array with a user-specified data type, or a PyTorch tensor. The array in the specified file must have been stored as a 1D python list that was directly translated into string form as such: str(the_list) and then written to the file. This is just how I've been doing it and the functions contained in this file are things I found myself doing often.

##make\_gauss\_p\_spec\_database.py:
This was a function I wrote quickly to produce a database of images/distributions that had power spectra characterised by a single gaussian-profiled spike along the k-axis. This was for the purpose of having a simple, more-predictable problem to train a convolutional neural network on. An example of a small generated database can be found in the *Data Files* directory.