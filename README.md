# MemoryKalman

Modified code from Kording, Tenembaum, and Shadmehr 2007, originally from:
http://www.nature.com/neuro/journal/v10/n6/suppinfo/nn1901_S1.html

For use for COSMO 2016 tutorials.

The code here has been edited with some added comments and some key sections removed.
COSMO attendees should download and work through this code, first fixing the Kalman_filter.m to create
this figure

then go to the KTS.m file and reproduce the figures from the original paper.

Original script descriptions
---

Supplemental.zip is a ZIP file that contains a compressed directory. After uncompressing using 7Zip, WinZip, Unzip or essentially any other decompression program that extracts zip files it contains the following files:

KTS.m: A MATLAB script file that replicates all the eye movement simulations.

fitExponential.m: A MATLAB file that allows fitting an exponential to a set of data points.

fitLinear.m: A MATLAB file that allows fitting a linear function to a set of data points.

gaussian_prob.m: A MATLAB function that allows calculating the probability of data points under a gaussian probability distribution.

kalman_filter: A MATLAB function that allows simulating a Kalman filter.

kalman_update: A MATLAB function that is doing the timestep to timestep update that is called by kalman_filter.

sample_gaussian: A MATLAB function that allows sampling values out of a Gaussian distribution.

sample_lds: A MATLAB function that allows sampling values from a stochastic linear dynamical system.

All the MATLAB files can be viewed with any text editor and executed by MATLAB version 7.0.
