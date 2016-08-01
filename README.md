# MemoryKalman

Modified code from [Kording, Tenembaum, and Shadmehr 2007 supplemental information](http://www.nature.com/neuro/journal/v10/n6/suppinfo/nn1901_S1.html).

Created for COSMO 2016 tutorials.

## Instructions

The code here has been edited from the original with some added comments and some key sections removed.
First read through the original paper (Kording, Tenembaum, and Shadmehr 2007), to get an understanding of what you're doing. Then download the code and try to replicate the eye movement simulation figures.

When producing motor commands, errors occur due to a wide array of disturbances such as fatigue, damage, or development. All of these disturbances occur at different timescales, and require different adaptations. Problems at a shorter timescale should be forgotten, while longer ones should be kept around.

First fix the kalman_update.m with the kalman update equations. Work through the simple example from the test_kalman.m file with your now working kalman filter.

Then go to the KTS.m file and reproduce the figures from the original paper, specifically figures 2b,c and 3c,g,h

If you have finished all that, try writing up new code to replicate the contrast adaptation and word learning examples.

## Notes

In the kalman_update file I have removed the control component, u. This is important for the full Kalman Filter, but not for our use so can be ignored. There are some other bits that are stripped down or hopefully simplified.

Since this is modified from the original code linked above, you could download the original and look at the solutions.

### Original script descriptions

> KTS.m: A MATLAB script file that replicates all the eye movement simulations.

> fitExponential.m: A MATLAB file that allows fitting an exponential to a set of data points.

> fitLinear.m: A MATLAB file that allows fitting a linear function to a set of data points.

> gaussian_prob.m: A MATLAB function that allows calculating the probability of data points under a gaussian probability distribution.

> kalman_filter: A MATLAB function that allows simulating a Kalman filter.

> kalman_update: A MATLAB function that is doing the timestep to timestep update that is called by kalman_filter.

> sample_gaussian: A MATLAB function that allows sampling values out of a Gaussian distribution.

> sample_lds: A MATLAB function that allows sampling values from a stochastic linear dynamical system.

> All the MATLAB files can be viewed with any text editor and executed by MATLAB version 7.0.

These files have been run on MATLAB 2013a, but should work with any version.

##### Notation
Apologies about everything ever - some very smart people think that changing the notation for terms is unimportant, but for those learning it can be a huge wall. For my own sanity, I've made the (hopefully correct) table below to translate the important matrices from the Kalman update [wikipedia page](https://en.wikipedia.org/wiki/Kalman_filter), the paper Kording, Tenembaum, and Shadmehr 2007, and their accompanying code.

Term | Wiki | Paper | Code
--| -- | -- | --
State transition model | F | M | A
State transition variance | Q | Q | Q
Observation model | H | H | C
Observation uncertainty | P | V | V

In a Bayesian sense (the way I think), these relate to the following probabilities:

Prediction: P(x_t | x_t-1) = N(A x_t-1 , Q)
Estimation: P(y_t |
