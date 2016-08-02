% The purpose of this script is to test the kalman_update.m code after it has been
% fixed.

% Let's start with something simple. Using the example from:
% http://bilgin.esme.org/BitsAndBytes/KalmanFilterforDummies

% Simple example - constant function with more sensory noise than noise in
% state model.

A = 1;
C = 1;
Q = (0.01).^2;
R = (0.1).^2;
initx = 4;
initV = 1e-6;

T = 40;

[x0,y0] = sample_lds(A, C, Q, R, initx, T);
[xfilt, Vfilt, VVfilt, loglik, xpred] = kalman_filter(y0, A, C, Q, R, initx, initV);

plot(1:T, x0,'r', 1:T, y0,'g', 1:T, xfilt,'b');
