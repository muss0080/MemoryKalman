% The purpose of this script is to test the kalman_update.m code after it has been
% fixed.

% Let's start with something simple. Using the example from:
% http://bilgin.esme.org/BitsAndBytes/KalmanFilterforDummies
% let's start with assuming 1 hidden state, transitions are x_t = 2 * x_t-1.

A = 2;
C = 1;
Q = 1;
R = (0.1).^2;
initx = 0;
initV = 1e-6;

T = 10;

for i = 1:
  y0(i) = 1;
end

[x0,y0] = sample_lds(A, C, Q, R, initx, T);
[xfilt, Vfilt, VVfilt, loglik, xpred] = kalman_filter(y0, A, C, Q, R, initx, initV);
initV=Vfilt(:,:,end);
