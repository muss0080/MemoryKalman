% The purpose of this script is to test the kalman_update.m code after it has been
% fixed.

% Let's start with something simple:
% Assume we have a particle moving along 1 dimensions, with a certain
% position. We want to infer its true position, given noisy information.

% Dynamics state transition model: 