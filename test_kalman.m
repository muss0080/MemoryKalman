% The purpose of this script is to test the kalman_update.m code after it has been
% fixed.

% Let's start with something simple:
% Assume we have a particle moving along 1 dimensions, with a certain
% position. We want to infer its true position, given noisy information.

% Simple model - imagine just 2 states.
states=2; %how many hidden states to use - we want 30 here
taus=exp(-linspace(log(0.000003),log(0.5),states)); %calculate the timescales -
A=diag(1-1./taus); % the transition matrix A - diagnal matrix using equation (1) (matrix M from appendix)
C = ones(1,states); % the observation matrix C - matrix H from appendix.
Q = diag(1./taus); % the state noise matrix Q - matrix Q from appendix
Q=0.000001475*Q/sum(Q(:)); % this trick with normalizing makes it easier
                         % to experiment with other power laws for Q
                         % This way c=0.001 but its easy to play with the
                         % parameters
R = (0.05).^2; % the observation noise R - sigma_w^2 from appendix
initx = zeros(states,1); %system starts out in unperturbed state
initV = diag(1e-6*ones(states,1)); %% rough estimate of variance


[x0,y0] = sample_lds(A, C, Q, R, initx, T);
[xfilt, Vfilt, VVfilt, loglik, xpred] = kalman_filter(y0, A, C, Q, R, initx, initV);
