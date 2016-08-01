%%
% Adapted from Kording, Tenembaum, and Shadmehr 2007.
%

%% Joint parameters - initialize transition and observation matrices across all examples.
% Construct these matrices
states=30; %how many hidden states to use - we want 30 here
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

% Note that there are 30 memory states (30 'disturbances'), which will produce the error. The gain is implicit - error is always
% assumed to be around 0 (errors are addative). Not solving a multiplicative gain, but an addative offset.
%
% Trying to find the states that when summed equal the error, so thats why C is ones
% ypred = C*xpred, which can be compared to any y

%% Improve estimate of initV
% This is just to get a better estimate of initV, and shows an example of
% the use of sample_lds and kalman_filter.
T = 40000;
[x0,y0] = sample_lds(A, C, Q, R, initx, T);
[xfilt, Vfilt, VVfilt, loglik, xpred] = kalman_filter(y0, A, C, Q, R, initx, initV);
initV=Vfilt(:,:,end);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Replicate Hopp and Fuchs
% These reproduce figure 2b - simulates the data (sample_lds) and then perform inference on them.
%{
In these experiments, subjects had to make a 10 degree saccade (to a particular target),
but then (after 200 ‘practice’ trials) the target moves during the saccade.
Since saccades are too short to be able to use the feedback to adjust mid-saccade,
the subject can only adjust the gain on a trail by trial basis.
The subject eventually adapts by reducing the gain on the saccade (for an overshot).
Then at trial 1400, the target stops moving, and the subject adapts back to the original location.
%}

T = 4200; % Timescale of experiment, ie number of trials.
%% INSERT CODE
% Simulate the unperturbed plan using sample_lds, and then purturb it:
[x0,y] = sample_lds(A, C, Q, R, initx, T); % simulate an unperturbed plant
y(1201:2600)=y(1201:2600)-0.3; % external perturbation on y - from 1200 to 2600, produce a 30% disturbance
% Then run the filtration on it, producing xpred to be used below (graphing).
[xfilt, Vfilt, VVfilt, loglik, xpred] = kalman_filter(y, A, C, Q, R, initx, initV);
%% HERE

%% Produce figure 2b, given the filters outputs.
figure(1);
clf
subplot(2,1,1)
% We weigh all the hidden states equally, so just sum accross all of them to get the resulting
a=sum(xpred(:,1001:end)); % first 1000 trials are just to ensure initV doesnt matter much
b=a-y(1001:end); % Subjectract the gain y from the predicted values x.
b(201:1600)=b(201:1600)-0.3; %transform back because we want to plot the actual gain
plot(1+b,'.') % Plot the 'trials'
[paras]=fitExponential([1:1400],b(201:1600)); % Fit an exponential to the trial data
[paras2]=fitExponential([1:1600],b(1601:3200));
hold on
plot([201:1600],1+paras(1)+paras(2)*exp(paras(3)*[1:1400]),'r'); % Plot the exponentials
plot([1601:3200],1+paras2(1)+paras2(2)*exp(paras2(3)*[1:1600]),'r');b
xlabel('time (saccades)')
ylabel('relative size of saccade');
title('Replication of standard target jump paradigm')
subplot(2,1,2)
imagesc(xpred(:,1001:end),[-1 1]*max(abs(xpred(:)))); % Plot the values of the hidden states.
colorbar
xlabel('time (saccades)')
ylabel('log timescale (2-33000)');
title('Inferred disturbances for all timescales');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Replicate Kojima faster second time
% This replicates figure 3c.
T=4200;
for i=1:5
    %% INSERT CODE
    % First, simulate the full trial sets:
    [x0,y] = sample_lds(A, C, Q, R, initx, T); %simulate the plant
    % Then produce positive perturbation from 1001 till 1800, and then negative perturbation till the end.
    y(1001:1800)=y(1001:1800)+0.35; % positive perturbation
    y(1801:end)=y(1801:end)-0.35; % negative perturbation until gain=1
    % Then run the Kalman filter on this perturbation.
    [xfilt, Vfilt, VVfilt, loglik,xpred] = kalman_filter(y, A, C, Q, R, initx, initV);
    %% HERE
    % Now, figure out when the gain goes back to normal.
    nG=find(sum(xpred)<sum(xpred(:,1001)));
    bord=min(nG(find(nG>1800)))+1; %figure out when the gain is back to normal
    % And then produce the positive perturbation again starting at that point - remember the negative perturbation.
    y(bord:end)=y(bord:end)+0.7; %switch back to positive
    % Now run the kalman filter again...
    [xfilt, Vfilt, VVfilt, loglik,xpred] = kalman_filter(y, A, C, Q, R, initx, initV);
    % And produce a linear fit on the first and second positive perturbation.
    [parasF]=fitLinear([1:200],sum(xpred(:,1001:1200))-y(1001:1200));
    [parasS]=fitLinear([1:200],sum(xpred(:,bord:bord+199))-y(bord:bord+199));
    first(i)=parasF(2);
    second(i)=parasS(2);
end

% Produces figure 3c - need xpred = state predicted and y = observation
figure(2);
clf
subplot(3,1,1);
plot([ones(size(first));2*ones(size(first))],[first;second],'k.-');
xlabel('first time, second time')
ylabel('speed of adaptation');
title('adaptation speed: first time versus second time w reversal');
subplot(3,1,2);
a=sum(xpred(:,801:end));
b=a-y(801:end);
b(201:1000)=b(201:1000)+0.35; % remove perturbation to report standard gains
b(1001:bord-800)=b(1001:bord-800)-0.35;
b(bord-800:end)=b(bord-800:end)+0.35;
plot(1+b,'.');
[paras]=fitLinear([1:200],b(201:400)); % Fit linear functions to the resulting data.
[paras2]=fitLinear([1:200],b(bord-800:bord-800+199));
hold on
plot([201:800],1+paras(1)+paras(2)*[1:600],'r');
plot([bord-800:bord+599-800],1+paras2(1)+paras2(2)*[1:600],'r');
xlabel('time (saccades)')
ylabel('relative size of saccade');
title('adaptation, up down and up again');
subplot(3,1,3)
imagesc(xpred(:,1001:end),[-1 1]*max(abs(xpred(:))));
xlabel('time (saccades)')
ylabel('log timescale (2-33000)');
title('inferred disturbances for all timescales');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Replicate Kojimas change in the dark experiment
% Figure 3g
T=4200;
%% INSERT CODE
[x0,y] = sample_lds(A, C, Q, R, initx, T); %simulate the plant
y(1001:1800)=y(1001:1800)+0.35; % positive perturbation
y(1801:end)=y(1801:end)-0.35; % negative perturbation until gain=1
[xfilt, Vfilt, VVfilt, loglik,xpred] = kalman_filter(y, A, C, Q, R, initx, initV);
nG=find(sum(xpred)<sum(xpred(:,1001)));
bord=min(nG(find(nG>1800)))+1; %figure out when the gain is back to normal
y(bord:end)=y(bord:end)+0.7; %switch back to positive
% Now run the kalman filter, but set the isObserved to zero for the dark periods.
% The monkey has no observations for this time period, after the gain comes back to 0 for 500 time points.
% We use a time of 500 to represent 'darkness'
[xfilt, Vfilt, VVfilt, loglik,xpred] = kalman_filter(y, A, C, Q, R, initx, initV,'isObserved',[ones(1,bord),zeros(1,500),ones(1,10000)]);
a=sum(xpred);
b=y-a;
b(1001:1800)=b(1001:1800)-0.35;
b(1801:bord)=b(1801:bord)+0.35;
b(bord+1:end)=b(bord+1:end)-0.35;
figure(3)
clf
subplot(2,1,1)
plot(-b(1,1001:end).*[ones(1,bord-1-1000),zeros(1,501),ones(1,4200-bord-500)],'.')
xlabel('time (saccades)')
ylabel('relative size of saccade');
title('adaptation, up down, darkness, and up again');
subplot(2,1,2)
imagesc(xpred(:,1001:end),[-1 1]*max(abs(xpred(:))));
xlabel('time (saccades)')
ylabel('log timescale (2-33000)');
title('inferred disturbances for all timescales');
clear first second

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Replicate Kojimas no-ISS experiment  - no perturbation after darkness
% Figure 3h
figure(6)
T=4200;
[x0,y] = sample_lds(A, C, Q, R, initx, T); %simulate the plant
y(1001:1800)=y(1001:1800)+0.35; % positive perturbation
y(1801:end)=y(1801:end)-0.35; % negative perturbation until gain=1
[xfilt, Vfilt, VVfilt, loglik,xpred] = kalman_filter(y, A, C, Q, R, initx, initV); % Run kalman filter
nG=find(sum(xpred)<sum(xpred(:,1001)));
bord=min(nG(find(nG>1800)))+1; %figure out when the gain is back to normal (gain = 1)
y(bord:end)=y(bord:end)+0.35; %switch perturbation to 0.
[xfilt, Vfilt, VVfilt, loglik,xpred] = kalman_filter(y, A, C, Q, R, initx, initV,'isObserved',[ones(1,bord),zeros(1,500),ones(1,10000)]);
a=sum(xpred); % Average across all hidden states to find the adaptation gain
b=y-a; % Subtract the resulting averaged gain from the
b(1001:1800)=b(1001:1800)-0.35;
b(1801:bord)=b(1801:bord)+0.35;

% Plot figure 3h
subplot(2,1,1)
plot(1-b(1,1001:end).*[ones(1,bord-1-1000),zeros(1,501),ones(1,4200-bord-500)],'.')
xlabel('time (saccades)')
ylabel('relative size of saccade');
title('up, down adaptation followed by darkness and no-ISS trials');
subplot(2,1,2)
imagesc(xpred(:,1001:end),[-1 1]*max(abs(xpred(:))));
xlabel('time (saccades)')
ylabel('log timescale (2-33000)');
title('inferred disturbances for all timescales');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Robinson - memory over multiple days with darkness.
% Figure 2c
figure(4)
T = 40*3000;
[x0,y0] = sample_lds(A, C, Q, R, initx, T);
y(1:floor(T/2))=y0(1:floor(T/2))-0.5;
isObserved=ones(size(y));
for i=0:40
    isObserved((i*3000)+1:(i*3000)+1500)=0;
end
[xfilt, Vfilt, VVfilt, loglik, xpred] = kalman_filter(y, A, C, Q, R, initx, initV,'isObserved',isObserved);

r=sum(xpred)-y;
r(1:end/2)=r(1:end/2)-0.5;
for i=0:39
    subplot(2,1,1)
    hold on
    plot(i*1500+1500+(1:1500),1+r(i*3000+1500+(1:1500)),'ko');
    [x4(i+1,:)]=fitExponential([1:1500]/1000,1+r(i*3000+1500+(1:1500)));
    plot(i*1500+1500+(1:1500),x4(i+1,1)+x4(i+1,2)*exp(x4(i+1,3)*(1:1500)/1000),'r');
end
xlabel('time (saccades during the experiment)')
ylabel('relative size of saccade');
title('adaptation interleaved with nights in the dark');

subplot(2,1,2)
hold on
imagesc(xpred,[-1 1]*max(abs(xpred(:))));
xlabel('time (saccades, including those simulated in the dark)')
ylabel('log timescale (2-33000)');
title('inferred disturbances for all timescales');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
