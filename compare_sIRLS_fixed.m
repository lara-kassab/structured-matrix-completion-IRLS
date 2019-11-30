%% This code compares between sIRLS and Structured sIRLS for a fixed sampling rate and a fixed zero sampling rate
close all;  clear all;
format compact;  format long e;

% Dimension (m x n) of matrices to be considered
m = 500; n = 500;

% Pick a fixed sampling rate
samp = 0.6;

% Pick a fixed zero sampling rate
zero_rate = 0.9;

% Rank guess
r = 10;

% sIRLS parameters
type = 2;
q = 1; p = 1;

% Number of matrices to be averaged
numMat = 5;

% Initialize relative errors
error_Structured_sIRLS = 0;
error_sIRLS = 0;

%% Matrix Completion using both methods
for k = 1 : numMat
    
    % Construct a random matrix
    YL = sprand(m,r,0.3);
    YR = sprand(r,n,0.5);
    Y = YL*YR;
    Y = full(Y)/norm(Y, 'fro');
    
    [f,h,s] = find(Y);
    k1 = size(f,1); % number of non-zero entries in M
    
    [u,v] = find(Y == 0);
    k2 = size(u,1);  % number of zero entries in M
    
    rate2 = 1 - ((zero_rate/k1)*(1-samp)*(k1+k2)); % Rate for sampling zero entries
    rate1 = (1/k2)*(samp*(k1+k2) - (rate2*k1)); % Rate for sampling non-zero entries
    
    % Subsmapling non-zero entries
    samp_k1 = round(rate1*k1);
    [y_f,idx] = datasample(f,samp_k1,'Replace',false); % randomly subsample samp_k1 non-zero entries
    y_h = h(idx);
    
    % Subsmapling zero entries
    samp_k2 = round(rate2*k2);
    [y_u,idu] = datasample(u,samp_k2,'Replace',false); % randomly subsample samp_k2 zero entries
    y_v = v(idu);
    
    % Storing the entries of the "observed" entries
    Obs_i = [y_f ; y_u];
    Obs_j = [y_h ; y_v];
    
    % Constructing the Mask
    Mask = zeros(m,n);
    Mask(sub2ind(size(Y), Obs_i, Obs_j)) = 1;
    
    % Indices of missing entries
    [mis_i,mis_j] = find(Mask == 0);
    
    % Construct M for sIRLS
    M = [Obs_i, Obs_j, Y(sub2ind(size(Y), Obs_i, Obs_j))];
    
    % Find the error using sIRLS-1
    error_sIRLS = error_sIRLS + run_sIRLS_p(Y,M,m,n,r,type);
    
    % Find the error using Structured sIRLS-1,1
    error_Structured_sIRLS = error_Structured_sIRLS + run_structured_sIRLS(q,p,Y,M,m,n,r,type);
    
end

avg_error_sIRLS = error_sIRLS/numMat;
avg_error_Structured_sIRLS = error_Structured_sIRLS/numMat;
