%% This code compares sIRLS and Structured sIRLS on structured synthetic data

%% --- This is the code associated with the paper:
% --- "An Iterative Method for Structured Matrix Completion"
% --- Code written by: Lara Kassab(kassab@math.colostate.edu)
% -------------- LAST UPDATE: 12/20/2019 -------------- %

close all;  clear all;
format compact;  format long e;

%% ------------- INPUTS -------------
m = 500; n = 500; % size of m-by-n matrices
r = 10; % guess of rank of the matrices
numMat = 10; % number of matrices to average over
q = 1; % sIRLS low-rankness parameter
p = 1; % Structured sIRLS sparsity parameter

rate1 = 0.7; % sampling rate of non-zero entries
rate2 = 0.2; % sampling rate of zero entries (rate2 << rate1)

% CHOOSE noise_exp = 0 to run exact recovery experiments
% CHOOSE noise_exp = 1 to run experiments with noise
noise_exp = 0;             
eps_noise = 10^(-3); % set the noise parameter (or noise ratio)

% CHOOSE 1 if the Algorithm is allowed to use the information on the rank of the true solution
% CHOOSE 0 if the Algorithm is unware of the rank of the true solution
rknown = 1;

%% ------------- END OF INPUTS -------------

% Check if the inputs q, p are between 0 and 1
while(q < 0 || q > 1)
    q = input('\n Enter a real number between 0 and 1:  ');
end

while(p < 0 || p > 1)
    p = input('\n Enter a real number between 0 and 1:  ');
end

% Check if rknown equals 0 or 1 only
while(rknown <0 || rknown > 1 || abs(rknown - floor(rknown)) > 0)
    rknown = input('\n Enter either 0 or 1:  ');
end

% Check if noise_exp equals 0 or 1 only
while(noise_exp < 0 || noise_exp > 1 || abs(noise_exp - floor(noise_exp)) > 0)
    noise_exp = input('\n Enter either 0 or 1:  ');
end

% Initialize relative errors
error_Structured_sIRLS = 0;
error_sIRLS = 0;

%% Matrix Completion using both methods
for k = 1 : numMat
    
    % Construct a random matrix
    YL = sprand(m,r,0.3);
    YR = sprand(r,n,0.5);
    Y = YL*YR; Y = full(Y);
    Y = Y/norm(Y);
    Y_original = Y;
    
    % Subsmapling non-zero entries
    [f,h,s] = find(Y);
    szi1 = size(f,1);
    k1 = round(rate1*szi1);
    [y_f,idx] = datasample(f,k1,'Replace',false); % randomly subsample k1 non-zero entries
    y_h = h(idx);
    
    % Subsmapling zero entries
    [u,v] = find(Y == 0);
    szi2 = size(u,1);
    k2 = round(rate2*szi2);
    [y_u,idu] = datasample(u,k2,'Replace',false); % randomly subsample k2 zero entries
    y_v = v(idu);

    % Storing the entries of the "observed" entries
    Obs_i = [y_f ; y_u];
    Obs_j = [y_h ; y_v];
    
    % Constructing the Mask
    Mask = zeros(m,n);
    Mask(sub2ind(size(Y), Obs_i, Obs_j)) = 1;
    [mis_i,mis_j] = find(Mask == 0); % Indices of missing entries
    
    % Perturbing the Obeserved Entry for noisy experiments
    if noise_exp == 1
        N_noise = randn(size(Obs_i));
        noise_ratio =  norm(Y(sub2ind(size(Y), Obs_i, Obs_j)),'fro')/norm(N_noise,'fro');
        Z_noise = eps_noise * noise_ratio* N_noise;
        Y(sub2ind(size(Y), Obs_i, Obs_j)) = Y(sub2ind(size(Y), Obs_i, Obs_j)) + Z_noise;
    end
    
    % Construct M for sIRLS
    M = [Obs_i, Obs_j, Y(sub2ind(size(Y), Obs_i, Obs_j))];
    
    % Find the error using sIRLS-q
    [err_sIRLS, Xalgo] = run_sIRLS_q(q,Y_original,M,m,n,r,rknown,2,0);
    error_sIRLS = error_sIRLS + err_sIRLS;
    
    % Find the error using Structured sIRLS-q,p
    [err_sIRLS_s, Xalgo_s] = run_structured_sIRLS(q,p,Y_original,M,m,n,r,rknown,mis_i,mis_j,0);
    error_Structured_sIRLS = error_Structured_sIRLS+err_sIRLS_s;
    
end

avg_error_sIRLS = error_sIRLS/numMat;
avg_error_Structured_sIRLS = error_Structured_sIRLS/numMat;

fprintf('\n\n Average sIRLS error = %3.6e, \n Average Structured sIRLS error = %3.6e \n\n', avg_error_sIRLS, avg_error_Structured_sIRLS);
