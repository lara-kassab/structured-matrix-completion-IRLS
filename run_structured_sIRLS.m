%% ---- Set Parameters and Run Structured sIRLS ---------------- %%

%% --- This is the code associated with the paper:
% --- "An Iterative Method for Structured Matrix Completion"
% --- Code written by: Lara Kassab(kassab@math.colostate.edu)

% -------------- LAST UPDATE: 12/13/2019 -------------- %

function [error_structured_sIRLS,Xalgo] = run_structured_sIRLS(q,p,Y,M,m,n,r,rknown, mis_i,mis_j,user)

% Choose remaining parameters
measurements = size(M,1);
sr = measurements/(m*n); % sampling ratio
numb_ms = measurements; % number of Measurements

rmax = ceil(n*(1 - sqrt(1 - sr))); % used if the rank of the matrix is unknown

niter = 5000; % max number of iterations to perfrom for Structured sIRLS
incr = 100; % parameter in rand_svd
tol = 1e-5; % Tolerance for convergence

%% ----------- ALGORITHM BEGINS ------------ %%

fprintf('\n -------------------');
fprintf('\n Algorithm begins...');
fprintf('\n -------------------\n\n');

% run Structured sIRLS algorithm
[avgiterno, TT,timeperiter, TTcpu, Xalgo] = structured_sirls_pq(m,n,r,rmax,rknown,q,p,tol,niter,incr,M, mis_i, mis_j);

% compute the error of Structured sIRLS
error_structured_sIRLS = norm(Y - Xalgo, 'fro')/norm(Y, 'fro');


%% ----------- OUTPUT ------------ %%
if user == 0
    fprintf('\n\n m = %d, n = %d, r = %d, measurements = %d, samp.ratio = %3.2f', m,n,r,numb_ms,sr);
    fprintf(' # Iters = %d, Clock time = %3.2f, \n Clock time/iter = %3.3f, Cpu time = %3.2f, relative err = %3.6e \n\n', avgiterno, TT,timeperiter, TTcpu, error_structured_sIRLS);
else
    fprintf('\n\n m = %d, n = %d, r = %d, measurements = %d, samp.ratio = %3.2f', m,n,r,numb_ms,sr);
    fprintf(' # Iters = %d, Clock time = %3.2f, \n Clock time/iter = %3.3f, Cpu time = %3.2f \n\n', avgiterno, TT,timeperiter, TTcpu);
    
end

    fprintf('\n The completed matrix is given by Xalgo_s.mat ...\n');
