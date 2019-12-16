%% ---- Set Parameters and Run Structured sIRLS ---------------- %%

%% --- This is the code associated with the paper:
% --- "Matrix Completion for Structured Observations Using Iteratively Reweighted Algorithms"
% --- Lara Kassab(kassab@math.colostate.edu)

% -------------- LAST UPDATE: 12/13/2019 -------------- %

function [error_structured_sIRLS] = run_structured_sIRLS(q,p,Y,M,m,n,r,rknown, mis_i,mis_j)

% Choose remaining parameters
measurements = size(M,1);
sr = measurements/(m*n);
numb_ms = measurements; % Number of Measurements

rmax = ceil(n*(1 - sqrt(1 - sr)));

niter = 5000;
incr = 100;
tol = 1e-5; % Tolerance for convergence

%% ----------- ALGORITHM BEGINS ------------ %%

fprintf('\n -------------------');
fprintf('\n Algorithm begins...');
fprintf('\n -------------------\n\n');

[avgiterno, TT,timeperiter, TTcpu, Xalgo] = structured_sirls_pq(m,n,r,rmax,rknown,q,p,tol,niter,incr,M, mis_i, mis_j);
error_structured_sIRLS = norm(Y - Xalgo, 'fro')/norm(Y, 'fro');


%% ----------- OUTPUT ------------ %%

fprintf('\n\n m = %d, n = %d, r = %d, measurements = %d, samp.ratio = %3.2f', m,n,r,numb_ms,sr);
fprintf(' # Iters = %d, Clock time = %3.2f, \n Clock time/iter = %3.3f, Cpu time = %3.2f, relative err = %3.6e \n\n', avgiterno, TT,timeperiter, TTcpu, error_structured_sIRLS);

fprintf('\n The completed matrix is given by Xalgo.mat ...\n');
end
