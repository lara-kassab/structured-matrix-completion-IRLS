%% ---- Set Parameters and Run Structured sIRLS ---------------- %%

%% --- This is the code associated with the paper:
% --- "Matrix Completion for Structured Observations Using Iteratively Reweighted Algorithms"
% --- Henry Adams(adams@math.colostate.edu), Lara Kassab(kassab@math.colostate.edu), and Deanna Needell(deanna@math.ucla.edu)

% -------------- LAST UPDATE: 12/9/2019 -------------- %

function [error_structured_sIRLS] = run_structured_sIRLS(q,p_spar,Y,M,m,n,r)

%% Check if the inputs q, p_spar are correct

while(q < 0 || q > 1)
    q = input('\n Enter a real number between 0 and 1:  ');
end

while(p_spar < 0 || p_spar > 1)
    p_spar = input('\n Enter a real number between 0 and 1:  ');
end

%% Set whether the algorithm is allowed to use the rank

% CHOOSE 1 if the Algorithm is allowed to use the information on the rank of the true solution.
% CHOOSE 0 if the Algorithm is unware of the rank of the true solution.
rknown = 1;

while(rknown <0 || rknown > 1 || abs(rknown - floor(rknown)) > 0)
    rknown = input('\n Enter either 0 or 1:  ');
end

%% Choose all remaining parameters
non_zero = size(M,1);

sr = non_zero/(n*n);
p = non_zero; % # Measurements

rmax = ceil(n*(1 - sqrt(1 - sr)));
fr = r*(2*n - r)/p;

if(fr < 0.4)
    niter = 500; 
    incr = 50;
else
    niter = 5000;
    incr = 100;
end

tol = 1e-3; % Tolerance for convergence

%% ----------- ALGORITHM BEGINS ------------ %%

fprintf('\n -------------------');
fprintf('\n Algorithm begins...');
fprintf('\n -------------------\n\n');

[avgiterno, TT,timeperiter, TTcpu, Xalgo] = structured_sirls_pq(m,n,r,rmax,rknown,q,p_spar,tol,niter,incr,M);
error_structured_sIRLS = norm(Y - Xalgo, 'fro')/norm(Y, 'fro');


%% ----------- OUTPUT ------------ %%

fprintf('\n\n m = %d, n = %d, r = %d, p = %d, samp.ratio = %3.2f, freedom = %3.2f \n', m,n,r,p,sr,fr);
fprintf(' # Iters = %d, Clock time = %3.2f, Clock time/iter = %3.3f Cpu time = %3.2f \n\n\n', avgiterno, TT,timeperiter, TTcpu);

fprintf('\n The completed matrix is given by Xalgo.mat ...\n');
end
