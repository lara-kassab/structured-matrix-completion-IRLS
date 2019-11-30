%% ---- Test file for sIRLS-q ---------------- %%

%% ----- This is the code associated with  the paper:
% ----- "Iterative Reweighted Algorithms for Matrix Rank Minimization"
% ----- Karthik Mohan (karna@uw.edu) and Maryam Fazel (mfazel@uw.edu).

% ----- LAST UPDATE: 8/28/2012 --------------%

function [rel_error_sIRLS] = run_sIRLS(alg,q,p_spar,Y,M,m,n,r,type)

%% PROBLEM SETUP

%Problem_setup; % Set the problem up.
nrg = 1; % Enter the number of random generations of the data required to average the results.

%% CHOOSE ALL REMAINING PARAMETERS

if(type == 2)
    non_zero = size(M,1);
    [sr,p,rmax,fr,eta,niter,svditer,incr,gam0,gammin,tol] = Algorithm_parameters(n,r,non_zero,type);
end

while(q < 0 || q > 1)
    q = input('\n Enter a real number between 0 and 1:  ');
end

while(p_spar < 0 || p_spar > 1)
    p_spar = input('\n Enter a real number between 0 and 1:  ');
end

rknown = 1; % CHOOSE 1 if the Algorithm is allowed to use the
            % information on the rank of the true solution.
            % CHOOSE 2 if the Algorithm is unware of the 
            % rank of the true solution.
while(rknown <0 || rknown > 1 || abs(rknown - floor(rknown)) > 0)
    rknown = input('\n Enter either 0 or 1:  ');
end

fprintf('\n -------------------');
fprintf('\n Algorithm begins...');
fprintf('\n -------------------\n\n');

if alg == 1
    [NS, avgerr,avgiterno, TT,timeperiter, TTcpu,Xalgo] = sirls_q(m,n,sr,r,rmax,rknown,eta,gam0,gammin,q,tol,nrg,niter,svditer,incr,type,M);
    rel_error_sIRLS = norm(Y - Xalgo, 'fro')/norm(Y, 'fro');
end

if alg == 2
    [NS, avgerr,avgiterno, TT,timeperiter, TTcpu, Xalgo] = structured_sirls_pq(m,n,r,rmax,rknown,q,p_spar,tol,niter,incr,type,M);
    rel_error_sIRLS = norm(Y - Xalgo, 'fro')/norm(Y, 'fro');
    
end

%% ----------- OUTPUT ---------------------- %%

if(type == 2)
fprintf('\n\n m = %d, n = %d, r = %d, p = %d, samp.ratio = %3.2f, freedom = %3.2f, eta = %1.3f \n', m,n,r,p,sr,fr,eta);
fprintf(' # Iters = %d, Clock time = %3.2f, Clock time/iter = %3.3f Cpu time = %3.2f \n\n\n', avgiterno, TT,timeperiter, TTcpu);
end

fprintf('\n The completed matrix is given by Xalgo.mat ...\n');
end
