%% -------------- Structured sIRLS-p,q (0 <= p,q <= 1) algorithm -------------------- %

%% --- This is the code associated with the paper:
% --- "Matrix Completion for Structured Observations Using Iteratively Reweighted Algorithms"
% --- Henry Adams(adams@math.colostate.edu), Lara Kassab(kassab@math.colostate.edu), and Deanna Needell(deanna@math.ucla.edu)

% -------------- LAST UPDATE: 12/13/2019 -------------- %

function [avgiterno, TT,timeperiter, TTcpu, Xnew] = structured_sirls_pq(m,n,r,rmax,rknown,q,p,tol,niter,incr,M, mis_i, mis_j)


%% Set the rank

if(rknown == 1)
    countstart = r; %-- r if rank known
else
    countstart = rmax; %-- rmax if rank NOT known
end

%% Algorithm

B = zeros(m,n);

for i = 1:size(M,1)
    B(M(i,1),M(i,2)) = M(i,3); % initialize missing entries with zeros
end

alpt = M(:,1); betat = M(:,2); % find the indices of the observed entries

% Initialization of regularizers for weights
gam = 2; eps = 1;

%FIRST ITERATION OF sIRLS
k = 1;
Xnew = B; Xold = B; % Initialization -- impute all missing entries with zeros
svditer =  niter; % Parameter in rand_svd
count = countstart; % set the rank
L = 2; %Initial Lipschitz constant
extra_rank = 0;
tstart = cputime;
startclock = clock;
fprintf('\n');

N = size(mis_i,1); % number of missing entries
h = ones(N,1); % sparsity weights intialization

while(k<niter) 
    k_sparsity = 2; % maximum number of gradient steps to promote sparsity
    [Xnew] = grad_proj_sparsity(Xnew,k_sparsity,mis_i,mis_j,h,N); % promote sparsity
    
    % Update weights for low-rankness
    [U,S,V] = rand_svd(Xnew,count,k,svditer,incr);
    s = diag(S);
    gam = .5*gam; % update the regularizer gamma
    gam = max(gam, 10^(-17)); % make sure gamma is not too small (~0)
    g = gam*ones(count,1); s = s(1:count,1);
    D1 = diag( (g.^(1 - q/2))./((s.*s + g).^(1 - q/2))- ones(count,1));
    V = V(:,1:count);
    
    % Estimating rank to truncate SVD in next iteration
    if(rknown == 1)
        count = r;
    else
        count = min(size(find(s > max(s)*1e-2),1)+extra_rank,rmax);
    end
    
    % Promote low-rankness
    k_lowrank = 11; % maximum number of gradient steps to promote low-rankness
    [Xnew,err,terr,l] = grad_proj(B,L,Xnew,V,D1,m,n,alpt,betat,k_lowrank); % promote low-rankness
    
    % Update epsilon (regularizer for sparsity weights)
    eps = 0.9*eps;
    eps = max(eps, 10^(-17));
    
    % Update weights for sparsity
    v = Xnew(sub2ind(size(Xnew), mis_i, mis_j));
    for i = 1 : N
        h(i) = (v(i)^2 + eps).^((p/2)-1);
    end
    Xnew(sub2ind(size(Xnew), mis_i, mis_j)) = v;
    
    err = norm(Xnew - Xold,'fro')/norm(Xnew,'fro');
    Xold = Xnew;
    k = k + 1;
    if(mod(k,20) == 0)
        fprintf('.');
        if(err < tol)
            break;
        end
    end
end

TT = cputime - tstart;
TTcpu = TT;
avgiterno = k;
timeperiter = TT/avgiterno;

end



  