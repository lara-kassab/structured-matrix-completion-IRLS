%% -------------- Structured IRLS-p,q (0 <= p,q <= 1) algorithm -------------------- %

%% ----- This is the code associated with the paper:
% ----- "Low-rank Matrix Completion for Structured Observations Using Iteratively Reweighted Algorithms"
% ----- Henry Adams (email), Lara Kassab (email), and Deanna Needell (email)

% -------------- LAST UPDATE: 11/20/2019 ------------------------------ %

function [NS, avgerr,avgiterno, TT,timeperiter, TTcpu,Xnew] = structured_sirls_pq(m,n,r,rmax,rknown,q,p,tol,niter,incr,type,M)


%% PARAMETERS

if(rknown == 1)
    countstart = r; %-- r if rank known
else
    countstart = rmax; %-- rmax if rank NOT known
end
TT = 0;TTcpu = 0; NS = 0; % # succesful instances
err = 0; avgerr = 0; avgiterno = 0;

if(type == 2)
    
    B = zeros(m,n);
    Mask = zeros(m,n);
    
    for i = 1:size(M,1)
        B(M(i,1),M(i,2)) = M(i,3);
        Mask(M(i,1),M(i,2)) = 1;
    end
    
    alpt = M(:,1); betat = M(:,2);
    [mis_i,mis_j] = find(Mask == 0);
    
    % Initialization of regularizers for Weights
    gam = 2; eps = 1;
    
    %FIRST ITERATION OF sIRLS
    k = 1;
    Xnew = B; Xold = B; % Initialization -- impute all missing entries with zeros
    svditer =  niter; %Parameter in rand_svd
    count = countstart;
    L = 2; %Initial Lipschitz constant
    extra_rank = 0;
    tstart = cputime;
    startclock = clock;
    fprintf('\n');
    
    N = size(mis_i,1);
    h = ones(N,1); % sparsity weights intialization
    
    while(k<1000) %%%%%niter
        k_sparsity = 2; % max number of gradient step iterations
        [Xnew,err,terr,l] = grad_proj_sparsity(Xnew,k_sparsity,mis_i,mis_j,h,N);
        
        % Update weights for low-rankness
        [U,S,V] = rand_svd(Xnew,count,k,svditer,incr);
        s = diag(S);
        
        % Update the regularizer gamma (for low-rank promoting weights)
        gam = .5*gam;
        gam = max(gam, 10^(-17)); 
        g = gam*ones(count,1); s = s(1:count,1);
        D1 = diag( (g.^(1 - q/2))./((s.*s + g).^(1 - q/2))- ones(count,1));
        V = V(:,1:count);
        
        count = min(size(find(s > max(s)*1e-2),1)+extra_rank,rmax);
        % Estimating rank to truncate SVD in next iteration
        if(rknown == 1)
            count = r;
        end
        
        % Promote low-rankness
        [Xnew,err,terr,l] = grad_proj(B,L,Xnew,V,D1,m,n,alpt,betat,11);
        
        L = 2; %Lipschitz constant
        
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
    avgerr = err;
    avgiterno = k;
    timeperiter = TT/avgiterno;
    NS = 1;
    
end
end


  