%% -------------- sIRLS-q (0 <= p <= 1) algorithm -------------------- %

%% ----- This is the code associated with the paper:
% ----- "Iterative Reweighted Algorithms for Matrix Rank Minimization"
% ----- Karthik Mohan (karna@uw.edu) and Maryam Fazel (mfazel@uw.edu).

% -------------- LAST UPDATE: 8/28/2012 ------------------------------ %


function [NS, avgerr,avgiterno, TT,timeperiter, TTcpu,Xnew] = sirls_q_larav2_generalized(m,n,sr,r,rmax,rknown,eta,gam0,gammin,q,p,tol,nrg,niter,svditer,incr,type,M,Spar)




%% PARAMETERS

if(rknown == 1)
    countstart = r; %-- r if rank known
else
    countstart = rmax; %-- rmax if rank NOT known
end
TT = 0;TTcpu = 0; NS = 0;%# succesful instances
err = 0; avgerr = 0; avgiterno = 0;


if(type == 1)
    
    for(ng = 1: nrg)
        
        %% GENERATE MATRIX COMPLETION OPERATOR AND LOW RANK MATRIX X0
        
        Y1 = sprand(m,r,0.3); Y2 = sprand(n,r,0.5);
        X01 = full(Y1*Y2'); X0 = X01/norm(X01); gam = gam0*norm(X0);
        
        OM = binornd(1,sr,m,n);
        [alp,beta] = find(OM==1); %vectors defining support Omega
        [betat,alpt] = find(OM' == 1);
        [mis_i,mis_j] = find(OM' == 0);
        p1 = size(alp,1); %NUMBER OF MEASUREMNTS TAKEN ~= p*m*n
        CO = size(n,1);
        
        %% MEASUREMENTS
        B = zeros(m,n);
        for(i = 1:p1)
            B(alpt(i),betat(i)) = X0(alpt(i),betat(i));
        end
        
        %% sIRLS FOR MATRIX COMPLETION PROBLEM
        
        %FIRST ITERATION OF sIRLS
        k = 1;
        Xnew = B;
        svditer =  niter; %Parameter in rand_svd
        count = countstart;
        L = 2; %Initial Lipschitz constant
        V = 0; D1 = 0;
        extra_rank = 0;
        tstart = cputime;
        startclock = clock;
        fprintf('\n');
        
        
        v = Xnew(sub2ind(size(Xnew), mis_i, mis_j));
        N = size(mis_i,1);
        eps = 1; h = ones(N,1);
        
        
        while(k<niter)
            
            [Xnew,err,terr,l] = grad_proj(B,L,Xnew,V,D1,m,n,alpt,betat,2,mis_i,mis_j,h,p,N);
            [U,S,V] = rand_svd(Xnew,count,k,svditer,incr);
            s = diag(S);
            g = gam*ones(count,1); s = s(1:count,1);
            D1 = diag( (g.^(1 - q/2))./((s.*s + g).^(1 - q/2))- ones(count,1));
            V = V(:,1:count);
            
            count = min(size(find(s > max(s)*1e-2),1)+extra_rank,rmax);
            % Estimating rank to truncate SVD in next iteration
            if(rknown == 1)
                count = r;
            end;
            
            L = 2; %Lipschitz constant
            err = norm(Xnew - X0,'fro')/norm(X0,'fro');
            gam = max(gam/eta,gammin);
            
            x = maxk(v, Spar+1);
            l = x(Spar+1);
            eps = min(eps, l/N);
            
            k = k + 1;
            if(mod(k,40) == 0)
                fprintf('.');
                if(err < tol)
                    break
                end;
            end;
        end;
        
        if(err < tol)
            avgerr = err + avgerr;
            avgiterno = k + avgiterno;
            TTcpu = TTcpu + cputime - tstart;
            TT = TT + etime(clock,startclock);
            NS = NS + 1;
        end;
        
    end;
    
else %IF TYPE = 2
    
    B = zeros(m,n);
    Mask = zeros(m,n);
    
    for i = 1:size(M,1)
        B(M(i,1),M(i,2)) = M(i,3);
        Mask(M(i,1),M(i,2)) = 1;
    end
    
    alpt = M(:,1); betat = M(:,2);
    %gam = gam0*norm(B);
    gam = 2;
 
    [mis_i,mis_j] = find(Mask == 0);
    
    %FIRST ITERATION OF sIRLS
    k = 1;
    Xnew = B; Xold = B;
    svditer =  niter; %Parameter in rand_svd
    count = countstart;
    L = 2; %Initial Lipschitz constant
    V = 0; D1 = 0;
    extra_rank = 0;
    tstart = cputime;
    startclock = clock;
    fprintf('\n');
    
    N = size(mis_i,1);
    eps = 1; h = ones(N,1);
    
    while(k<1000) %%%%%%%%%%%%%%%%%%%%%%% niter
        p = 1; k_sparsity = 2;
        [Xnew,err,terr,l] = grad_proj_larav2(Xnew,k_sparsity,mis_i,mis_j,h,N,Spar,p,eps);
        
        % Update weights for low-rankness
        [U,S,V] = rand_svd(Xnew,count+1,k,svditer,incr);
        s = diag(S); s_countK1 = s(count+1);
        
%         % Update the regularizer gamma
%         if(k<55)  %55
%             gam = .5*gam; %.5
%         else
%             gam = min(gam, 0.2*s_countK1); %.2
%         end
        gam = .5*gam;
        gam = max(gam, 10^(-17));
        
        g = gam*ones(count,1); s = s(1:count,1);
        D1 = diag( (g.^(1 - q/2))./((s.*s + g).^(1 - q/2))- ones(count,1));
        V = V(:,1:count);
        
        count = min(size(find(s > max(s)*1e-2),1)+extra_rank,rmax);
        % Estimating rank to truncate SVD in next iteration
        if(rknown == 1)
            count = r;
        end;
        
        % Promote low-rankness
        [Xnew,err,terr,l] = grad_proj(B,L,Xnew,V,D1,m,n,alpt,betat,11);
        
        L = 2; %Lipschitz constant
        
        % Update epsilon (regularizer for sparsity weights)
        v = Xnew(sub2ind(size(Xnew), mis_i, mis_j));
%         x = sort(v(:),'descend');
%         if Spar < N
%             x_Spar1 = x(Spar+1);
%         elseif Spar == N
%             x_Spar1 = 0; %x(N);
%         end
%         eps = min(eps, x_Spar1/(N));  %1000
        eps = 0.9*eps;
        eps = max(eps, 10^(-17)); %17/N
        
        % Update weights for sparsity
        v = Xnew(sub2ind(size(Xnew), mis_i, mis_j));
        for i = 1 : N
            h(i) = (v(i)^2 + eps).^((p/2)-1); 
        end
        
        %THRESHOLD FOR SPARSITY %1e-8
        thresh = 1e-8; 
        
        for i = 1 : N
            if (abs(v(i)) < thresh)
                v(i) = 0;
            end
        end
        
        Xnew(sub2ind(size(Xnew), mis_i, mis_j)) = v;
        err = norm(Xnew - Xold,'fro')/norm(Xnew,'fro');
        Xold = Xnew;
        k = k + 1;
        if(mod(k,20) == 0)
            fprintf('.');
            if(err < tol) 
                break;
            end;
        end;
    end; 
    
    TT = cputime - tstart;
    TTcpu = TT;
    avgerr = err;
    avgiterno = k;
    timeperiter = TT/avgiterno;
    NS = 1;
    
end;

if(type == 1)
    if(NS > 0)
        TTcpu = TTcpu/NS;
        TT = TT/NS;
        avgerr = avgerr/NS;
        avgiterno = avgiterno/NS;
        timeperiter = TT/avgiterno;
    else
        TTcpu = cputime - tstart;
        TT = etime(clock,startclock);
        avgerr = err;
        avgiterno = k;
        timeperiter = TT/avgiterno;
    end;

    
end;
save('avgiterno.mat', 'avgiterno')
%save('EPS.mat', 'EPS')
end


  