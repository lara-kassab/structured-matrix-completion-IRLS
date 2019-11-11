function[Xnew,err,terr,k] =   grad_proj_larav2(Xprev,kmax,mis_i,mis_j,h,N,Spar,p,eps)

%% PARAMETERS

Xnew = Xprev;
tol = 1e-4; %TOLERANCE

k = 1; err = 10;
terr = zeros(kmax,1);

while(k < kmax && err > tol)
    
    % Reset values
    Xold = Xnew;
    step2 = 10^(-5); %5 or 6
    v = Xnew(sub2ind(size(Xnew), mis_i, mis_j));
    
%     % Sparsity Check
%     numb_k = size(find(v==0),2);
%     
%     if numb_k >= (size(v,1) - Spar)
%         step2 = 0;
%     end
    
    % Gradient Step
    for i = 1 : N
        v(i) = v(i) - (step2*h(i)*v(i));
    end
    
    % Update X
    Xnew(sub2ind(size(Xnew), mis_i, mis_j)) = v;
    
    % Error
    err = norm(Xnew - Xold,'fro')/norm(Xold,'fro');
    terr(k,1) = err;
    k = k + 1;
    
end;

return;