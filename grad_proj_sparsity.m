


function [Xnew] =   grad_proj_sparsity(Xprev,kmax,mis_i,mis_j,h,N)

%% PARAMETERS

Xnew = Xprev;
tol = 1e-4; %TOLERANCE

step2 = 10^(-6);
k = 1; err = 10;

while(k < kmax && err > tol)
    
    % Reset values
    Xold = Xnew;
    v = Xnew(sub2ind(size(Xnew), mis_i, mis_j));
    
    % Gradient Step
    for i = 1 : N
        v(i) = v(i) - (step2*h(i)*v(i));
    end
    
    % Update X
    Xnew(sub2ind(size(Xnew), mis_i, mis_j)) = v;
    
    % Compute the Error
    err = norm(Xnew - Xold,'fro')/norm(Xold,'fro');
    k = k + 1;
    
end

return