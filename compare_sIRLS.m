%% Plot the relative average error between sIRLS and Structured sIRLS (gradient algorithm)
close all;  clear all;
format compact;  format long e;
tic

% Range of the sampling rate of non-zero entries
rate1_vector = 0.1:0.05:1;
r1 = size(rate1_vector,2);

% Range of the sampling rate of zero entries
rate2_vector = 0.1:0.05:1;
r2 = size(rate2_vector,2);

% Number of matrices considered
Matrices = [];
m = 100; n = 100;
numMat = 10; % number of matrices
r = 10; % rank of the matrices
errorMatA = zeros(r1,r2); % errors of of sIRLS-1
errorMatB = zeros(r1,r2); % errors of Structured sIRLS-1,1

% sIRLS parameters
type = 2;
q = 1; p = 1;

for k = 1 : numMat
    
    YL = sprand(m,r,0.3);
    YR = sprand(r,n,0.5);
    Y = YL*YR;
    Y = full(Y)/norm(Y, 'fro');
    Matrices = [Matrices, Y];
    
end

%% Matrix Completion using both methods
for k = 1 : numMat
    Y = Matrices (:,n*(k-1)+1:n*k);
    
    for i = 1 : r1
        rate1 = rate1_vector(i);
        
        [f,h,s] = find(Y);
        szi1 = size(f,1);
        k1 = round(rate1*szi1);
        
        % Subsmapling (100*rate1) percent of non-zero entries
        [y_f,idx] = datasample(f,k1,'Replace',false); % randomly subsample k1 non-zero entries
        y_h = h(idx);
        
        for j = 1 : r2
            rate2 = rate2_vector(j);
            [u,v] = find(Y == 0);
            szi2 = size(u,1);
            k2 = round(rate2*szi2);
            
            % Subsmapling (100*rate2) percent of zero entries
            [y_u,idu] = datasample(u,k2,'Replace',false); % randomly subsample k2 zero entries
            y_v = v(idu);
            
            % Storing the entries of the "observed" entries
            Obs_i = [y_f ; y_u];
            Obs_j = [y_h ; y_v];
            
            % Constructing the Mask
            Mask = zeros(m,n);
            Mask(sub2ind(size(Y), Obs_i, Obs_j)) = 1;
            [mis_i, mis_j] = find(Mask == 0);
            
            % Compute sparsity
            Sparsity = find(Y(sub2ind(size(Y), mis_i, mis_j)));
            Spar = size(Sparsity,1);
            
            % Construct M for sIRLS
            M = [Obs_i, Obs_j, Y(sub2ind(size(Y), Obs_i, Obs_j))];
            
            % Find the error using sIRLS-1
            errorMatA(i,j) = errorMatA(i,j) + run_sIRLS(1,q,p,Y,M,m,n,r,Spar,type);
            
            % Find the error using Structured sIRLS-1,1
            errorMatB(i,j) = errorMatB(i,j) + run_sIRLS(2,q,p,Y,M,m,n,r,Spar,type);
            
        end
    end
end
% ratio of the relative error between the two methods
relError =  errorMatB./errorMatA;
toc

%% Plot the matrix of relative average error
figure;
imagesc(rate2_vector, rate1_vector, flipud(relError))
%caxis([0 mmax])
set(gca, 'XTick', 0.1:0.1:1, 'XTickLabel', 0.1:0.1:1,'FontSize',14);
set(gca, 'YTick', 0.1:0.1:1, 'YTickLabel', 1:-0.1:0.1, 'FontSize',14);
xlabel('Sampling rate of zero entries', 'FontSize',16); ylabel('Sampling rate of non-zero entries', 'FontSize',16);
%title('Relative average error')
colorbar

%% Plot relative errors for each method
errorMatA = errorMatA./50;
errorMatB = errorMatB./50;

eA = max(errorMatA(:));
eB = max(errorMatB(:));
emax = max(eA,eB);

figure;
imagesc(rate2_vector, rate1_vector, flipud(errorMatA))
caxis([0 emax])
set(gca, 'XTick', 0.1:0.1:1, 'XTickLabel', 0.1:0.1:1,'FontSize',14); 
set(gca, 'YTick', 0.1:0.1:1, 'YTickLabel', 1:-0.1:0.1,'FontSize',14);
xlabel('Sampling rate of zero entries', 'FontSize',16); ylabel('Sampling rate of non-zero entries', 'FontSize',16);
%title('Relative average error of sIRLS')
colorbar

figure;
imagesc(rate2_vector, rate1_vector, flipud(errorMatB))
caxis([0 emax])
set(gca, 'XTick', 0.1:0.1:1, 'XTickLabel', 0.1:0.1:1,'FontSize',14); 
set(gca, 'YTick', 0.1:0.1:1, 'YTickLabel', 1:-0.1:0.1,'FontSize',14);
xlabel('Sampling rate of zero entries', 'FontSize',16); ylabel('Sampling rate of non-zero entries', 'FontSize',16);
%title('Relative average error of Structured sIRLS')
colorbar

