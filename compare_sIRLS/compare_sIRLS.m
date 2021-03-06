%% This code compares sIRLS and Structured sIRLS on different structured settings

%% --- This is the code associated with the paper:
% --- "An Iterative Method for Structured Matrix Completion"
% --- Code written by: Lara Kassab(kassab@math.colostate.edu)
% -------------- LAST UPDATE: 12/13/2019 -------------- %

close all;  clear all;
format compact;  format long e;

%% ------------- INPUTS -------------
m = 500; n = 500; % size of m-by-n matrices
numMat = 10; % number of matrices to average over
r = 10; % guess of rank of the matrices
q = 1; % sIRLS low-rankness parameter
p = 1; % Structured sIRLS sparsity parameter

% CHOOSE noise_exp = 0 to run exact recovery experiments
% CHOOSE noise_exp = 1 to run experiments with noise
noise_exp = 0;
eps_noise = 10^(-3); % set the noise parameter (or noise ratio)

% CHOOSE 1 if the Algorithm is allowed to use the information on the rank of the true solution
% CHOOSE 0 if the Algorithm is unware of the rank of the true solution
rknown = 1;

%% ------------- END OF INPUTS -------------

% Check if the inputs q, p are between 0 and 1
while(q < 0 || q > 1)
    q = input('\n Enter a real number between 0 and 1:  ');
end

while(p < 0 || p > 1)
    p = input('\n Enter a real number between 0 and 1:  ');
end

% Check if rknown equals 0 or 1 only
while(rknown <0 || rknown > 1 || abs(rknown - floor(rknown)) > 0)
    rknown = input('\n Enter either 0 or 1:  ');
end

% Check if noise_exp equals 0 or 1 only
while(noise_exp < 0 || noise_exp > 1 || abs(noise_exp - floor(noise_exp)) > 0)
    noise_exp = input('\n Enter either 0 or 1:  ');
end

% Range of the sampling rate of non-zero entries
rate1_vector = 0.1:0.05:1;
r1 = size(rate1_vector,2);

% Range of the sampling rate of zero entries
rate2_vector = 0.1:0.05:1;
r2 = size(rate2_vector,2);

errorMatA = zeros(r1,r2); % stores errors of sIRLS-q,p
errorMatB = zeros(r1,r2); % stores errors of Structured sIRLS-q,p

%% ------------- Matrix Completion using both methods -------------
for k = 1 : numMat
    
    % Construct a random matrix
    YL = sprand(m,r,0.3);
    YR = sprand(r,n,0.5);
    Y = YL*YR; Y = full(Y);
    Y = Y/norm(Y);
    Y0 = Y;
    
    for i = 1 : r1
        rate1 = rate1_vector(i);
        
        [f,h,s] = find(Y);
        szi1 = size(f,1);
        k1 = round(rate1*szi1);
        
        % Subsmapling non-zero entries
        [y_f,idx] = datasample(f,k1,'Replace',false); % randomly subsample k1 non-zero entries
        y_h = h(idx);
        
        for j = 1 : r2
            rate2 = rate2_vector(j);
            [u,v] = find(Y == 0);
            szi2 = size(u,1);
            k2 = round(rate2*szi2);
            
            % Subsmapling zero entries
            [y_u,idu] = datasample(u,k2,'Replace',false); % randomly subsample k2 zero entries
            y_v = v(idu);
            
            % Storing the entries of the "observed" entries
            Obs_i = [y_f ; y_u];
            Obs_j = [y_h ; y_v];
            
            % Constructing the Mask
            Mask = zeros(m,n);
            Mask(sub2ind(size(Y), Obs_i, Obs_j)) = 1;
            [mis_i, mis_j] = find(Mask == 0);
            
            % Perturbing the Obeserved Entry for noisy experiments
            if noise_exp == 1
                N_noise = randn(size(Obs_i));
                noise_ratio =  norm(Y(sub2ind(size(Y), Obs_i, Obs_j)),'fro')/norm(N_noise,'fro');
                Z_noise = eps_noise * noise_ratio* N_noise;
                Y(sub2ind(size(Y), Obs_i, Obs_j)) = Y(sub2ind(size(Y), Obs_i, Obs_j)) + Z_noise;
            end
            
            % Construct M for sIRLS
            M = [Obs_i, Obs_j, Y(sub2ind(size(Y), Obs_i, Obs_j))];
            
            % Find the error using sIRLS-p
            errorMatA(i,j) = errorMatA(i,j) + run_sIRLS_q(q,Y,M,m,n,r,rknown,2,0);
            
            % Find the error using Structured sIRLS-q,p
            errorMatB(i,j) = errorMatB(i,j) + run_structured_sIRLS(q,p,Y,M,m,n,r,rknown, mis_i, mis_j,0);
            
            % Reset Y to original value
            Y = Y0;
            
            fprintf('\n\n Working with matrix number %2.1f, \n Sampling rate of nonzero entries = %2.2f, \n Sampling rate of zero entries = %2.2f \n\n', k, rate1, rate2);
            
        end
    end
    save('errors.mat','errorMatA','errorMatB', 'k')
end

%% ------------- FIGURES -------------
% Plot for the ratio between the average error of the two methods
relError =  errorMatB./errorMatA; % (errors of Structured sIRLS-q,p)/(errors of sIRLS-q,p)
relError(isnan(relError)) = 1;

figure;
imagesc(rate2_vector, rate1_vector, flipud(relError))
set(gca, 'XTick', 0.1:0.1:1, 'XTickLabel', 0.1:0.1:1,'FontSize',14);
set(gca, 'YTick', 0.1:0.1:1, 'YTickLabel', 1:-0.1:0.1, 'FontSize',14);
xlabel('Sampling rate of zero entries', 'FontSize',18); ylabel('Sampling rate of non-zero entries', 'FontSize',18);
colorbar

% Plot relative errors for each method
errorMatA = errorMatA./k;
errorMatB = errorMatB./k;

eA = max(errorMatA(:));
eB = max(errorMatB(:));
emax = max(eA,eB);

% Relative average error of sIRLS
figure;
imagesc(rate2_vector, rate1_vector, flipud(errorMatA))
caxis([0 emax])
set(gca, 'XTick', 0.1:0.1:1, 'XTickLabel', 0.1:0.1:1,'FontSize',14);
set(gca, 'YTick', 0.1:0.1:1, 'YTickLabel', 1:-0.1:0.1,'FontSize',14);
xlabel('Sampling rate of zero entries', 'FontSize',18); ylabel('Sampling rate of non-zero entries', 'FontSize',18);
colorbar

% Relative average error of Structured sIRLS
figure;
imagesc(rate2_vector, rate1_vector, flipud(errorMatB))
caxis([0 emax])
set(gca, 'XTick', 0.1:0.1:1, 'XTickLabel', 0.1:0.1:1,'FontSize',14);
set(gca, 'YTick', 0.1:0.1:1, 'YTickLabel', 1:-0.1:0.1,'FontSize',14);
xlabel('Sampling rate of zero entries', 'FontSize',18); ylabel('Sampling rate of non-zero entries', 'FontSize',18);
colorbar

% Plot binary results
relError_scaled = relError;
for i = 1 : r1
    for j = 1 : r2
        if(relError(i,j)>= 1)
            relError_scaled(i,j) = 0;
        else
            relError_scaled(i,j) = 1;
        end
    end
end

figure;
imagesc(rate2_vector, rate1_vector, flipud(relError_scaled))
colormap(gray);
set(gca, 'XTick', 0.1:0.1:1, 'XTickLabel', 0.1:0.1:1,'FontSize',14);
set(gca, 'YTick', 0.1:0.1:1, 'YTickLabel', 1:-0.1:0.1, 'FontSize',14);
xlabel('Sampling rate of zero entries', 'FontSize',18);
ylabel('Sampling rate of non-zero entries', 'FontSize',18);

%% ------------- END OF FIGURES -------------
