%% This code compares sIRLS and Structured sIRLS on user-inputted data

%% --- This is the code associated with the paper:
% --- "An Iterative Method for Structured Matrix Completion"
% --- Code written by: Lara Kassab(kassab@math.colostate.edu)
% -------------- LAST UPDATE: 12/20/2019 -------------- %

close all;  clear all;
format compact;  format long e;

%% ------------- INPUTS -------------
load('M.mat'); % load the matrix with missing entries in the special format (see README.txt file)
m = 500; n = 500; % size of the m-by-n matrix with missing entries
r = 10; % guess of rank of the matrix

% CHOOSE 1 if the Algorithm is allowed to use the information on the rank of the true solution
% CHOOSE 0 if the Algorithm is unware of the rank of the true solution
rknown = 1;

% CHOOSE 1 to mask a few extra entries and test the performance of the algorithm on these entries
% CHOOSE 0 to not alter the observed entries (i.e. without masking extra entries)
mask_extra = 1;

if mask_extra == 1
    mask_rate = 10^(-3); % percent (in decimal form) of the observed entries to be masked
    max_val = 10^(-4); % upper bound on the value of extra entries to be masked
    % max_val -- depends on the values of the data
    % -- it should not pick out an all zero or non-sparse vector
    % note -- the masked entries are not guaranteed to be sparse
end

p = 1; % sIRLS low-rankness parameter
q = 1; % Structured sIRLS sparsity parameter

%% ------------- END OF INPUTS -------------

% Check if the inputs q, p are between 0 and 1
while(p < 0 || p > 1)
    p = input('\n Enter a real number between 0 and 1:  ');
end

while(q < 0 || q > 1)
    q = input('\n Enter a real number between 0 and 1:  ');
end

% Check if rknown equals 0 or 1 only
while(rknown <0 || rknown > 1 || abs(rknown - floor(rknown)) > 0)
    rknown = input('\n Enter either 0 or 1:  ');
end

% Check if mask_extra equals 0 or 1 only
while(mask_extra <0 || mask_extra > 1 || abs(mask_extra - floor(mask_extra)) > 0)
    mask_extra = input('\n Enter either 0 or 1:  ');
end

% Make sure that M.mat exists
if(exist('M.mat') == 0)
    return
end

if size(M,2) ~= 3
    fprintf('Enter the matrix M in the right form.\n');
    return
end

% Initialize missing entries with zeros
Y_original = zeros(m,n);
for i = 1:size(M,1)
    Y_original(M(i,1),M(i,2)) = M(i,3);
end

%% Matrix Completion using both methods

% Find the indices of the missing entries
Mask = zeros(m,n);
for i = 1:size(M,1)
    Mask(M(i,1),M(i,2)) = 1;
end

[mis_i,mis_j] = find(Mask == 0);

if mask_extra == 1
    % Mask extra entires
    rands = round(size(M,1)*mask_rate);
    obs_sparse = find(M(:,3)<max_val);
    
    if rands > size(obs_sparse,1)
        fprintf('There is not %4.0f entries under %3.2e to sample.\n',rands, max_val);
        return
    end
    
    % Select random entries (with an upper bound) to mask
    idu_sparse = datasample(obs_sparse,rands,'Replace',false);
    M_masked = M(idu_sparse,3); % true values of the masked entries
    
    id_x = zeros(size(idu_sparse)); id_y = zeros(size(idu_sparse));
    for j = 1:size(idu_sparse,1)
        id_s = idu_sparse(j);
        Mask(M(id_s,1),M(id_s,2)) = 0; % update the mask
        id_x(j) = M(id_s,1); id_y(j) = M(id_s,2); % indices of the masked entries
    end
    
    [mis_i,mis_j] = find(Mask == 0);
    
    for k = 1:size(idu_sparse,1)
        idu_sparse = sort(idu_sparse,'descend');
        id_s = idu_sparse(k);
        M(id_s,:) = []; % update the matrix M - holding the observed entries
    end
    
end

% Find the error using sIRLS-q
[~, Xalgo] = run_sIRLS_q(p,Y_original,M,m,n,r,rknown,2,1);

% Find the error using Structured sIRLS-q,p
[~, Xalgo_s] = run_structured_sIRLS(p,q,Y_original,M,m,n,r,rknown,mis_i,mis_j,1);

if mask_extra == 1
    % Find the recovered extra masked entries
    v = zeros(size(idu_sparse)); % sIRLS
    vs = zeros(size(idu_sparse)); % Structured sIRLS
    
    for j = 1:size(idu_sparse,1)
        v(j,1) = Xalgo(id_x(j),id_y(j));
        vs(j,1) = Xalgo_s(id_x(j),id_y(j));
    end
    
    error = norm(v-M_masked, 'fro')/norm(M_masked, 'fro'); % sIRLS error on masked entries
    error_s = norm(vs-M_masked, 'fro')/norm(M_masked, 'fro'); % Structured sIRLS error on masked entries
end

if mask_extra == 1
    fprintf('\n\n Number of masked entries = %3.0f', rands);
    fprintf('\n sIRLS error on masked entries = %3.6e, \n Structured sIRLS error on masked entries = %3.6e \n', error, error_s);
end
