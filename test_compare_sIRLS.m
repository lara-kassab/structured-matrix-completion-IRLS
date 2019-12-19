%% This code compares sIRLS and Structured sIRLS on user-inputted data

%% --- This is the code associated with the paper:
% --- "An Iterative Method for Structured Matrix Completion"
% --- Code written by: Lara Kassab(kassab@math.colostate.edu)
% -------------- LAST UPDATE: 12/19/2019 -------------- %

close all;  clear all;
format compact;  format long e;

%% ------------- INPUTS -------------
load('M.mat');
[m,n] = size(M); % size of m-by-n matrices
q = 1; % sIRLS low-rankness parameter
p = 1; % Structured sIRLS sparsity parameter

% CHOOSE 1 if the Algorithm is allowed to use the information on the rank of the true solution
% CHOOSE 0 if the Algorithm is unware of the rank of the true solution
rknown = 1;

if rknown == 1
    r = 10; % guess of rank of the matrices
end

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

% Make sure if it is a user input data that M.mat exists

if(exist('M.mat') == 0)
    return
end


%% Matrix Completion using both methods
Y_original = zeros(m,n); %-- not used for user-inputted data

% Find the indices of the missing entries
Mask = zeros(m,n);
for i = 1:size(M,1)
    Mask(M(i,1),M(i,2)) = 1;
end
[mis_i,mis_j] = find(Mask == 0);

% Find the error using sIRLS-1
[~, Xalgo] = run_sIRLS_q(q,Y_original,M,m,n,r,rknown,2);

% Find the error using Structured sIRLS-1,1
[~, Xalgo_s] = run_structured_sIRLS(q,p,Y_original,M,m,n,r,rknown,mis_i,mis_j);


%fprintf('\n\n sIRLS error = %3.6e, \n Structured sIRLS error = %3.6e \n\n', error_sIRLS, error_Structured_sIRLS);



