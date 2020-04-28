# structured-matrix-completion-IRLS
This is the code associated with the paper: "An Iterative Method for Structured Matrix Completion"
By Henry Adams, Lara Kassab, and Deanna Needell.

In the implementation of our code, we use parts of the code corresponding to the paper 
"Iterative reweighted algorithms for matrix rank minimization" by Karthik Mohan and Maryam Fazel. 
Further, when comparing both algorithms we use the code fully. 
The code is found in the folders with the corresponding 'sIRLS_read_me.txt' file.


### To run Structured sIRLS: Open the folder "structured_sIRLS" 
1. Use structured_sIRLS_synthetic.m to run Structured sIRLS on synthetically generated data.

2. Use structured_sIRLS_user.m to run Structured sIRLS on user inputted data.
     i) Create an input data matrix M.mat which has the following format:
    "The matrix M.mat has 3 columns. The first two columns denote the row-index and column-index.
     The last column has the values of matrix M at the row-column indices specified in the first two columns." 
     The matrix M represents an incomplete matrix that would be completed by the algorithm.
     ii) Specify an estimate for the rank of the true matrix. 



### To compare between sIRLS and Structured sIRLS or to reproduce the plots in the paper: Open the folder "compare_sIRLS" 
1. Use compare_sIRLS_synthetic.m to test sIRLS and Structured sIRLS on synthetically generated data.

2. Use compare_sIRLS_user.m to test sIRLS and Structured sIRLS on user inputted data.
     i) Create an input data matrix M.mat which has the following format:
    "The matrix M.mat has 3 columns. The first two columns denote the row-index and column-index.
     The last column has the values of matrix M at the row-column indices specified in the first two columns." 
     The matrix M represents an incomplete matrix that would be completed by the algorithm.
     ii) Specify an estimate for the rank of the true matrix. 

3. Use compare_sIRLS.m to produce the four plots displayed in the paper for synthetic data 
on different structured settings. The expected computation time depends on the data size, 
the difficulty of the completion problem and the number of matrices to average over.
This is equivalent to running item #1 (19*19 cases)*(20 runs) = 7220 times, where item #1
runs both algorithms, and hence can be slow for some of the figures.

See 'sIRLS_read_me.txt' file corresponding to the IRLS code of Karthik Mohan and Maryam Fazel 
corresponding to the paper "Iterative reweighted algorithms for matrix rank minimization". Journal 
of Machine Learning Research, 13 (Nov):3441–3473, 2012.
