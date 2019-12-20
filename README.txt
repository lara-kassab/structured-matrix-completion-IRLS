Paper: “An Iterative Method for Structured Matrix Completion"
by Henry Adams, Lara Kassab and Deanna Needell
Code written by: Lara Kassab (kassab@math.colostate.edu)
LAST UPDATE: 12/20/2019

1. Use compare_sIRLS_synthetic.m to test sIRLS and Structured sIRLS on synthetically generated data.

2. Use compare_sIRLS_user.m to test sIRLS and Structured sIRLS on user inputted data.
     i) Create an input data matrix M.mat which has the following format:
    "The matrix M.mat has 3 columns. The first two columns denote the row-   index and column-index.
     The last column has the values of matrix M at the row-column indices specified in the first two columns." 
     The matrix M represents an incomplete matrix that would be completed by the algorithm.
     ii) Specify an estimate for the rank of the true matrix. 

3. Use compare_sIRLS.m to produce the four plots displayed in the paper for synthetic data on different structured settings.
