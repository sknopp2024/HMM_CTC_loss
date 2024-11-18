# Forward Algorithms <br>

This branch contains the relevant code and plots

compare 4 variants of forward-algorithms with regard to computational time, 
maximal used RAM and mean GPU-utilization, used input sizes (10, 100, 1000, 
10 000, 100 000, 500 000) <br>
* algorithm 0: sequential with for-loops <br>
* algorithm 1: sequential with matrixmulitplication <br>
* algorithm 2: parallel with scan_associative <br>
* algorithm 3: parallel with scan_associative, but utilising the sparsity  <br>
  - alg 3 version 1: use a matrix with predecessors <br>
  - alg 3 version 2: use a list of predecessors <br>
