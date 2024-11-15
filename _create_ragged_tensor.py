#function for creating a ragged tensor of a matrix A

import numpy as np
import tensorflow as tf

#function for creating a ragged tensor of matrix A
#input: matrix A of any dimension
#output: ragged tensor, row i of the ragged tensor contains 
#        the predecessors of the i-th state 
def _create_ragged_tensor(A):
    
    #determine the values of the ragged tensor
    values = np.empty(np.count_nonzero(A))  #save values in here

    count = 0
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[1]):
            if A[j, i] != 0:
                values[count] = j 
                count += 1

    #determine the row_lengths of the ragged tensor
    row_lengths = (A != 0).sum(0)      #number of non zero elements per column in A
                                        #is equal to the number of predecessors of state i

    #ragged tensor of A
    ragged_t = tf.RaggedTensor.from_row_lengths(
                values = values,
                row_lengths = row_lengths)
    
    return ragged_t