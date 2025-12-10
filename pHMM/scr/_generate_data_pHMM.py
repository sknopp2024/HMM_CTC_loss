#Sara Knopp, last modified on 25.06.2025

# Generate data for pHMMs

import tensorflow as tf

# Uncomment and adjust the following lines if you want to run this script standalone
# (i.e., not imported by another script):
#import os 
#homefolder = "..."
#project_dir = os.path.join(homefolder, "Programs", "pHMM_code")
#os.chdir(project_dir)

from _funcs_pHMM_tf import (          # in Programs/pHMM_code
    _create_emission_prb_matrix, 
    _create_initial_distribution,
    _create_sparse_transition_matrix,
    _create_transition_matrix)

def generate_data_pHMM(sparse_transition_matrix = False):
    '''
    Generates random input sequences and their one-hot encoded representations. 

    The function creates batches of integer sequences of varying lengths and 
    one-hot encodes them based on a predefined alphabet size. The random seed 
    is fixed for reproducibility.

    Args:
        sparse_transition_matrix (boolean): if True, sparse transition matrices
                                            are built

    Returns:
        input_batch_list (List[tf.Tensor]): A list of integer tensors with shape 
            (batchsize, sequence_length), where each element represents a token 
            index in the vocabulary.
        one_hot_batch_list (List[tf.Tensor]): A list of one-hot encoded tensors 
            with shape (batchsize, sequence_length, alphabet_size), representing 
            the encoded form of input_batch_list.
    '''
    print('Start of data generation.')
    tf.random.set_seed(28102024) 

    batchsize = 8  
    H = 20  # Alphabet size
    input_lengths = [128, 256, 512]
    K = [x * 3 + 1 for x in input_lengths]  # Number of states
    
    B_list = []  # Create emission matrices for all input_length
    for L in input_lengths:  # Loop over all input lengths
        B = _create_emission_prb_matrix(L, H)
        B_list.append(B)
    
    pi_list = []  # Create initial distribution for all input_length
    for L in input_lengths:  # Loop over all input lengths
        pi = _create_initial_distribution(L)
        pi_list.append(pi)   

    # Generate input sequences with batch dimension 
    # Elements of dim (batchsize x input_length)
    input_batch_list = [
        tf.random.uniform(shape=(batchsize, L), maxval=H, dtype=tf.int32)
        for L in input_lengths]

    # One-hot encode the input sequences 
    # Elements of dim (batchsize x input_length x H)
    one_hot_batch_list = [tf.one_hot(u, H) for u in input_batch_list] 
    
    if sparse_transition_matrix:
        A_insert_cols = []
        A_match_cols = []
        A_delete_cols = []
        
        for k in K:
            A_insert_col, A_match_col, A_delete_col = _create_sparse_transition_matrix(k)
            A_insert_cols.append(A_insert_col)
            A_match_cols.append(A_match_col)
            A_delete_cols.append(A_delete_col)
        
        print('Data generation finished.')
        return A_insert_cols, A_match_cols, A_delete_cols, B_list, pi_list, input_batch_list, one_hot_batch_list
    else:
        A_list = []
        for L in input_lengths:  # Loop over all input lengths
            A = _create_transition_matrix(L)
            A_list.append(A)
        
        print('Data generation finished.')
        return A_list, B_list, pi_list, input_batch_list, one_hot_batch_list
