#Sara Knopp, last modified on 08.05.2025

# Data for benchmark HMMs

from scipy.special import softmax
import numpy as np
import tensorflow as tf
    
def _generate_data():
    '''
    Generates input data and model parameters for evaluating the performance of
    forward algorithms in a Hidden Markov Model (HMM) setting.
    
    Returns:
        A (tf.Variable, dtype=tf.float32): Transition probability matrix of 
                                           shape (K, K)
        B (tf.Variable, dtype=tf.float32): Emission probability matrix of 
                                           shape (K, H)
        pi (tf.Tensor, dtype=tf.float32): Initial state distribution of shape 
                                          (batchsize, K)
        input_batch_list (list of tf.Tensor, dtype=tf.int64): List of input 
                                    sequences (each of shape [batchsize, L])
        one_hot_batch_list (list of tf.Tensor, dtype=tf.float32): Corresponding
              one-hot encoded input sequences (each of shape [batchsize, L, H])
    '''
    # Set random seed for reproducibility
    np.random.seed(28102024)
    tf.random.set_seed(28102024)
    print('Start of data generation.')

    K = 15  # Number of states
    H = tf.constant(4, dtype=tf.int32)  # Alphabet size (A, C, G, T)
    batchsize = tf.constant(8, dtype=tf.int32)  # Batch size

    # Transition matrix A (K x K), initialized to zeros
    A_np = np.zeros((K, K))

    # Transition probabilities between states (as per figure 2 from Tiberius paper, 
    # Gabriel et al., 2024)
    # IR state transitions
    A_np[0, 0] = 0.98   # IR to IR   
    A_np[0, 7] = 0.02   # IR to Start

    # Intron-0 state transitions 
    A_np[1, 1] = 0.98   # Intron-0 to Intron-0  
    A_np[1, 12] = 0.02  # Intron-0 to ASS-1

    # Intron-1 state transitions 
    A_np[2, 2] = 0.98   # Intron-1 to Intron-1  
    A_np[2, 13] = 0.02  # Intron-1 to ASS-2

    # Intron-2 state transitions
    A_np[3, 3] = 0.98   # Intron-2 to Intron-2
    A_np[3, 11] = 0.02  # Intron-2 to ASS-0

    # Exon-0 state transitions
    A_np[4, 5] = 0.98   # Exon-0 to Exon-1      
    A_np[4, 9] = 0.02   # Exon-0 to DSS-1               

    # Exon-1 state transitions 
    A_np[5, 6] = 0.96   # Exon-1 to Exon-2      
    A_np[5, 10] = 0.02  # Exon-1 to DSS-2
    A_np[5, 14] = 0.02  # Exon-1 to Stop

    # Exon-2 state transitions 
    A_np[6, 4] = 0.98   # Exon-2 to Exon-0   
    A_np[6, 8] = 0.02   # Exon-2 to DSS-0 

    # Start state transition 
    A_np[7, 5] = 1      # Start to Exon-1

    # DSS-0 state transition
    A_np[8, 1] = 1      # DSS-0 to Intron-0     

    # DSS-1 state transition
    A_np[9, 2] = 1      # DSS-1 to Intron 1

    # DSS-2 state transition
    A_np[10, 3] = 1     # DSS-2 to Intron-2

    # ASS-0 state transition
    A_np[11, 5] = 1     # ASS-0 to Exon-1

    # ASS-1 state transition
    A_np[12, 6] = 1     # ASS-1 to Exon-2

    # ASS-2 state transition
    A_np[13, 4] = 1     # ASS-2 to Exon-0

    # Stop state transition
    A_np[14, 0] = 1     # Stop to IR

    A = tf.Variable(A_np, dtype=tf.float32)

    # Generate random emission probabilities (K x H) using integers, then normalize
    B_rand = np.random.randint(100, size = (K, H)) / 100  # Random values between 0 and 1
    B_soft = softmax(B_rand, axis = 1)  
    # Apply softmax to each row to ensure probabilities sum to 1
    B = tf.Variable(B_soft, dtype=tf.float32)

    # Initial distribution (K, ), starting in the IR state
    pi_small = tf.Variable([1.] + [0.] * (K - 1), dtype=tf.float32)  # (K,)
    pi = tf.tile(tf.expand_dims(pi_small, axis=0), [batchsize, 1])  #(batchsize x K)

    # Define a list of different sequence lengths
    input_lengths = tf.constant([10, 100, 1000, 10000, 100000, 500000], dtype=tf.int32)

    # Generate input sequences with batch dimension 
    # elements of  dim (batchsize x input_length)
    input_batch_list = [tf.reshape(np.random.randint(H, size = (batchsize, L)), 
                                   (batchsize, L)) for L in input_lengths]
    # One-hot encode the input sequences
    # Elements of dim (batchsize x L x H)
    one_hot_batch_list = [tf.one_hot(u, H) for u in input_batch_list] 
    
    print('Data generation finished.')
    return A, B, pi, input_batch_list, one_hot_batch_list
