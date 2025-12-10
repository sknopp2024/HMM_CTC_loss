# Sara Knopp, last modified on 23.09.2025

# Benchmark HMMs, associative scan

import tensorflow as tf
from tensorflow_probability.python.math.scan_associative import scan_associative

#%%
#underflow save version 

@tf.function(reduce_retracing=True)
def associative_scan_scaled(A, B, pi, U):
    ''' 
    Forward algorithm computed underflow savein parallel using scan_associative().
    
    Args:
        A (tf.Tensor): Transition matrix (K, K), dtype=tf.float32
        B (tf.Tensor): Emission matrix (K, H), dtype=tf.float32
        pi (tf.Tensor): Initial state distribution (batchsize, K), dtype=tf.float32
        U (tf.Tensor): One-hot encoded input sequence (batchsize, L, H), dtype=tf.float32
        
    Returns: 
        tf.Tensor: Log-forward probabilities alpha (batchsize, K, L), dtype=tf.float32
    
    Notes:
        - K: Number of states
        - L: Length of the input sequence
        - H: Alphabet size
    '''    
    K = tf.shape(A)[0]  # Number of states
    L = tf.shape(U)[1]  # Length of input sequence
    batchsize = tf.shape(U)[0]
    
    # Expand the initial distribution and repeat it across the state dimension
    pi_large = tf.repeat(tf.expand_dims(pi, 1), K, axis = 1)  # (batchsize x K x K)

    # Precompute emission probabilities (UB = U * B.T)
    UB = tf.matmul(U, B, transpose_b = True)  
    # U: (batchsize x L x H), B.T: (H x K) -> UB: (batchsize x L x K)

    # Compute the inputs for the associative scan (M)
    # M is a tensor of shape (batchsize x (L-1) x K x K)
    M = tf.einsum('ab,htb->htab', A, UB[:, 1:, :])  

    # Calculate initial state M_0
    UB_ = tf.expand_dims(UB[:, 0, :], axis=1)  # Shape: (batchsize x 1 x K)
    M_0 = tf.expand_dims(pi_large * UB_, axis = 1)  # (batchsize x 1 x K x K)
    
    # Scale initial state 
    scaling_0 = tf.reduce_sum(M_0, axis=[-1], keepdims=True)[:, :, 0,] # (b x 1 x 1)
    scaling_0_expanded = tf.expand_dims(scaling_0, axis = -1)  # (b x 1 x 1 x 1)
    M_0_scaled = M_0 / scaling_0_expanded # (b x 1 x K x K)
    
    # Prepend the initial state to M
    M_complete = tf.concat([M_0_scaled, M], axis = 1)
    # (batchsize x L x K x K)

    # Create remaining scalings of M_1, ... M_(L-1) with value 0
    scalings = tf.zeros((batchsize, L-1, 1, 1)) # (b x (L-1) x 1 x 1)
    # Concatenate scalings of M_0, ..., M_(L-1)
    scaling_expanded = tf.concat([tf.math.log(scaling_0_expanded), scalings], axis=1)
    # (batchsize x L x 1 x 1)
    
    M_scaling = (M_complete, scaling_expanded) 
    # (batchsize x L x K x K) + (batchsize x L x 1 x 1)

    # Compute forward probabilities in parallel by associative scan
    F_all_scaled, scalings = scan_associative(func, M_scaling, axis = 1)  
    # (batchsize x L x K x K)

    # Extract the forward probabilities (F) by taking the first state from F_all
    F = F_all_scaled[:, :, 0, :]  # (batchsize x L x K)
    
    # Transpose the result to get shape (batchsize x K x L)
    F_transposed = tf.transpose(F, perm = [0, 2, 1])   
    
    return F_transposed, scalings

@tf.function(reduce_retracing=True)
def func(accumulator, M):
    ''' 
    Defines an associative binary operator suitable for use with `associative_scan`. 
    Performs a recursive update of a matrix-valued tensor and a corresponding 
    scaling term.
    
    log space num stab

    Args:
        accumulator (tuple): A tuple containing:
            - M_t_prev (tf.Tensor): Tensor of shape (batchsize, j, K, K), 
              representing the matrix M_1:(t-1) from the previous step.
            - scaling_t_prev (tf.Tensor): A scalar or broadcastable tensor, 
              representing the previous scaling term.

        M (tuple): A tuple containing:
            - M_t (tf.Tensor): Tensor of shape (batchsize, j, K, K), 
              representing the current matrix M_t.
            - e_scaling (tf.Tensor): A scalar or broadcastable tensor, 
              representing the scaling term associated with M_t.

    Returns: 
        tuple:
            - M_t_scaled (tf.Tensor): Tensor of shape (batchsize, j, K, K), 
              scaled M_1:t
            - scaling_t_ln (tf.Tensor): Tensor of shape (batchsize, j, 1, 1), 
              representing the log-scaling factor associated with M_1:t.
    Notes:
        - M_1:t := M_1 * ... * M_t with associative operator *
    '''
    M_t_prev, scaling_t_prev = accumulator  # x_t_prev: (batchsize, j, K, K)
    M_t, M_scaling = M  # M_t: (batchsize, j, K, K)
    
    c = tf.matmul(M_t_prev, M_t)  # (batchsize, j, K, K)
    
    # log Scaling
    d = tf.math.log(tf.expand_dims(
        tf.reduce_sum(c, axis=[-1], keepdims=True)[:, :, 0,], axis = -1))
    # (batchsize, j, 1, 1)
    
    M_t_scaled = c / tf.math.exp(d)  # (batchsize, j, K, K)
    
    scaling_t_ln = scaling_t_prev + M_scaling + d
    
    return (M_t_scaled, scaling_t_ln)

#%%
#small example
'''
from scipy.special import softmax
import numpy as np

# Set random seed for reproducibility
np.random.seed(28102024)
tf.random.set_seed(28102024)

K = 15  # Number of states
H = 4  # Alphabet size (A, C, G, T)
batchsize = 2  # Batch size

# Transition matrix A (K x K), initialized to zeros
A_np = np.zeros((K, K))

# Transition probabilities between states (as per figure 2 from Tiberius paper)
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
B_soft = softmax(B_rand, axis = 1)  # Apply softmax to each row to ensure probabilities sum to 1
B = tf.Variable(B_soft, dtype=tf.float32)

# Initial distribution (K, ), starting in the IR state
pi = tf.Variable([1.] + [0.] * (K - 1), dtype=tf.float32)  # (K,)
pi_batch = tf.tile(tf.expand_dims(pi, axis=0), [batchsize, 1])  #(batchsize x K)

# Define a list of different sequence lengths
input_lengths = [10, 100, 1000, 10000, 100000, 500000]

# Generate input sequences with batch dimension 
# elements of  dim (batchsize x input_length)
input_batch_list = [tf.reshape(np.random.randint(H, size = (batchsize, L)), 
                                (batchsize, L)) for L in input_lengths]
# One-hot encode the input sequences
# Elements of dim (batchsize x L x H)
one_hot_batch_list = [tf.one_hot(u, H) for u in input_batch_list] 

i = 0
u = input_batch_list[i]
U = one_hot_batch_list[i]
   
# underflow save
alpha_us, scalings_ = associative_scan_scaled(A, B, pi_batch, U)
with np.printoptions(precision = 5, suppress = True, linewidth = 100):
    print (alpha_us)
print(tf.argmax(alpha_us, axis=1))

#alpha_us: b x K x L
#scalings: b x L x 1 x 1

#underflow ab i = 1:
#alpha_rescaled_last_col = tf.math.exp(tf.squeeze(scalings_[:, -1], axis = 2)) * alpha_us[:, :, -1]
#print('Log Likelihood:\n', tf.math.log(tf.reduce_sum(alpha_rescaled_last_col, axis = 1)))

#kein underflow
print('Log Likelihood:\n', tf.squeeze(scalings_[:, -1], axis = 2))


#%%    
# Calculate gradient of A, B and pi

with tf.GradientTape() as tape:
    tape.watch(pi)  # Watch pi, if not gradient is None
    
    alpha, scalings = associative_scan_scaled(A, B, pi, one_hot_batch_list[i])  # (batch x K x L)
    #final_log_prob = tf.math.log(tf.reduce_sum(alpha[:, :, -1], axis=1))
    #loss = -tf.reduce_mean(final_log_prob)  # Negative Log-Likelihood
    loss = tf.squeeze(scalings[:, -1], axis = 2) # Negative Log-Likelihood

# Calculate gradient
grads = tape.gradient(loss, [A, B, pi])
print(grads)
'''