#Sara Knopp, last modified on 30.07.2025

# Benchmark HMMs, nested for-loop Version (in log-space)

import tensorflow as tf

@tf.function(reduce_retracing=True) 
def nested_for_loops_log(A, B, pi, u, U):
    """
    Sequential forward algorithm with for-loops to compute the forward 
    probabilities (alpha) using the log-sum-exp trick for better numerical 
    stability in the batch dimension.
    
    Args:
        A (tf.Tensor): Transition matrix (K, K), dtype=tf.float32
        B (tf.Tensor): Emission matrix (K, H), dtype=tf.float32
        pi (tf.Tensor): Initial state distribution (batchsize, K) or (K,), 
                        dtype=tf.float32
        u (tf.Tensor): Input sequence indices (batchsize, L), dtype=tf.int32 or 
                       tf.int64
        U (tf.Tensor): One-hot encoded input sequence (batchsize, L, H), 
                       dtype=tf.float32
        
    Returns: 
        tf.Tensor: Log-forward probabilities alpha (batchsize, K, L), 
                   dtype=tf.float32
    
    Notes:
        - K: Number of states
        - L: Length of input sequence
        - H: Alphabet size
    """
    L = tf.shape(U)[1] # Batch size, Length of input sequence
    K = tf.shape(A)[0] # Number of states
    
    # Masks all non-zero transitions in A to identify allowed predecessor states.
    pred_mask = tf.not_equal(A, 0)
    
    # Convert to log-space
    A_log = tf.where(A > 0, tf.math.log(A), -1e6)
    B_log = tf.where(B > 0, tf.math.log(B), -1e6)
    pi_log = tf.where(pi > 0, tf.math.log(pi), -1e6)
    
    # Precompute the emission probabilities for each time step (batch_size x L x K)
    UB_log = tf.matmul(U, tf.transpose(B_log))  
    # (batch_size x L x H) @ (H x K) -> (batch_size x L x K)
 
    # alpha: (batchsize x K x L)
    alpha = tf.TensorArray(dtype=tf.float32, size=L, clear_after_read=False)
    
    # Initialize first time step (t=0)
    alpha_0 = pi_log + UB_log[:, 0, :]  
    alpha = alpha.write(0, alpha_0)  # Update alpha
    
    # Sequentially compute forward probabilities using the log-sum-exp trick
    for t in tf.range(1, L):
        
        # Initialise current column of alpha
        log_alpha_t = tf.TensorArray(dtype=tf.float32, size=K)
        
        for q in tf.range(K):
            # Predecessors of state q
            i_indices = tf.cast(tf.where(pred_mask[:, q])[:, 0], tf.int32)
            
            # Create a temporary array to store the log-values
            temp_values = tf.TensorArray(dtype=tf.float32, size=tf.shape(i_indices)[0])
            
            for j in tf.range(tf.shape(i_indices)[0]):
                pred = i_indices[j]
                
                # Calculate the log-transformed forward step for state q at time t
                temp_values = temp_values.write(j, alpha.read(t-1)[:, pred] + A_log[pred, q]) 
            
            # Apply log-sum-exp trick: log(sum(exp(x))) => log(sum(exp(x - max(x))) + max(x))
            log_alpha_t = log_alpha_t.write(
                q, tf.reduce_logsumexp(temp_values.stack(), axis=0) + UB_log[:, t, q]
                )      

        alpha_t = tf.transpose(log_alpha_t.stack(), perm=[1, 0])  # [batch_size, K]
        
        # Store the computed alpha for the current time step
        alpha = alpha.write(t, alpha_t)  

    # Stack and transpose to (batchsize x K x L)
    log_alpha_final = tf.transpose(alpha.stack(), perm=[1, 2, 0])  
    # (L x batchsize x K) -> (batchsize x K x L)
    
    return log_alpha_final

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
#pi = tf.Variable([0.5] + [0.5]+ [0.] * (K - 2), dtype=tf.float32)  # (K,)
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
    
F_alg6 = nested_for_loops_log(A, B, pi_batch, u, U)
with np.printoptions(precision = 5, suppress = True, linewidth = 100):
    print (F_alg6)
print(tf.argmax(F_alg6, axis=1))
print(tf.reduce_logsumexp(F_alg6[:, :, -1], axis = 1))  # log likelihood

#%%    
# Calculate gradient of A and B

with tf.GradientTape() as tape:
    tape.watch(pi)  # Watch pi, if not gradient is None
    pi_batch = tf.tile(tf.expand_dims(pi, axis=0), [batchsize, 1])
    
    alpha = nested_for_loops_log(A, B, pi_batch, u, U)  # (batch x K x L)
    final_log_prob = tf.reduce_logsumexp(alpha[:, :, -1], axis=1)  
    loss = -tf.reduce_mean(final_log_prob)  # Negative Log-Likelihood

# Calculate gradient
grads = tape.gradient(loss, [A, B, pi])
print(grads)
'''