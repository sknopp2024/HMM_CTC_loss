#Sara Knopp, last modified on 11.08.2025

# Compute entries of the forward probability matrix of a pHMM columnwise using
# for loops

import tensorflow as tf

# Uncomment and adjust the following lines if you want to run this script standalone
# (i.e., not imported by another script):
# Change the working directory to the project directory
#import os
#homefolder = "..."
#project_dir = os.path.join(homefolder, "Programs", "pHMM_code")
#os.chdir(project_dir)

from _funcs_pHMM_tf import (
    _create_distance_array)
#%%
@tf.function(reduce_retracing = True)
def pHMM_nested_for(A, B, pi, u, U):
    """
    This function computes the forward probabilities (alpha) in log space for a 
    profile Hidden Markov Model (pHMM).
    
    Args:
        u (tf.Tensor): Input sequence indices (batchsize, input_length), dtype=tf.int32.
        U (tf.Tensor): One-hot encoded input sequence (batchsize, input_length, H), 
                        dtype=tf.float32.
        alphabet_size (int): Size of Alphabet.
        
    Returns:
        tf.Tensor: Forward probability matrix (alpha) (batchsize, K, input_length)
                    dtype=tf.float32. 
    Notes:
        - H: Alphabet size
        - L: Input length
        - K: Number of states
    """
    input_length = tf.shape(u)[1]  # Length of input sequence
    K = 3*input_length+1  # Number of states 
    batchsize = tf.shape(u)[0]  # Batch size
    
    # Masks all non-zero transitions in A to identify allowed predecessor states.
    pred_mask = tf.not_equal(A, 0)
    
    # Convert transition and emission matrix and initial distribution to log space
    A_log = tf.where(A > 0, tf.math.log(A), -1e6)  
    B_log = tf.where(B > 0, tf.math.log(B), -1e6)  
    pi_log = tf.where(pi > 0, tf.math.log(pi), -1e6) 
    
    # Duration array (K,)
    d = _create_distance_array(input_length) 
    
    # Precompute the emission probabilities for each time step (batch_size x L x K)
    UB_log = tf.matmul(U, tf.transpose(B_log))  # (batch_size x L x H) @ (H x K) -> (batch_size x L x K)

    # Initialisation of forward prb matrix, (batchsize x K x L)
    log_alpha = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True,
                      clear_after_read=True)

    # Initialise first time step (t = 0)
    log_alpha = log_alpha.write(0, pi_log + UB_log[:, 0])  # Update log_alpha
    
    for t in tf.range(1, input_length):  # Loop over time steps/columns
        # Initialize current column with -inf entries
        current_alpha_col = tf.fill([batchsize, K], tf.float32.min)
        
        for q in tf.range(K):  # Loop over all states/rows
            # Predecessors of state q
            pred_indices = tf.cast(tf.where(pred_mask[:, q])[:, 0], tf.int32)
            num_pred = tf.shape(pred_indices)[0]
            
            # Create a temporary Variable to store interim results
            temp_values = tf.fill([batchsize, num_pred], tf.float32.min)
            
            # For each time step, we construct an entire column of log_alpha 
            # before writing it back to the TensorArray. Depending on the state, 
            # the required previous column varies (current or earlier time step), 
            # based on the duration array `d`. 
            # We avoid inserting individual elements into the TensorArray, as 
            # that would be inefficient.
            if d[q] == 0:
                alpha_pred_col = current_alpha_col
            else:
                alpha_pred_col = log_alpha.read(int(t-d[q]))  # Previous column
                
            for j in tf.range(num_pred):  # Loop over all predecessors of q
                pred = pred_indices[j]

                updates_temp = A_log[pred, q] + alpha_pred_col[:, pred]
                scatter_indices_temp = tf.stack([tf.range(batchsize), tf.fill([batchsize], j)], axis=1)  # Indizes für das Update
                temp_values = tf.tensor_scatter_nd_update(temp_values,
                                    scatter_indices_temp, updates_temp)
            
            # Apply log-sum-exp trick: log(sum(exp(x - max(x))) + max(x))
            updates = tf.reduce_logsumexp(temp_values, axis = 1) + tf.gather(B_log[q], u[:, t])
            scatter_indices = tf.stack([tf.range(batchsize), tf.fill([batchsize], q)], axis=1)  # Indizes für das Update
            current_alpha_col = tf.tensor_scatter_nd_update(current_alpha_col,
                                scatter_indices, updates)
        
        # Update log_alpha with t-th column
        log_alpha = log_alpha.write(t, current_alpha_col)
        
    # Convert log_alpha back to the normal alpha values
    alpha_final = tf.transpose(log_alpha.stack(), perm=[1, 2, 0])
    
    return alpha_final

#%%
#small example
'''
from _funcs_pHMM_tf import (
    _create_transition_matrix, _create_emission_prb_matrix, 
    _create_initial_distribution)

tf.random.set_seed(28102024) 

batchsize = 2
H = 20  # Alphabet size
input_length = [3, 10, 50] # List of different sequence lengths

# Generate input sequences with batch dimension, elements of dim (batchsize x input_length)
input_batch_list = [
    tf.random.uniform(shape=(batchsize, L), maxval=H, dtype=tf.int32)
    for L in input_length]

# One-hot encode the input sequences, elements of dim (batchsize x input_length x H)
one_hot_batch_list = [tf.one_hot(u, H) for u in input_batch_list] 

i = 0

A = _create_transition_matrix(input_length[i])
B = _create_emission_prb_matrix(input_length[i], H)
pi = _create_initial_distribution(input_length[i])
u = input_batch_list[i]
U = one_hot_batch_list[i]

alpha_for_tf = pHMM_nested_for(A, B, pi, u, U)
print(alpha_for_tf)
tf.argmax(alpha_for_tf, axis=1)

last_log_alpha = alpha_for_tf[:, :, -1]  # shape: (batchsize, K)
log_likelihood = tf.reduce_logsumexp(last_log_alpha, axis=1)  # shape: (batchsize,)
print(-log_likelihood)

#%%
# Apply the algorithm and compute gradient
with tf.GradientTape() as tape:
    tape.watch(A)
    tape.watch(B)
    tape.watch(pi)  # Watch pi, if not gradient is None
    
    alpha_log = pHMM_nested_for(A, B, pi, u, U)  # (batch x K x L)
    final_log_prob = tf.reduce_logsumexp(alpha_log[:, :, -1], axis=1)  
    loss = -tf.reduce_mean(final_log_prob)  # Negative Log-Likelihood Loss

# Calculate gradient
grads = tape.gradient(loss, [A, B, pi])

print(grads)

'''
