#Sara Knopp, last modified on 24.06.2025

# Improve vectorised Version of CTC_HMM with for-loop by writing a function for
# calculating UB stepwise to save memory

import tensorflow as tf

# Uncomment and adjust the following lines if you want to run this script standalone
# (i.e., not imported by another script):
# Change the working directory to the project directory
#import os
#homefolder = "..."
#project_dir = os.path.join(homefolder, "Programs", "CTC_HMM_code")
#os.chdir(project_dir)

from _funcs_ctc_HMM_tf import (
    check_min_num_frames,
    _build_sparse_transition_matrix,
    _create_UB_slice,
    _create_initial_distribution,
    _label_expansion,
)

#%%
@tf.function(reduce_retracing=True)
def CTC_HMM_vectorised_nested(labels, logits, num_frames, blank):
    """
    Compute the CTC (Connectionist Temporal Classification) loss.

    Args:
        labels (tf.tensor): Input sequences, where each label is an integer
                         (batchsize x label_length).
        logits (tf.tensor): Logits from the neural network 
                         (batchsize x num_labels x num_frames). 
        num_frames (int): Number of frames in the input sequence.
        blank (int): The index representing the blank token in the set of labels.

    Returns:
        tensor: The CTC loss for each batch.

    Notes:
        - num_labels: Number of possible labels (including the blank token)
    """
    batchsize = tf.shape(labels)[0]
    
    # Ensure that there are enough frames to cover the label sequences
    penalty = check_min_num_frames(labels, batchsize, num_frames)
    #if num_frames too small -> penalty > 0
    tf.debugging.assert_less_equal(penalty, 0.0, 
    message = "The number of frames is too small to complete the given label sequence.")

    # Expand the labels to include the blank token
    expanded_labels = _label_expansion(labels, blank) #(batchsize x z)
    z = len(expanded_labels[0])

    # Initial distribution (1 x z), pi = (1, 1, 0, ..., 0)
    pi = _create_initial_distribution(z)
    pi_log = tf.where(pi > 0, tf.math.log(pi), -1e6)
    
    # Normalise logits with softmax (batchsize x num_frames x label_length)
    logits_log = tf.nn.log_softmax(logits)
    
    # Create a aparse transition matrix A (batchsize x z)
    A = _build_sparse_transition_matrix(expanded_labels, batchsize, z)
    
    # Calculate alpha
    subdiagonal_1_padding = tf.fill((batchsize, 1), -1e6)
    subdiagonal_2_padding = tf.fill((batchsize, 2), -1e6)
    
    # Initialize the forward matrix F (batch_size, z, num_frames)
    F = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True,
                       element_shape=(None, None),clear_after_read=False)
    
    # Initialize the first column of F
    F = F.write(0, pi_log + _create_UB_slice(logits_log[:, 0], expanded_labels))  # Update alpha
    
    # Define subdiagonals
    subdiagonal_1_padding = tf.fill((batchsize, 1), -1e6)
    subdiagonal_2_padding = tf.fill((batchsize, 2), -1e6)
    
    # Perform the forward pass through the frames
    for t in tf.range(1, num_frames):
        
        prev_t = F.read(t - 1)  # Shape: [batchsize, z]
        prev_t_slice1 = prev_t[:, :-1]
        prev_t_slice2 = prev_t[:, :-2]
        
        subdiagonal_1 = tf.concat([subdiagonal_1_padding, prev_t_slice1], axis=1)
        subdiagonal_2 = tf.concat([subdiagonal_2_padding, prev_t_slice2 + A[:, 2:]], axis=1)
        # [batchsize, z]
        
        F_t = tf.reduce_logsumexp(
            [prev_t, subdiagonal_1, subdiagonal_2],  # F_t-1 + A
            axis=0
        ) + _create_UB_slice(logits_log[:, t], expanded_labels)  # Add UB for the current time step
        
        F = F.write(t, F_t) # (num_frames x batchsize x z)
        
    # Compute the CTC loss for each batch
    last_frame = F.read(num_frames - 1)  # Shape: [batchsize, z]
    slice1 = last_frame[:, z - 1]  # Shape: [batchsize]
    slice2 = last_frame[:, z - 2]  # Shape: [batchsize]
    ctc_loss = -tf.math.reduce_logsumexp(tf.stack([slice1, slice2], axis=0), axis=0)
    
    return ctc_loss  


#%%
#small example data
'''
# Set random seed for reproducibility
tf.random.set_seed(28102024) 

H = 4  # Alphabet size (number of possible nucleotides: A, C, G, T)
blank = 0  # Blank index
num_labels = 5  # Number of labels [A, C, G, T, e]
batchsize = 4  # Batch size
num_frames = 50  # Number of Frames
label_length = 20  # Length of input sequences

# Create a list of input sequences of different length
labels = tf.random.uniform([batchsize, label_length],
                           minval = 1, maxval = H+1, dtype = tf.int32)

# Create list of logits
logits_single = tf.random.uniform([num_frames, num_labels])
logits = tf.tile(tf.expand_dims(logits_single, axis=0), [batchsize, 1, 1])

tf.print(CTC_HMM_vectorised_nested(labels, logits, num_frames, blank))
#tf.Tensor([39.25209  37.818047 40.20242  39.07262 ])

#%%    
# Calculate gradient of logits

with tf.GradientTape() as tape:
    tape.watch(logits)  
    
    loss = CTC_HMM_vectorised_nested(labels, logits, num_frames, blank)

# Calculate gradient
grads = tape.gradient(loss, logits)
print(grads)
'''
