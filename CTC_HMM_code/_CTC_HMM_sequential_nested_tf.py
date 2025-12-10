#Sara Knopp, last modified on 06.08.2025

# CTC_HMM, not vectorised version, for-loops

import tensorflow as tf

# Uncomment and adjust the following lines if you want to run this script standalone
# (i.e., not imported by another script):
# Change the working directory to the project directory
#import os
#homefolder = "..."
#project_dir = os.path.join(homefolder, "Programs", "CTC_HMM_code")
#os.chdir(project_dir)

from _funcs_ctc_HMM_tf import(
    check_min_num_frames,
    _create_A,
    _create_UB,
    _create_initial_distribution,
    _label_expansion,
)

#%%

@tf.function
def compute_thresh_list_s(z, num_frames):
    """
    Computes a list of state index thresholds (s) for each time step t.

    These thresholds define the minimum label index s that can be active at
    each time step t in the forward matrix F.

    Args:
        z (int): Length of the expanded label sequence (including blanks).
        num_frames (int): Number of input frames (time steps).

    Returns:
        tf.Tensor: A 1D tensor of shape (num_frames,) containing minimum
                   active label indices (s) for each time step t.

    Notes:
        - This is used to mask out invalid transitions in the forward algorithm.
    """
    thresh_list_s = tf.TensorArray(dtype=tf.int32, size=num_frames)

    for t in tf.range(num_frames):
        # Compute the minimum label index s that can be active at time step t
        value = tf.math.floor(
            tf.cast(z - (2 * (num_frames - t) - 1) - 1, tf.float32)
        )
        value = tf.cast(value, tf.int32)

        # Ensure the threshold is not negative
        value_clipped = tf.maximum(value, 0)

        thresh_list_s = thresh_list_s.write(t, value_clipped)

    return thresh_list_s.stack()

@tf.function(reduce_retracing=True)
def CTC_HMM_nested_sequential(labels, logits, num_frames, blank):
    """
    Compute the CTC (Connectionist Temporal Classification) loss using a CRF
    (Conditional Random Field).

    Args:
        labels (tensor): Input sequences, where each label is an integer
                         (batchsize x label_length), dtype=int64.
        logits (tensor): Logits from the neural network 
                         (batchsize x num_labels x num_frames), dtype=float32. 
        num_frames (tensor/int): Number of frames in the input sequence, dtpe = int32.
        blank (int): The index representing the blank token in the set of labels.

    Returns:
        tensor: The CTC loss for each batch, dtype=float32.

    Notes:
        - num_labels: Number of possible labels (including the blank token)
    """
    batchsize, input_length = tf.shape(labels)[0], tf.shape(labels)[1]
    
    # Ensure that there are enough frames to cover the label sequences
    penalty = check_min_num_frames(labels, batchsize, num_frames)
    #if num_frames too small -> penalty > 0
    tf.debugging.assert_less_equal(penalty, 0.0, 
    message = "The number of frames is too small to complete the given label sequence.")

    # Expand the labels to include the blank token
    expanded_labels = _label_expansion(labels, blank) #(batchsize x 1 x z)
    z = len(expanded_labels[0])

    # Initial distribution (1 x z), pi = (1, 1, 0, ..., 0)
    pi = _create_initial_distribution(z)
    pi_log = tf.where(pi > 0, tf.math.log(pi), -1e6)

    # Construct UB (logits matrix for each frame) (batchsize x num_frames x z)
    UB = _create_UB(expanded_labels, logits)
    UB_log = tf.where(UB > 0, tf.math.log(UB), -1e6)                    
    
    # Create the transition matrix A (batchsize x z x z)
    A = _create_A(expanded_labels, blank)
    A_log = tf.where(A > 0, tf.math.log(A), -1e6)

    # Initialize the forward matrix F (batch_size, z, num_frames)
    F = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True,
                       element_shape=(None, None), clear_after_read=False)
    
    # Initialize the first column of F
    F = F.write(0, pi_log + UB_log[:, 0])  # Update F

    # Determine the threshold for s, s row index of F
    thresh_list_s = compute_thresh_list_s(z, num_frames)
        
    for t in tf.range(1, num_frames):  # Loop over time/column index
        prev_t = F.read(t - 1)  # Shape: [batchsize, z]
        
        # Initialize current column with -inf entries
        current_F_column = tf.fill([batchsize, z], tf.float32.min)  
        
        # Lower bound: Not enough time steps left to complete the sequence
        for s in tf.range(thresh_list_s[t], z): 
            if t < input_length + tf.math.floordiv((s - z - 1), 2): 
                continue
            else:
                # Apply log-sum-exp trick: log(sum(exp(x - max(x))) + max(x))
                result = tf.reduce_logsumexp(prev_t + A_log[:, s], axis=1) + UB_log[:, t, s]

                # Update current_F_column[:, s] with result
                scatter_indices = tf.stack([tf.range(batchsize), tf.fill([batchsize], s)], axis=1)  
                current_F_column = tf.tensor_scatter_nd_update(current_F_column, 
                                                               scatter_indices, result)
            F = F.write(t, current_F_column)
    
    # Compute the CTC loss for each batch
    last_frame = F.read(num_frames - 1)  # Shape: [batchsize, z]
    slice1 = last_frame[:, z - 1]  # Shape: [batchsize]
    slice2 = last_frame[:, z - 2]  # Shape: [batchsize]
    ctc_loss = -tf.math.reduce_logsumexp(tf.stack([slice1, slice2], axis=0), axis=0)
    
    return ctc_loss  

#%%
# small example
'''
labels = tf.constant([[0, 1], [1, 1], [0, 1]], dtype=tf.int64) 
batch_size = labels.shape[0]

num_labels = 3
num_frames = tf.constant(4, dtype=tf.int32)
logits_single = tf.constant([0.7, 0.1, 0.6, 0.4, 0.2, 0.3, 0.3, 0.8, 0.4, 0.1, 0.5, 0.2],
                    shape = (num_frames, num_labels))
logits = tf.tile(tf.expand_dims(logits_single, axis=0), [batch_size, 1, 1])

blank = 2

print(CTC_HMM_nested_sequential(labels, logits, num_frames, blank))
#[1.2743268 2.9117095 1.2743268]

#%%    
#small example 
# Set random seed for reproducibility
tf.random.set_seed(28102024)

H = 4  # Alphabet size (number of possible nucleotides: A, C, G, T)
blank = 0
num_labels = 5  # Number of labels [A, C, G, T, e]
batchsize = 4  # Batch size
num_frames = 50  
label_length = 20  # Length of input sequences

# Create a list of input sequences of different length
labels = tf.random.uniform([batchsize, label_length],
                           minval = 1, maxval = H+1, dtype = tf.int32)
# Create list of logits
logits_single = tf.random.uniform([num_frames, num_labels])
logits = tf.tile(tf.expand_dims(logits_single, axis=0), [batchsize, 1, 1])

print(CTC_HMM_nested_sequential(labels, logits, num_frames, blank))
#[39.25209  37.818047 40.20242  39.07262]

#%%    
# Calculate gradient of logits and num_frames 

with tf.GradientTape() as tape:
    tape.watch(logits)  # Watch pi, if not gradient is None
    
    loss = CTC_HMM_nested_sequential(labels, logits, num_frames, blank)

# Calculate gradient
grads = tape.gradient(loss, logits)
print(grads)
'''