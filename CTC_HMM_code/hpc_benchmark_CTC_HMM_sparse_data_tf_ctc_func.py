#Sara Knopp, last modified on 08.08.2025

# TensorFlow CTC loss function with sparse data

print('Start loading packages.')

import argparse
parser = argparse.ArgumentParser(description="Set GPU index and base project directory.")
parser.add_argument('--gpu_index', '-gpu', type=int, required=True, help='GPU Index to use')
parser.add_argument('--homefolder', '-hf', type=str, required=True, help='Base path to the project folder')
args = parser.parse_args()  # Read in arguments
# example: python3 hpc_pHMM_diag_nested_tf.py -gpu 1 -hf /home/s-saknop/Thesis/

import os
# Configure TensorFlow to allow GPU memory growth, to avoid allocating all GPU memory upfront
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# Limit GPU visibility to a single GPU for accurate usage and isolated resource allocation.
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

import tensorflow as tf
import sys
import numpy as np
import time
import GPUtil
import threading

homefolder = args.homefolder.rstrip("/")

# Change the working directory to the project directory
project_dir = os.path.join(homefolder, "Programs")
os.chdir(project_dir)  

if project_dir not in sys.path:
    sys.path.append(project_dir)
        
from _save_data import save_data 

subfolder = "CTC_HMM_code"
os.chdir(subfolder)  

from _generate_data_ctc import generate_data_ctc

#path in which data should be saved
save_dir = os.path.join(homefolder, "Programs", "results_data_plots",
    "batchsize_8_gradient", "CTC_HMM")

#%%
# Generate data
labels_list, logits_batch, label_length_list, logit_length_list, blank = generate_data_ctc(tf_ctc_loss_fct = True)

#%%
def gpu_ram_measure(gpu_list, ram_list, num_measure, stop_event):
    '''
    Function to measure GPU and RAM utilization periodically.
    
    This function monitors the GPU and RAM utilisation at a specified rate 
    (in measurements per second). It collects measurements 'num_measure' times 
    per second and appends the results to the provided lists for GPU and RAM.

    Args:
        gpu_list (list): A list to append GPU utilization measurements 
                         (in percentage).
        ram_list (list): A list to append RAM memory usage measurements 
                         (in MB).
        num_measure (int): The number of measurements per second 
                            (e.g., 10 for 10 times per second).
        stop_event (threading.Event): Event to stop the loop when set.

    Returns:
        tuple: A tuple of two lists:
            - gpu_list: A list of GPU utilization measurements with the 
                        format [input_size, utilization_percentage].
            - ram_list: A list of RAM memory usage measurements with the 
                        format [input_size, memory_used_in_MB].

    Note:
        The function continuously measures the GPU and RAM utilization 
        until the stop_event is set.
    '''
    while not stop_event.is_set():
        gpu = GPUtil.getGPUs()[args.gpu_index]  # Choose GPU from the available GPUs
        
        # Append current GPU load (in %) and memory used (in MB) to 
        # respective lists
        gpu_list.append([L, gpu.load * 100])  # GPU utilization in %
        ram_list.append([L, gpu.memoryUsed])  # RAM usage in MB

        # Reduce number of measurements for fast algs
        if num_measure == 5 and L in [500, 1000, 2000]:
            num_measure = 1  # Measure GPU and RAM every second
                    
        # Measure GPU and RAM utilization at the specified frequency
        time.sleep(1 / num_measure)
    
    return gpu_list, ram_list

num_of_repeat_fast_alg = 100  # Repeat algorithms several times
num_of_repeat_slow_alg = 4

num_measure_fast_alg = 5  # Measure GPU and RAM 5 times per second
num_measure_slow_alg = 1 # Measure GPU and RAM every second

#%%
def dense_to_sparse_preserve_zeros(dense, lengths):
    """
    Convert dense tensor to SparseTensor without treating zeros as padding.
    All values are preserved, including zeros.

    Args:
        dense (tf.Tensor): A 2D dense tensor of shape (batch_size, max_len), 
                           containing values including zeros.
        lengths (tf.Tensor): A 1D tensor of shape (batch_size,) specifying the 
                             valid length for each sequence in the batch.

    Returns:
        tf.SparseTensor: A sparse tensor containing only the valid (non-padded) 
                         values from `dense`, including zeros.

    """
    max_len = tf.shape(dense)[1]

    # Create mask for valid label positions
    mask = tf.sequence_mask(lengths, maxlen=max_len)  # shape: (batch, max_len)

    # Get indices of valid positions
    indices = tf.where(mask)

    # Gather values
    values = tf.gather_nd(dense, indices)

    # Define shape
    shape = tf.cast(tf.shape(dense), tf.int64)

    return tf.SparseTensor(indices=indices, values=values, dense_shape=shape)

#%%
print('Algorithm starts now.')

# Initialize arrays to store computational time, GPU, and RAM usage
comp_time_tf = np.zeros((len(labels_list), num_of_repeat_fast_alg), dtype=float)  
gpu_list_tf = []
ram_list_tf = [] 
results_tf = []
    
# Event for stopping the thread that monitors GPU and RAM usage
stop_event = threading.Event()
    
# Start a thread to monitor GPU and RAM during the execution of the algorithm
gpu_monitor_thread = threading.Thread(target=gpu_ram_measure, 
                                      args=(gpu_list_tf, ram_list_tf, 
                                            num_measure_fast_alg, stop_event))
gpu_monitor_thread.start()

try:
    # Iterate over all input sequences
    for i in range(len(labels_list)):
        # Length of the current input sequence
        L = len(labels_list[i][0])
        
        sparse_labels = dense_to_sparse_preserve_zeros(labels_list[i], 
                                                       label_length_list[i])
        
        # Repeat the algorithm multiple times for each input size
        for r in range(num_of_repeat_fast_alg):  
            # Start timer
            start = time.time()

            # Apply the algorithm and compute gradient
            with tf.GradientTape() as tape:
                tape.watch(logits_batch[i])  # Watch pi, if not gradient is None
                
                # Sparse labels
                CTC_loss = tf.nn.ctc_loss(
                    labels = sparse_labels,             #dim batch_size x max_label_seq_length
                    logits = logits_batch[i],             #dim batch_size x frames x num_labels
                    label_length = label_length_list[i], #dim batch_size,
                    logit_length = logit_length_list[i], #dim batch_size,
                    logits_time_major = False,   # if False, logits shape is [batch_size, frames, num_labels]. 
                    blank_index = blank)

            # Calculate gradient
            grads = tape.gradient(CTC_loss, logits_batch[i])

            # Stop timer
            end = time.time()

            # Calculate computational time for this run
            comp_time_tf[i, r] = end - start
            
            # Print progress message
            tf.print(f'Tensorflow CTC Loss, Input Length: {L}, Repetition {r + 1} of {num_of_repeat_fast_alg} completed.')
        #tf.print(grads)
        tf.print('CTC loss:', CTC_loss)
        results_tf.append(CTC_loss)
finally:
    # Stop the GPU and RAM monitoring thread
    stop_event.set()
    gpu_monitor_thread.join()  # Wait for the thread to finish 

# Save data
save_data(save_dir, comp_time_tf, gpu_list_tf, ram_list_tf, 'tf_sparse')

# Save results
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

algorithm_index = 'tf'
# Construct file paths for the computational time, GPU, and RAM data
results_file = os.path.join(save_dir, f'results_{algorithm_index}_tf_sparse.txt')

# Save results
try:
    with open(results_file, 'w') as comp_file:
        comp_file.write(f"{results_tf}")
    
    print(f"Data successfully saved for Algorithm {algorithm_index}:\n"
          f"Results: {results_file}\n")

except Exception as e:
    print(f"An error occurred while saving data for Algorithm {algorithm_index}: {e}")

#%%
# small example, compare loss values with dense and sparse data
'''
sparse_labels = dense_to_sparse_preserve_zeros(labels_list[i], label_length_list[i])


tf.nn.ctc_loss(
    labels = labels_list[i],             #dim batch_size x max_label_seq_length
    logits = logits_batch[i],            #dim batch_size x frames x num_labels
    label_length = label_length_list[i], #dim batch_size,
    logit_length = logit_length_list[i], #dim batch_size,
    logits_time_major = False,   # if False, logits shape is [batch_size, frames, num_labels]. 
    blank_index = blank)


tf.nn.ctc_loss(
    labels = sparse_labels,             #dim batch_size x max_label_seq_length
    logits = logits_batch[i],             #dim batch_size x frames x num_labels
    label_length = label_length_list[i], #dim batch_size,
    logit_length = logit_length_list[i], #dim batch_size,
    logits_time_major = False,   # if False, logits shape is [batch_size, frames, num_labels]. 
    blank_index = blank)
'''
