#Sara Knopp, last modified on 30.07.2025

# forward alg, matrix multiplication without stabilisation

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
# Limit GPU visibility to GPU 0 for accurate usage and isolated resource allocation.
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

import numpy as np
import tensorflow as tf
import time
import GPUtil
import threading
import sys

homefolder = args.homefolder.rstrip("/")

# Change the working directory to the project directory
project_dir = os.path.join(homefolder, "Programs")
os.chdir(project_dir)  

if project_dir not in sys.path:
    sys.path.append(project_dir)
from _save_data import save_data 

benchmark_path = os.path.join(homefolder, "Programs", "benchmark_HMM_code")
os.chdir(benchmark_path)
if benchmark_path not in sys.path:
    sys.path.append(benchmark_path)
from _generate_data_hmm import _generate_data

sub_folder = os.path.join(homefolder, "Programs", "stability_investigations_code")
os.chdir(sub_folder)
from _matrix_mult_unstable_tf import matrix_mult_unstable

#path in which data should be saved
save_dir = os.path.join(homefolder, "Programs", "results_data_plots",
    "batchsize_8_gradient", "stability")
#%%
# Generate data
A, B, pi, input_batch_list, one_hot_batch_list = _generate_data()

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

        # Reduce number of measurements for slow algs
        if num_measure == 4 and L in [10000, 100000]:
            num_measure = 2  # Measure GPU and RAM twice per second
        if num_measure == 2 and L == 500000:
            num_measure = 1  # Measure GPU and RAM every second
        # Reduce number of measurements for fast algs
        if num_measure == 5 and L in [10000, 100000, 500000]:
            num_measure = 1  # Measure GPU and RAM every second
            
        # Measure GPU and RAM utilization at the specified frequency
        time.sleep(1 / num_measure)
    
    return gpu_list, ram_list

num_of_repeat_fast_alg = 11  # Repeat algorithms several times
num_of_repeat_slow_alg = 4

num_measure_fast_alg = 5  # Measure GPU and RAM 5 times per second
num_measure_slow_alg = 4  # Measure GPU and RAM 4 times per second

#%%
@tf.function
def compute_loss_and_grads(A, B, pi, U):
    '''
    Function to compute the forward algorithm, negative log-likelihood loss, 
    and corresponding gradients for a given batch of input sequences.

    Args:
        A (tf.Tensor): Transition matrix of shape (K, K), where K is the number 
                       of hidden states.
        B (tf.Tensor): Emission matrix of shape (K, M), where M is the number 
                       of observable symbols.
        pi (tf.Tensor): Initial state distribution vector of shape (K,).
        input_batch (tf.Tensor): A batch of one-hot encoded observation sequences 
                                 of shape (batch_size, M, L), where L is the 
                                 sequence length.

    Returns:
        alpha (tf.Tensor): Tensor of forward probabilities with shape 
                           (batch_size, K, L), computed using the scaled 
                           forward algorithm.
        loss (tf.Tensor): Scalar tensor representing the negative log-likelihood 
                          of the batch.
        grads (list of tf.Tensor): Gradients of the loss with respect to A, B, 
                                   and pi, in the same order.
    '''
    with tf.GradientTape() as tape:
        tape.watch(pi)  # Watch pi, if not gradient is None
        
        # Apply forward algorithm
        alpha = matrix_mult_unstable(A, B, pi, U)  # (batch x K x L)
        
        # Calculate Negative Log-Likelihood
        final_log_prob = tf.math.log(tf.reduce_sum(alpha[:, :, -1], axis=1))
        loss = -tf.reduce_mean(final_log_prob)
        
    grads = tape.gradient(loss, [A, B, pi])  # Gradient
    
    return alpha, loss, grads
#%%
print('Algorithm starts now.')

# Initialize arrays to store computational time, GPU, and RAM usage
comp_time_1 = np.zeros((len(one_hot_batch_list), num_of_repeat_fast_alg), dtype = float)  #-3
gpu_list_1 = []
ram_list_1 = [] 
    
# Event for stopping the thread that monitors GPU and RAM usage
stop_event = threading.Event()

# Start a thread to monitor GPU and RAM during the execution of the algorithm
gpu_monitor_thread = threading.Thread(target = gpu_ram_measure, 
                                      args = (gpu_list_1, ram_list_1, num_measure_fast_alg, stop_event))
gpu_monitor_thread.start()

try:
    # Iterate over all input sequences
    for i in range(len(one_hot_batch_list)):  #-3 just length 10, 100, 1000
        # Length of the current input sequence
        L = len(one_hot_batch_list[i][0])
        
        # Repeat the algorithm multiple times for each input size
        for r in range(num_of_repeat_fast_alg):  
            # Start timer
            start = time.time()

            # Apply the algorithm and compute gradient
            alpha, loss, grads = compute_loss_and_grads(A, B, pi, 
                                                        one_hot_batch_list[i])
            _ = loss.numpy()  # Ensures all TF operations finish before stopping timer
            
            # Stop timer
            end = time.time()

            # Calculate computational time for this run
            comp_time_1[i, r] = end - start
            
            # Print progress message
            tf.print(f'Matrix Mult unstable, Input Length: {L}, Repetition {r + 1} of {num_of_repeat_fast_alg} completed.')
        #tf.print(alpha)
        #tf.print(grads)
        tf.print(tf.argmax(alpha, axis=1))
        tf.print('Loss:', loss)
finally:
    # Stop the GPU and RAM monitoring thread
    stop_event.set()
    gpu_monitor_thread.join()  # Wait for the thread to finish

save_data(save_dir, comp_time_1, gpu_list_1, ram_list_1, 'matrix_mult_unstable_tf')