#Sara Knopp, last modified on 07.08.2025

# CTC HMM, sequential scan L = 20 000

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
from _CTC_HMM_sequential_scan_tf import CTC_HMM_sequential_scan

#path in which data should be saved
#save_dir = os.path.join(homefolder, "Programs", "results_data_plots",
#    "batchsize_8", "CTC_HMM")
save_dir = os.path.join(homefolder, "Programs", "results_data_plots",
    "batchsize_8_gradient", "CTC_HMM")
#%%
# Generate data
labels_list, logits_list, num_frames, blank = generate_data_ctc()

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

num_of_repeat_fast_alg = 11  # Repeat algorithms several times
num_of_repeat_slow_alg = 2

num_measure_fast_alg = 5  # Measure GPU and RAM 5 times per second
num_measure_slow_alg = 1 # Measure GPU and RAM every second

#%%
@tf.function
def compute_loss_and_grads(labels, logits, num_frames, blank):
    '''
    Function to compute the forward algorithm, negative log-likelihood loss, 
    and corresponding gradients for a given batch of input sequences.

    Args:
       labels (tensor): Input sequences, where each label is an integer
                        (batchsize x label_length), dtype=int64.
       logits (tensor): Logits from the neural network 
                        (batchsize x num_labels x num_frames), dtype=float32. 
       num_frames (tensor/int): Number of frames in the input sequence, dtpe = int32.
       blank (int): The index representing the blank token in the set of labels.

    Returns:
        CTC_loss (tf.Tensor): Scalar tensor representing the CTC loss of the batch.
        grads (tf.Tensor): Gradients of the loss with respect to logits.
    '''
    # Apply the algorithm and compute gradient
    with tf.GradientTape() as tape:
        tape.watch(logits)  # Watch logits, if not gradient is None
        
        # Apply the algorithm
        CTC_loss = CTC_HMM_sequential_scan(labels, logits, num_frames, blank)
    
    # Calculate gradient
    grads = tape.gradient(CTC_loss, logits)
    
    return CTC_loss, grads

#%%
print('Algorithm starts now.')

# Initialize arrays to store computational time, GPU, and RAM usage
comp_time_scan = np.zeros((len(labels_list), num_of_repeat_slow_alg), dtype=float)  
gpu_list_scan = []
ram_list_scan = [] 
results_scan = []

# Event for stopping the thread that monitors GPU and RAM usage
stop_event = threading.Event()

# Start a thread to monitor GPU and RAM during the execution of the algorithm
gpu_monitor_thread = threading.Thread(target=gpu_ram_measure, 
                                      args=(gpu_list_scan, ram_list_scan, 
                                            num_measure_slow_alg, stop_event))
gpu_monitor_thread.start()

try:
    # Iterate over all input sequences
    for i in [4]:  
        # Length of the current input sequence
        L = len(labels_list[i][0])
        
        # Repeat the algorithm multiple times for each input size
        for r in range(num_of_repeat_slow_alg):  
            # Start timer
            start = time.time()

            # Apply the algorithm and compute gradient
            CTC_loss, grads = compute_loss_and_grads(labels_list[i], logits_list[i], 
                                      num_frames[i], blank)
            _ = CTC_loss.numpy()  # Ensures all TF operations finish before stopping timer
            
            # Stop timer
            end = time.time()

            # Calculate computational time for this run
            comp_time_scan[i, r] = end - start
            
            # Print progress message
            tf.print(f'Scan CTC_HMM, Input Length: {L}, Repetition {r + 1} of {num_of_repeat_slow_alg} completed.')
        #tf.print(grads)
        tf.print('CTC loss:', CTC_loss)
        results_scan.append(CTC_loss)
        
finally:
    # Stop the GPU and RAM monitoring thread
    stop_event.set()
    gpu_monitor_thread.join()  # Wait for the thread to finish

# Save data
save_data(save_dir, comp_time_scan, gpu_list_scan, ram_list_scan, 'scan_long_seq_tf')


# Save results
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

algorithm_index = 'scan_long_seq'
# Construct file paths for the computational time, GPU, and RAM data
results_file = os.path.join(save_dir, f'results_{algorithm_index}_tf.txt')

# Save results
try:
    with open(results_file, 'w') as comp_file:
        comp_file.write(f"{results_scan}")
    
    print(f"Data successfully saved for Algorithm {algorithm_index}:\n"
          f"Results: {results_file}\n")

except Exception as e:
    print(f"An error occurred while saving data for Algorithm {algorithm_index}: {e}")

