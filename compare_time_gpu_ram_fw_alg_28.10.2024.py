#Sara Knopp, last modified on 25.11.2024

#compare 4 variants of forward-algorithms with regard to computational time, 
#maximal used RAM and mean GPU-utilization, used input sizes (10, 100, 1000, 
# 10 000, 100 000, 500 000)
#algorithm 0: sequential with for-loops
#algorithm 1: sequential with matrixmulitplication
#algorithm 2: parallel with scan_associative
#algorithm 3: parallel with scan_associative, but utilising the sparsity
#  alg 3 version 1: use a matrix with predecessors
#  alg 3 version 2: use a list of predecessors


#result:
#computational time: for nearly all algs it can be observed that the comp time
#  increases with increasing sample size.
#  alg 0 is much slower for large input sizes than all other algs
#  with a mean comp time of around 25 000 secs for the largest sample size.
#  Alg 1 and alg 2 are both fast even for the largest input size. Alg 1 needs
#  around 10 sec for the maximal used input size. The comp time of alg 2 stays
#  nearly constant for all input sizes and is below 1 sec
#  alg 3.1 and alg 3.2 need the same comp time for all input sizes and for the
#  max input size they take around 1700 secs

#GPU-utilization: It can be observed that for the smallest sample size 
# alg 3.1, 2 and 0 have a quite high mean GPU-utilization with 23%, 19%
# and 7%. Afterwards it decreases and stays at around 1.5% from input 
# size 1000. Algorithm 1 has at input size 10 a mean GPU-utilization
# around 1.5% and 18% at input size 100 000. For the remaning input
# sizes it is around 1%. Alg 3.2 has a nearly constant mean GPU-
# utilization around 1%, but for input size 10 it is 0%.

# Attention: The measurements are subject to fluctuations. If you run all 
# algorithms always run for the same input size, for example [10, 100, 1000], 
# the results vary slightly -> ok
# If you run all algorithms for inputs up to size 1000, i.e. [10, 100, 1000] 
# and up to input size 500 000, i.e. [10, 100, 1000, 10 000, 100 000, 500 000] 
# the results differ considerably for the values [10, 100, 1000] -> reason?

#maximum used RAM: the maximum used RAM of all algs is constant for all input
#  sizes and is between 6900 and 7000 MiB




#structure of this code: define variables for algorithms, define algorithms,
#                        investigation of computational time, RAM and GPU-
#                        utilization



import sys                       
import numpy as np
import tensorflow as tf
import time                     
import matplotlib.pyplot as plt  
import GPUtil
import threading

from scipy.special import softmax
from tensorflow_probability.python.math.scan_associative import scan_associative

#import function for creating a ragged tensor
sys.path.append("Thesis/")
#sys.path.append("C:/Users/Lenovo/Documents/Sara/Uni/Master/5. Semester/Masterarbeit/Programme/")
from _create_ragged_tensor import _create_ragged_tensor 



#%%
#first, define all variables used for all algorithms: A, B, pi, U, K, A_ragged

np.random.seed(28102024)   # have some random components

#Create a K x K matrix A, K = 15, with transitionprobabilities. Order of the states: 
#IR Intron-0 Intron-1 Intron-2 Exon-0 Exon-1 Exon-2 
#Start DSS-0 DSS-1 DSS-2 ASS-0 ASS-1 ASS-2 Stop (15 states)

K = 15    #number of states

A = np.zeros((K, K)) # 15x15 matrix filled with zeros

#insert transitionprb into transitionmatrix A after figure 2 from Tiberius paper
A[0, 0] = 0.98   # transitionprb from IR to IR    
A[0, 7] = 0.02   # IR to Start    

A[1, 1] = 0.98   # Intron-0 to Intron-0   
A[1, 12] = 0.02  # Intron-0 to ASS-1

A[2, 2] = 0.98   # Intron-1 to Intron-1    
A[2, 13] = 0.02  # Intron-1 to ASS-2

A[3, 3] = 0.98   # Intron-2 to Intron-2
A[3, 11] = 0.02  # Intron-2 to ASS-0

A[4, 5] = 0.98   # Exon-0 to Exon-1      
A[4, 9] = 0.02   # Exon-0 to DSS-1              

A[5, 6] = 0.96   # Exon-1 to Exon-2      
A[5, 10] = 0.02  # Exon-1 to DSS-2
A[5, 14] = 0.02  # Exon-1 to Stop

A[6, 4] = 0.98   # Exon-2 to Exon-0   
A[6, 8] = 0.02   # Exon-2 to DSS-0 

A[7, 5] = 1      # Start to Exon-1

A[8, 1] = 1      # DSS-0 to Intron-0     
 
A[9, 2] = 1      # DSS-1 to ASS-2

A[10, 3] = 1     # DSS-2 to Intron-2

A[11, 5] = 1     # ASS-0 to Exon-1

A[12, 6] = 1     # ASS-1 to Exon-2

A[13, 4] = 1     # ASS-2 to Exon-0

A[14, 0] = 1     # Stop to IR


H = 4     # alphabet size (A T C G)

#Emissionmatrix B with random prbs
B_rand = np.random.randint(100, size = (K, 4)) / 100   
B = softmax(B_rand, axis = 1)    #sum of rows = 1 now


#initial distribution
pi = np.array((1, 0, 0, 0, 0, 
               0, 0, 0, 0, 0, 
               0, 0, 0, 0, 0), dtype = 'f')   #start in IR


#input sequence: nucleotide sequence ACGT
#for one-hot encoding: A: 0, C: 1, G: 2, T: 3 
u = tf.constant([1, 2, 2, 3, 0, 1])

#one_hot encoding of input sequence
U = tf.one_hot(u, H)    #dim: length input x 4



#%%
#Algorithm 0
#forward-algorithm sequential, for-loops

#define sequential forward algorithm with for-loops
#input: Transitionmatrix A (K x K), Emissionmatrix B (K x 4), 
#       initial distribution pi (1 x K), input sequence u (L x 1) 
#       one-hot encoded input sequence U (L x 4)
#output: matrix F with probabilities (K x L)
#K: number of states, L: length of input sequence

def forward_sequential_loops(A, B, pi, u, U):
    
    #create predecessor list 
    #pred_list is a list of lists. the j-th list in pred_list contains the
    #predecessors of j
    pred_list = []
    for j in range(A.shape[1]):      #go through all columns of A
        pred_j = []                  #save predecessors of j in here        
        for i in range(A.shape[0]):  #go through all rows of A
            if A[i,j] != 0:          #if this entry of A is non-zero, make list element
                pred_j.append(i)
        pred_list.append(pred_j)
    
    # precompute emission probs
    UB = np.matmul(U, B.transpose())
    
    L =  U.shape[0]   #length of input sequence
    K =  A.shape[0]   #number of states
    
    alpha = np.zeros((L, K))
    
    #initialization
    alpha[0] = pi * UB[0]
    
    for t in range(1, L):
        for q in range(K):
            for j in range(len(pred_list)):
                alpha[t, q] += A[j, q] * alpha[(t-1), j] * B[q, u[t]]
    
    return alpha


#F_alg0 = forward_sequential_loops(A, B, pi, u, U)
#with np.printoptions(precision = 9, suppress = True, linewidth = 100):
#    print (F_alg0.transpose())



#%%
#algorithm 1
#forward-Algorithm sequential matrixmultiplication (from E-Mail)


#define sequential forward algorithm
#input: Transitionmatrix A (K x K), Emissionmatrix B (K x 4), 
#       initial distribution pi (1 x K), one-hot encoded input sequence U (L x 4)
#output: matrix F with probabilities (K x L)
#K: number of states, L: length of input sequence
def forward_sequential(A, B, pi, U):
    
    # precompute emission probs
    UB = np.matmul(U, B.transpose())
    
    L =  U.shape[0]   #length of input sequence
    K =  A.shape[0]   #number of states
    
    F = np.zeros((L, K))
    
    #initialization
    F[0] = pi * UB[0]
    
    #matrixmultilpication
    for t in range(1, L):
        F[t] = np.matmul(F[t-1], A) * UB[t]
        
    return F 


#F_alg1 = forward_sequential(A, B, pi, U)
#with np.printoptions(precision = 9, suppress = True, linewidth = 100):
#    print (F_alg1.transpose())



#%%
#algorithm 2: parallel with scan_associative
#Use Algorithm "parallel_fw" 


#forward-algorithm computet in parallel using scan_associative 
#input: Transitionmatrix A (K x K), Emissionmatrix B (K x 4), 
#       initial distribution pi (1 x K), one-hot encoded input sequence U (L x 4)
#output: matrix F with probabilities (K x L)
#K: number of states, L: length of input sequence
def forward_parallel_scan_associative(A, B, pi, U):
    
    #number of states
    K =  A.shape[0]
    
    # make K x K transition matrix into 0-th state
    Pi_alg2 = tf.repeat(tf.expand_dims(pi, 0), K, axis = 0)
    
    # precompute emission probs
    UB = tf.matmul(U, B, transpose_b = True)  #B ist transposed before multiplication
    # U: length input seq x 4,    B: K x 4,   UB: length input seq x K
    
    # compute inputs E = (a0, a1, a2, ..., aL) to assoc scan
    E = tf.einsum('ab,tb->tab', A, UB[1:,:])
    
    #UB[1:,:] (length input seq -1) x K, UB without first row, why?
    #A: K x K
    #ab,tb->tab means: A_ab * UB_tb = E_tab
    #E: (length input seq -1) x K x K = 5 x 15 x 15
    
    # prepend initial state to E
    E_complete = tf.concat([tf.expand_dims(Pi_alg2, 0) * UB[0, :], E], axis = 0)
    
    # compute forward probabilities in parallel (Särkkä and Garcı́a-Fernández)
    F_alg2_all = scan_associative(tf.matmul, E_complete, axis = 0)
    
    #scan_associative: "Perform a scan (= prefix sum alg? -> 
    #for parallel computing) with an associative binary operation, in parallel."
    #first argument: associative binary operation
    #second argument: Tensor
    #axis: "axis along which to perform the scan"
    
    F_alg2 = F_alg2_all[:,0,:]   # first state has no predecessor 
    #why is this our final matrix?
    #chat GPT: F_alg2 is your final matrix of forward probabilities because it summarizes 
    #the likelihood of being in each state at each time step, given the observed sequence. 
    #Each entry in F_alg2 provides the total probability of transitioning into a state 
    #at a particular time step, factoring in both the transition and emission probabilities 
    #according to your model.

    return F_alg2


#F_alg2 = forward_parallel_scan_associative(A, B, pi, U)
#print ("forward probs:")
#with np.printoptions(precision = 9, suppress = True, linewidth = 100):
#    print (F_alg2.numpy().transpose())



#%%
#algorithm 3.1 
#parallel, scan_associative, but utilise sparsity of Transitionmatrix A
#with matrix of predecessors (ragged tensor)


#forward-algorithm computet in parallel using scan_associative and utilise sparsity of A
#input: Transitionmatrix A (K x K), Emissionmatrix B (K x 4), 
#       initial distribution pi (1 x K), one-hot encoded input sequence U (L x 4)
#output: matrix F with probabilities (K x L)
#K: number of states, L: length of input sequence
def forward_parallel_scan_associative_sparse(A, B, pi, U):
    
    #number of states
    K =  A.shape[0]
    
    #ragged tensor of A
    A_ragged = _create_ragged_tensor(A)
    
    # make KxK transition matrix into 0-th state
    Pi_alg3 = tf.repeat(tf.expand_dims(pi, 0), K, axis = 0)
    
    # precompute emission probs
    UB = tf.matmul(U, B, transpose_b = True)  
    
    
    # compute inputs E = (a0, a1, a2, ..., aL) to assoc scan
    #instead of E = tf.einsum('ab,tb->tab', A, UB[1:,:]) use sparsity of A:
    
    #transform UB so that matrix multiplication is possible
    #each row of UB is transformed to a diagonal matrix
    #get dimensions of UB
    X, Y = UB.shape

    #initialise UB_large
    UB_large = np.zeros((X, Y, Y)) #X-1

    for i in range(X):  #X-1
        for j in range(Y):
            UB_large[i, j, j] = UB[i, j]   #i+1
    
    #create a sparse Tensor of A using the ragged tensor of A 
    #save indices and values in here
    indices = []
    values = []

    for i in range(A_ragged.shape[0]):     #go through all rows of ragged tensor of A
        for j in range(len(A_ragged[i])):  #go through all elements of row i of ragged tensor of A
            
            #position of an entry that is non-zero
            indices.append([A_ragged[i][j], i])   
            
            #value of that non-zero entry
            values.append(A[tf.cast(A_ragged[i][j], dtype = tf.int32), i])      
            
    #convert for SparseTensor
    indices = np.array(indices)
    values = np.array(values)

    #shape of A for SparseTensor
    shape = A.shape

    #create Sparse Tensor
    A_sparse = tf.sparse.SparseTensor(indices = indices, values = values, 
                                      dense_shape = shape)

    #matrixmultiplication that replaces einsum()
    E_sparse = []
    for k in range(UB_large.shape[0]-1):  #create all k matrices 
        E_sparse.append(tf.sparse.sparse_dense_matmul(A_sparse, UB_large[k+1]))
    
    E_sparse_stack = tf.cast(tf.stack(E_sparse, axis = 0), dtype=tf.float32)
    
    # prepend initial state to E
    E_sparse_complete = tf.concat([tf.expand_dims(Pi_alg3, 0) * UB[0, :], 
                                   E_sparse_stack], axis = 0)
       
    # compute forward probabilities in parallel (Särkkä and Garcı́a-Fernández)
    F_alg3_all = scan_associative(tf.matmul, E_sparse_complete, axis = 0)
    
    F_alg3 = F_alg3_all[:,0,:] # first state has no predecessor 

    return F_alg3


#F_alg3_1 = forward_parallel_scan_associative_sparse(A, A_ragged, B, pi, U)
#print ("forward probs:")
#with np.printoptions(precision = 9, suppress = True, linewidth = 100):
#    print (F_alg3_1.numpy().transpose())




#%%
#algorithm 3.2 
#parallel, scan_associative, but utilise sparsity of Transitionmatrix A
#with list of predecessors


#forward-algorithm computet in parallel using scan_associative and utilise sparsity of A
#input: Transitionmatrix A (K x K), Emissionmatrix B (K x 4), 
#       initial distribution pi (1 x K), one-hot encoded input sequence U (L x 4)
#output: matrix F with probabilities (K x L)
#K: number of states, L: length of input sequence
def forward_parallel_scan_associative_sparse_list(A, B, pi, U):
    
    #number of states
    K =  A.shape[0]
    
    #create predecessor list of A, list containing lists
    #structure of pred_list elements: [node, predecessor of this node, prb]
    A_t = np.transpose(A)
    pred_list = []

    for i in range(A_t.shape[0]):      #go through all coloums of A
        for j in range(A_t.shape[1]):  #go through all rows of A
            if A_t[i,j] != 0:          #if this entry of A is non-zero, make list element
               pred_list.append([i, j, A_t[i,j]])
    
    # make KxK transition matrix into 0-th state
    Pi_alg3 = tf.repeat(tf.expand_dims(pi, 0), K, axis = 0)
    
    # precompute emission probs
    UB = tf.matmul(U, B, transpose_b = True)  
    
    # compute inputs E = (a0, a1, a2, ..., aL) to assoc scan
    #instead of E = tf.einsum('ab,tb->tab', A, UB[1:,:]) use sparsity of A:
    
    #transform UB so that matrix multiplication is possible
    #each row of UB is transformed to a diagonal matrix
    #get dimensions of UB
    X, Y = UB.shape

    #initialise UB_large
    UB_large = np.zeros((X, Y, Y)) 

    for i in range(X):  
        for j in range(Y):
            UB_large[i, j, j] = UB[i, j]   
    
    #create a sparse Tensor of A using the ragged tensor of A 
    #save indices and values in here
    indices = []
    values = []

    for i in range(len(pred_list)):  #go through all elements of the list of predecessors
            
            #position of an entry that is non-zero
            indices.append([pred_list[i][1], pred_list[i][0]])   
            
            #value of that non-zero entry
            values.append(pred_list[i][2])
            
    #convert for SparseTensor
    indices = np.array(indices)
    values = np.array(values)

    #shape of A for SparseTensor
    shape = A.shape

    #create Sparse Tensor
    A_sparse = tf.sparse.SparseTensor(indices = indices, values = values, 
                                      dense_shape = shape)

    #matrixmultiplication that replaces einsum()
    E_sparse = []
    for k in range(UB_large.shape[0]-1):  #create all k matrices 
        E_sparse.append(tf.sparse.sparse_dense_matmul(A_sparse, UB_large[k+1]))
    
    E_sparse_stack = tf.cast(tf.stack(E_sparse, axis = 0), dtype=tf.float32)
    
    # prepend initial state to E
    E_sparse_complete = tf.concat([tf.expand_dims(Pi_alg3, 0) * UB[0, :], 
                                   E_sparse_stack], axis = 0)

    # compute forward probabilities in parallel (Särkkä and Garcı́a-Fernández)
    F_alg3_all = scan_associative(tf.matmul, E_sparse_complete, axis = 0)
    
    F_alg3 = F_alg3_all[:,0,:] # first state has no predecessor 

    return F_alg3


#F_alg3_2 = forward_parallel_scan_associative_sparse_list(A, B, pi, U)
#print ("forward probs:")
#with np.printoptions(precision = 9, suppress = True, linewidth = 100):
#    print (F_alg3_2.numpy().transpose())




#%%
#compare results of the algorithms, should be similar

#look at the differences between the results of each alg
#F_alg0 - F_alg1    #really small differences
#F_alg0 - F_alg2    #slightly different
#F_alg0 - F_alg3    #slightly different

#F_alg2 - F_alg3    #no differences




#%%
#create input sequences with different length
np.random.seed(28102024)

#for one-hot encoding: A: 0, C: 1, G: 2, T: 3 
H = 4
#input length L: 10, 100, 1000, 10 000, 100 000, 500 000
u_10 = tf.reshape(np.random.randint(H, size=(1, 10)), (10, ))  
u_100 = tf.reshape(np.random.randint(H, size=(1, 100)), (100, ))
u_1000 = tf.reshape(np.random.randint(H, size=(1, 1000)), (1000, ))
u_10_000 = tf.reshape(np.random.randint(H, size=(1, 10000)), (10000, ))
u_100_000 = tf.reshape(np.random.randint(H, size=(1, 100000)), (100000, ))
u_500_000 = tf.reshape(np.random.randint(H, size=(1, 500000)), (500000, ))
#reshape for function, input u must have dim L x 1

#u
#input_list = [u_10, u_100, u_1000]
input_list = [u_10, u_100, u_1000, u_10_000, u_100_000, u_500_000] 


#one_hot encoding of input sequence
U_10 = tf.reshape(tf.one_hot(u_10, H), (10, H))    #dim: L x 4
U_100 = tf.reshape(tf.one_hot(u_100, H) , (100, H)) 
U_1000 = tf.reshape(tf.one_hot(u_1000, H), (1000, H)) 
U_10_000 = tf.reshape(tf.one_hot(u_10_000, H), (10000, H))    
U_100_000 = tf.reshape(tf.one_hot(u_100_000, H), (100000, H))    
U_500_000 = tf.reshape(tf.one_hot(u_500_000, H), (500000, H))

#U
#one_hot_list = [U_10, U_100, U_1000]
one_hot_list = [U_10, U_100, U_1000, U_10_000, U_100_000, U_500_000] 





#%%
#determine computational time, used RAM and GPU-utilization for each algorithm
#using the generated input sequences

num_of_repeat = 3   #repeat each algorithm several times   


#function for measuring RAM and GPU-utilization 10 times per second
#input: gpu_list and ram_list: lists to which measurments are added
#       L: int, size of the current input
#output: gpu_list and ram_list with GPU and RAM measurements
def get_gpu_and_ram(gpu_list, ram_list):
    while not stop_event.is_set():
        
        #select GPU in each round of while, otherwise the measurements are incorrect
        gpus = GPUtil.getGPUs()
        gpu = gpus[0]

        #add current RAM and GPU utilization to lists
        gpu_list.append([L, (gpu.load * 100)])
        ram_list.append([L, gpu.memoryUsed])
    
        #measure RAM and GPU-utilization 10 times per second 
        time.sleep(0.01)
    return gpu_list, ram_list

#same function, but measures GPU and RAM 1000 times per second
def get_gpu_and_ram_often(gpu_list, ram_list):
    while not stop_event.is_set():
        
        #select GPU in each round of while, otherwise the measurements are incorrect
        gpus = GPUtil.getGPUs()
        gpu = gpus[0]

        #add current RAM and GPU utilization to lists
        gpu_list.append([L, (gpu.load * 100)])
        ram_list.append([L, gpu.memoryUsed])
    
        #measure RAM and GPU-utilization 10 times each second 
        time.sleep(0.0001)
    return gpu_list, ram_list



    
#algorithm 0
#save computational time, RAM and GPU-utilization in here
comp_time_0 = np.zeros((len(one_hot_list), num_of_repeat), dtype = float)  
#row 0 contains the comp time of sample size 10 and its 5 repetitions, ect
gpu_list_0 = []
ram_list_0 = [] 

#for stopping the thread
stop_event = threading.Event()

#start thread for measuring RAM and GPU during the algorithm is running
gpu_monitor_thread = threading.Thread(target = get_gpu_and_ram, 
                                      args = (gpu_list_0, ram_list_0))
gpu_monitor_thread.start()
    
for i in range(len(one_hot_list)):  #go through all input sequences
    #length of current input sequence    
    L = len(one_hot_list[i])
    
    for r in range(num_of_repeat):  #repeat the alg several times for each input size      
        #start timer
        start = time.time()

        #apply algorithm
        F_alg0 = forward_sequential_loops(A, B, pi, input_list[i], one_hot_list[i])
        
        # stop timer 
        end = time.time()

        #compute computational time of this run
        comp_time_0[i, r] = end - start
        
        print('Algorithm 0, Sample Size', len(one_hot_list[i]), 
              'repetition', r + 1, 'of', num_of_repeat, 'completed.')
    #print(gpu_list_0, ram_list_0, comp_time_0)
    
#stop the thread so that measuring of RAM and GPU stops
stop_event.set()


#open .txt files for saving data 
with open('comp_time_0.txt', 'w') as comp_file, \
     open('gpu_list_0.txt', 'w') as gpu_file, \
     open('ram_list_0.txt', 'w') as ram_file:
    
         #save computational time
    comp_file.write(f"{comp_time_0}")
    
    #save GPU-utilization
    for gpu_usage in gpu_list_0:
        gpu_file.write(f"{gpu_usage}\n")
    
    #save used RAM
    for ram_usage in ram_list_0:
        ram_file.write(f"{ram_usage}\n")
     



#algorithm 1
#save computational time, RAM and GPU-utilization in here
comp_time_1 = np.zeros((len(one_hot_list), num_of_repeat), dtype = float)  
#row 0 contains the comp time of sample size 10 and its 5 repetitions, ect
gpu_list_1 = []
ram_list_1 = [] 

#for stopping the thread
stop_event = threading.Event()

#start thread for measuring RAM and GPU during the algorithm is running
gpu_monitor_thread = threading.Thread(target = get_gpu_and_ram_often, 
                                      args = (gpu_list_1, ram_list_1))
gpu_monitor_thread.start()
 
for i in range(len(one_hot_list)):  #go through all input sequences
    #length of current input sequence    
    L = len(one_hot_list[i])
    
    for r in range(num_of_repeat):  #repeat the alg several times for each input size      
        #start timer
        start = time.time()

        #apply algorithm
        F_alg1 = forward_sequential(A, B, pi, one_hot_list[i])
        
        # stop timer 
        end = time.time()
        
        #compute computational time of this run
        comp_time_1[i, r] = end - start
        
        print('Algorithm 1, Sample Size', len(one_hot_list[i]), 
              'repetition', r + 1, 'of', num_of_repeat, 'completed.')
    #print(gpu_list_1, ram_list_1, comp_time_1)
    
#stop the thread so that measuring of RAM and GPU stops
stop_event.set()


#open .txt files for saving data 
with open('comp_time_1.txt', 'w') as comp_file, \
     open('gpu_list_1.txt', 'w') as gpu_file, \
     open('ram_list_1.txt', 'w') as ram_file:

    #save computational time
    comp_file.write(f"{comp_time_1}")
    
    #save GPU-utilization
    for gpu_usage in gpu_list_1:
        gpu_file.write(f"{gpu_usage}\n")
    
    #save used RAM
    for ram_usage in ram_list_1:
        ram_file.write(f"{ram_usage}\n")




#algorithm 2
#save computational time, RAM and GPU-utilization in here
comp_time_2 = np.zeros((len(one_hot_list), num_of_repeat), dtype = float)  
#row 0 contains the comp time of sample size 10 and its 5 repetitions, ect
gpu_list_2 = []
ram_list_2 = [] 

#for stopping the thread
stop_event = threading.Event()

#start thread for measuring RAM and GPU during the algorithm is running
gpu_monitor_thread = threading.Thread(target = get_gpu_and_ram_often, 
                                      args = (gpu_list_2, ram_list_2))
gpu_monitor_thread.start()
         
for i in range(len(one_hot_list)):  #go through all input sequences
    #length of current input sequence    
    L = len(one_hot_list[i])
    
    for r in range(num_of_repeat):  #repeat the alg several times for each input size      
        #start timer
        start = time.time()

        #apply algorithm
        F_alg2 = forward_parallel_scan_associative(A, B, pi, one_hot_list[i])
        
        # stop timer 
        end = time.time()
        
        #compute computational time of this run
        comp_time_2[i, r] = end - start
        
        print('Algorithm 2, Sample Size', len(one_hot_list[i]), 
              'repetition', r + 1, 'of', num_of_repeat, 'completed.')
    #print(gpu_list_2, ram_list_2, comp_time_2)
    
#stop the thread so that measuring of RAM and GPU stops
stop_event.set()
            

#open .txt files for saving data
with open('comp_time_2.txt', 'w') as comp_file, \
     open('gpu_list_2.txt', 'w') as gpu_file, \
     open('ram_list_2.txt', 'w') as ram_file:        

    #save computational time
    comp_file.write(f"{comp_time_2}")
    
    #save GPU-utilization
    for gpu_usage in gpu_list_2:
        gpu_file.write(f"{gpu_usage}\n")
    
    #save used RAM
    for ram_usage in ram_list_2:
        ram_file.write(f"{ram_usage}\n")



        
#algorithm 3.1
#save computational time, RAM and GPU-utilization in here
comp_time_3_1 = np.zeros((len(one_hot_list), num_of_repeat), dtype = float)  
#row 0 contains the comp time of sample size 10 and its 5 repetitions, ect
gpu_list_3_1 = []
ram_list_3_1 = [] 

#for stopping the thread
stop_event = threading.Event()

#start thread for measuring RAM and GPU during the algorithm is running
gpu_monitor_thread = threading.Thread(target = get_gpu_and_ram_often, 
                                      args = (gpu_list_3_1, ram_list_3_1))
gpu_monitor_thread.start()
      
for i in range(len(one_hot_list)):  #go through all input sequences
    #length of current input sequence    
    L = len(one_hot_list[i])
    
    for r in range(num_of_repeat):  #repeat the alg several times for each input size      
        #start timer
        start = time.time()

        #apply algorithm
        F_alg3_1 = forward_parallel_scan_associative_sparse(A, B, pi, one_hot_list[i])
        
        # stop timer 
        end = time.time()
        
        #compute computational time of this run
        comp_time_3_1[i, r] = end - start
        
        print('Algorithm 3.1, Sample Size', len(one_hot_list[i]), 
              'repetition', r + 1, 'of', num_of_repeat, 'completed.')
    #print(gpu_list_3_1, ram_list_3_1, comp_time_3_1)
    
#stop the thread so that measuring of RAM and GPU stops
stop_event.set()


#open .txt files for saving data after each input size
with open('comp_time_3_1.txt', 'w') as comp_file, \
     open('gpu_list_3_1.txt', 'w') as gpu_file, \
     open('ram_list_3_1.txt', 'w') as ram_file:

    #save computational time
    comp_file.write(f"{comp_time_3_1}")
    
    #save GPU-utilization
    for gpu_usage in gpu_list_3_1:
        gpu_file.write(f"{gpu_usage}\n")
    
    #save used RAM
    for ram_usage in ram_list_3_1:
        ram_file.write(f"{ram_usage}\n")


        

#algorithm 3.2
#save computational time, RAM and GPU-utilization in here
comp_time_3_2 = np.zeros((len(one_hot_list), num_of_repeat), dtype = float)  
#row 0 contains the comp time of sample size 10 and its 5 repetitions, ect
gpu_list_3_2 = []
ram_list_3_2 = [] 

#for stopping the thread
stop_event = threading.Event()

#start thread for measuring RAM and GPU during the algorithm is running
gpu_monitor_thread = threading.Thread(target = get_gpu_and_ram_often, 
                                      args = (gpu_list_3_2, ram_list_3_2))
gpu_monitor_thread.start()
       
for i in range(len(one_hot_list)):  #go through all input sequences
    #length of current input sequence    
    L = len(one_hot_list[i])
    
    for r in range(num_of_repeat):  #repeat the alg several times for each input size      
        #start timer
        start = time.time()

        #apply algorithm
        F_alg3_2 = forward_parallel_scan_associative_sparse_list(A, B, pi, one_hot_list[i])

        # stop timer 
        end = time.time()
        
        #compute computational time of this run
        comp_time_3_2[i, r] = end - start
        
        print('Algorithm 3.2, Sample Size', len(one_hot_list[i]), 
              'repetition', r + 1, 'of', num_of_repeat, 'completed.')
        #print(gpu_list_3_2, ram_list_3_2, comp_time_3_2)

#stop the thread so that measuring of RAM and GPU stops
stop_event.set()


#open .txt files for saving data after each input size
with open('comp_time_3_2.txt', 'w') as comp_file, \
     open('gpu_list_3_2.txt', 'w') as gpu_file, \
     open('ram_list_3_2.txt', 'w') as ram_file:            

    #save computational time
    comp_file.write(f"{comp_time_3_2}")
    
    #save GPU-utilization
    for gpu_usage in gpu_list_3_2:
        gpu_file.write(f"{gpu_usage}\n")
    
    #save used RAM
    for ram_usage in ram_list_3_2:
        ram_file.write(f"{ram_usage}\n")





#%%
#prepare RAM and GPU-utilization measures for plotting

#function which prepares data from RAM and GPU-utilization measuring for plotting
#input: gpu_data: list of list, each list as two values. the first is the input 
#       size, the second the measurement   
#       ram_data: as gpu_data
#output: mean_gpu: list of the mean of the GPU-utilisation per input size
#        max_ram: list of the maximum of used RAM per input size
#        first entry of these lists corresponds to input size 10, 2nd to 100, ect
def _prep_ram_and_gpu_data(gpu_list, ram_list):
    #dicts for measurments
    dict_gpu_data = {}
    dict_ram_data = {}

    #prepare gpu_data
    for input_size, measurement in gpu_list:   #go through measured GPU values
        if input_size not in dict_gpu_data: #check, if this size already exists in dict_gpu_data 
            
            dict_gpu_data[input_size] = [] #if not create empty list for this input size
        
        dict_gpu_data[input_size].append(measurement) #add measured GPU-utilization
    
    #prepare ram_data
    for input_size, measurement in ram_list:   #go through measured GPU values
        if input_size not in dict_ram_data: #check, if this size already exists in dict dict_gpu_data 
            
            dict_ram_data[input_size] = [] #if not create empty list for this input size
        
        dict_ram_data[input_size].append(measurement) #add measured GPU-utilization
    
    #compute mean of GPU-utilization and max of RAM
    mean_gpu = []
    for input_size, measurements in dict_gpu_data.items():
        mean_gpu.append(np.mean(measurements))
    
    max_ram = []
    for input_size, measurements in dict_ram_data.items():
        max_ram.append(np.mean(measurements))

    return mean_gpu, max_ram


#prepare data for plotting, mean of GPU-utilization and maximum used RAM
[mean_gpu_0, max_ram_0] = _prep_ram_and_gpu_data(gpu_list_0, ram_list_0)
[mean_gpu_1, max_ram_1] = _prep_ram_and_gpu_data(gpu_list_1, ram_list_1)
[mean_gpu_2, max_ram_2] = _prep_ram_and_gpu_data(gpu_list_2, ram_list_2)
[mean_gpu_3_1, max_ram_3_1] = _prep_ram_and_gpu_data(gpu_list_3_1, ram_list_3_1)
[mean_gpu_3_2, max_ram_3_2] = _prep_ram_and_gpu_data(gpu_list_3_2, ram_list_3_2)

#mean of computational time
comp_mean_0 = np.mean(comp_time_0, axis = 1)
comp_mean_1 = np.mean(comp_time_1, axis = 1)
comp_mean_2 = np.mean(comp_time_2, axis = 1)
comp_mean_3_1 = np.mean(comp_time_3_1, axis = 1)
comp_mean_3_2 = np.mean(comp_time_3_2, axis = 1)        




#%% 
#Visualisation
# x-axis labels
x_labels = [10, 100, 1000, 10000, 100000, 500000]

#Computational Time
#create plot
plt.figure(figsize = (10, 6))
plt.plot(x_labels, comp_mean_0, marker = 'o', label = 'Algorithm 0')
plt.plot(x_labels, comp_mean_1, marker = 'o', label = 'Algorithm 1')
plt.plot(x_labels, comp_mean_2, marker = 'o', label = 'Algorithm 2')
plt.plot(x_labels, comp_mean_3_1, marker = 'o', label = 'Algorithm 3.1')
plt.plot(x_labels, comp_mean_3_2, marker = 'o', label = 'Algorithm 3.2')
plt.xscale('log')  #log-scale x-axis
plt.yscale('log')  #log-scale y-axis
plt.xticks(x_labels)  
plt.xlabel('Length of Input Sequence')
plt.ylabel('Mean Computational Time (sec)')
plt.title('Computational Time of the Different Implemented Forward-Algorithms')
plt.grid(True)
plt.legend()   
plt.savefig('compare_time_all_alg.png') #save plot
plt.show()


#GPU
#create plot
plt.figure(figsize = (10, 6))
plt.plot(x_labels, mean_gpu_0, marker = 'o', label = 'Algorithm 0')
plt.plot(x_labels, mean_gpu_1, marker = 'o', label = 'Algorithm 1')
plt.plot(x_labels, mean_gpu_2, marker = 'o', label = 'Algorithm 2')
plt.plot(x_labels, mean_gpu_3_1, marker = 'o', label = 'Algorithm 3.1')
plt.plot(x_labels, mean_gpu_3_2, marker = 'o', label = 'Algorithm 3.2')
plt.xscale('log')  #log-scale x-axis
plt.xlabel('Length of Input Sequence')
plt.ylabel('Mean of GPU-Utilization (%)')
plt.title('GPU-Utilization of the Different Implemented Forward-Algorithms')
plt.grid(True)
plt.legend()
plt.savefig('compare_gpu_all_alg.png') #save plot
plt.show()


#RAM
#create plot
plt.figure(figsize = (10, 6))
plt.plot(x_labels, max_ram_0, marker = 'o', label = 'Algorithm 0')
plt.plot(x_labels, max_ram_1, marker = 'o', label = 'Algorithm 1')
plt.plot(x_labels, max_ram_2, marker = 'o', label = 'Algorithm 2')
plt.plot(x_labels, max_ram_3_1, marker = 'o', label = 'Algorithm 3.1')
plt.plot(x_labels, max_ram_3_2, marker = 'o', label = 'Algorithm 3.2')
plt.xscale('log')  #log-scale x-axis
plt.xlabel('Length of Input Sequence')
plt.ylabel('Maximum used RAM (MiB)')
plt.title('Maximum Used RAM of the Different Implemented Forward-Algorithms')
plt.axhline(y = 8192, color = 'gray', linestyle = '--', label = 'Max RAM')
plt.grid(True)
plt.legend()
plt.savefig('compare_ram_all_alg.png') #save plot
plt.show()



