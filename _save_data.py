import os 

def save_data(save_dir, comp_time_data, gpu_data, ram_data, algorithm_index):
    '''
    Function to save computational time, GPU, and RAM utilisation for a given 
    algorithm.
    
    Args:
        save_dir (str): Directory where the data should be saved.
        comp_time_data (np.array): Computational time data for the algorithm.
        gpu_data (list): GPU utilization data for the algorithm.
        ram_data (list): RAM usage data for the algorithm.
        algorithm_index (str): Index/Short of the algorithm 
        
    Returns:
        None
    '''
    
    # Create the directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Construct file paths for the computational time, GPU, and RAM data
    comp_time_file = os.path.join(save_dir, f'comp_time_{algorithm_index}.txt')
    gpu_file = os.path.join(save_dir, f'gpu_list_{algorithm_index}.txt')
    ram_file = os.path.join(save_dir, f'ram_list_{algorithm_index}.txt')
    
    # Save computational time
    try:
        with open(comp_time_file, 'w') as comp_file:
            comp_file.write(f"{comp_time_data}")
        
        # Save GPU utilization
        with open(gpu_file, 'w') as gpu_file:
            for gpu_usage in gpu_data:
                gpu_file.write(f"{gpu_usage}\n")
        
        # Save RAM utilization
        with open(ram_file, 'w') as ram_file:
            for ram_usage in ram_data:
                ram_file.write(f"{ram_usage}\n")
        
        print(f"Data successfully saved for Algorithm {algorithm_index}:\n"
              f"Computational Time: {comp_time_file}\n"
              f"GPU Utilization: {gpu_file}\n"
              f"RAM Usage: {ram_file}")
    
    except Exception as e:
        print(f"An error occurred while saving data for Algorithm {algorithm_index}: {e}")