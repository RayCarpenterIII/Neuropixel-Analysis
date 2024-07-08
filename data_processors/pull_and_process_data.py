from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import pandas as pd
import pickle
import numpy as np
import os
from tqdm import tqdm
import concurrent.futures
import torch
import json
import time
import gc


def create_directory_and_manifest(directory_name='output'):
    """
    Create the output directory and check or create the manifest.json file.
    
    Parameters:
    directory_name (str): Name of the directory to be created. Default is 'output'.
    
    Returns:
    tuple: A tuple containing the path to the output directory and the path to the manifest.json file.
    """
    output_dir = os.path.join(os.getcwd(), directory_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.exists(manifest_path):
        print("Using existing manifest.json file.")
    else:
        print("Creating a new manifest.json file.")
    
    return output_dir, manifest_path

def create_cache_get_session_table(manifest_path):
    """
    Create an instance of the EcephysProjectCache class and get the session table.
    
    Parameters:
    manifest_path (str): Path to the manifest.json file.
    
    Returns:
    tuple: A tuple containing the cache object and the session table.
    """
    cache = EcephysProjectCache(manifest=manifest_path)
    session_table = cache.get_session_table()
    return cache, session_table

def pick_session_and_pull_data(cache, session_number, timeout=300):
    try:
        session = cache.get_session_data(session_number,
                                         isi_violations_maximum=np.inf,
                                         amplitude_cutoff_maximum=np.inf,
                                         presence_ratio_minimum=-np.inf,
                                         timeout=timeout)
        spike_times = session.spike_times
        stimulus_table = session.get_stimulus_table("natural_scenes")
        
        # Display objects in session
        print("Session objects")
        print([attr_or_method for attr_or_method in dir(session) if attr_or_method[0] != '_'])
        
        return spike_times, stimulus_table
    
    except TimeoutError:
        print(f"Timeout occurred while fetching session data for session {session_number}")
        return None, None

def filter_valid_spike_times(spike_times, session):
    """
    Filter the valid spike times using invalid times from the session object.
    
    Parameters:
    session (object): The session object.
    
    Returns:
    dict: A dictionary containing the valid spike times.
    """
    # Access the invalid_times DataFrame
    invalid_times = session.invalid_times

    # Function to check if a spike time is valid
    def is_valid_time(spike_times, invalid_intervals):
        invalid = np.zeros_like(spike_times, dtype=bool)
        for _, row in invalid_intervals.iterrows():
            start, end = row['start_time'], row['stop_time']
            invalid |= (spike_times >= start) & (spike_times <= end)
        return ~invalid
    
    # Filter the valid spike times
    valid_spike_times = {}
    with tqdm(total=len(spike_times), desc='Filtering valid spike times') as pbar:
        for neuron, times in spike_times.items():
            valid_mask = is_valid_time(times, session.invalid_times)
            valid_spike_times[neuron] = times[valid_mask]
            pbar.update(1)

    return valid_spike_times


def get_stimulus_table(session, stimulus_type="natural_scenes"):
    """Retrieve the stimulus table for a given stimulus type."""
    return session.get_stimulus_table(stimulus_type)

def calculate_bins(stimulus_table, timesteps_per_frame=1):
    """Calculate the bin sizes and total bins based on the stimulus table."""
    image_start_times = torch.tensor(stimulus_table.start_time.values)
    image_end_times = torch.tensor(stimulus_table.stop_time.values)
    image_durations = image_end_times - image_start_times
    #bin_sizes = image_durations / timesteps_per_frame             #Removed, doesn't seem to be used 
    bins_per_image = timesteps_per_frame
    total_bins = bins_per_image * len(image_start_times)
    return image_start_times, image_end_times, total_bins, bins_per_image

def process_neuron(times, image_start_times, image_end_times, total_bins, bins_per_image):
    """Process a single neuron's spike times."""
    start_bin = 0
    neuron_spike_bins = torch.zeros(total_bins, dtype=torch.int32)
    for image_idx, (start_time, end_time) in enumerate(zip(image_start_times, image_end_times)):
        bin_edges = torch.linspace(start_time, end_time, bins_per_image + 1)
        binned_spike_times = torch.histc(torch.tensor(times), bins=bin_edges.shape[0]-1, min=bin_edges.min(), max=bin_edges.max())
        end_bin = start_bin + bins_per_image
        if len(binned_spike_times) == len(neuron_spike_bins[start_bin:end_bin]):
            neuron_spike_bins[start_bin:end_bin] = binned_spike_times
        start_bin = end_bin
    return neuron_spike_bins

def process_neuron_wrapper(args):
    """Wrapper function to unpack arguments and call process_neuron."""
    return process_neuron(*args)

def process_all_neurons(spike_times, image_start_times, image_end_times, total_bins, bins_per_image, batch_size=100):
    """Process neurons in parallel."""
    args_list = [(times, image_start_times, image_end_times, total_bins, bins_per_image) for times in spike_times.values()]
    spike_matrix = []
    
    #Process neurons in smaller batches to prevent memory overload
    for i in range(0, len(args_list), batch_size):
        batch = args_list[i:i+batch_size]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            batch_result = list(tqdm(executor.map(process_neuron_wrapper, batch), total=len(batch), desc=f'Processing neurons batch {i//batch_size + 1}'))
        spike_matrix.extend(batch_result)
        
        gc.collect() 
        
    print(f'Spike matrix size -> {np.shape(spike_matrix)}')
    #Return list of tensors instead of torch.stack for memory efficiency 
    return [torch.tensor(matrix) for matrix in spike_matrix]
    
def create_and_prepare_spike_dataframe(spike_matrix, spike_times, stimulus_table, timesteps_per_frame):
    """
    Create a spike DataFrame and prepare it by adding a frame column.
    
    Parameters:
    spike_matrix (torch.Tensor): The spike matrix containing processed spike times for all neurons.
    spike_times (dict): A dictionary containing the original spike times.
    stimulus_table (pd.DataFrame): The stimulus table containing stimulus-related data.
    timesteps_per_frame (int): Number of timesteps per frame for binning.
    
    Returns:
    pd.DataFrame: A prepared DataFrame containing spike data and a frame column.
    """
    # Create the DataFrame
    spike_df = pd.DataFrame(spike_matrix.numpy(), index=spike_times.keys())
    
    # Transpose the DataFrame
    spike_df = spike_df.T
    
    # Add and populate the frame column
    spike_df['frame'] = 'nan'
    spike_df['frame'] = np.repeat(np.array(stimulus_table['frame']), timesteps_per_frame)

    return spike_df


def save_and_count_spike_dataframe(spike_df, session_number, output_dir, timesteps_per_frame):
    """
    Save the spike DataFrame to a pickle file in the specified directory and count the number of NaN values.
    
    Parameters:
    spike_df (pd.DataFrame): The spike DataFrame to be saved.
    session_number (int): The session number.
    output_dir (str): The directory where the DataFrame should be saved.
    timesteps_per_frame (int): Number of timesteps per frame for binning.
    
    Returns:
    str: The name of the saved file.
    """
    nan_count = spike_df.isna().sum().sum()
    print(f"Number of NaN values in the DataFrame: {nan_count}")
    
    file_name = f'spike_trains_with_stimulus_session_{session_number}_{timesteps_per_frame}.pkl'
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(spike_df, f)
        
    return file_name


def normalize_firing_rates(df):
    """Normalize the firing rates by calculating z-scores."""
    df_copy = df.drop('frame', axis=1)
    normalized_firing_rates = (df_copy - df_copy.mean()) / df_copy.std()
    normalized_firing_rates.insert(0, 'frame', df['frame'])
    return normalized_firing_rates

def filter_and_save_neurons(normalized_firing_rates, highest_value=100, lowest_value=0, session_number=None, output_dir=None): #TODO: Don't use mask??
    """
    Filter neurons based on firing rate thresholds, check for NaN values, and save the filtered data.
    
    Parameters:
    normalized_firing_rates (pd.DataFrame): DataFrame containing normalized firing rates.
    highest_value (float): Upper z-score threshold for filtering neurons.
    lowest_value (float): Lower z-score threshold for filtering neurons.
    session_number (int, optional): The session number, required if saving the data.
    output_dir (str, optional): The directory where the DataFrame should be saved, required if saving the data.
    
    Returns:
    pd.DataFrame: DataFrame containing filtered neurons.
    """
    # Filter neurons
    selected_neurons_mask = (normalized_firing_rates > highest_value).any() | (normalized_firing_rates < lowest_value).any()
    filtered_normalized_firing_rates = normalized_firing_rates.loc[:, selected_neurons_mask]
    
    # Check for NaN values
    nan_present = filtered_normalized_firing_rates.isna().any().any()
    print(f"There {'are' if nan_present else 'are no'} NaN values in the DataFrame")
    
    # Save the filtered data, if session_number and output_dir are provided
    if session_number is not None and output_dir is not None:
        file_name = f'filtered_normalized_pickle_{session_number}.pkl'
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, 'wb') as f:
            pickle.dump(filtered_normalized_firing_rates, f)
        print(f"Filtered normalized firing rates saved to {file_path}")
        
    return filtered_normalized_firing_rates


def process_and_save_data(session, spike_times, session_number, output_dir, timesteps_per_frame=10, highest_value=100, lowest_value=0):
    """
    Orchestrates the entire data processing, cleaning, and saving workflow.
    
    Parameters:
    session (object): The session object containing neural data.
    spike_times (dict): The spike times for neurons.
    session_number (int): The session number.
    output_dir (str): The directory where data should be saved.
    timesteps_per_frame (int, optional): Number of timesteps per frame for binning. Default is 10.
    highest_value (float, optional): Upper z-score threshold for filtering neurons. Default is 100.
    lowest_value (float, optional): Lower z-score threshold for filtering neurons. Default is 0.
    
    Returns:
    str: The name of the saved filtered data file.
    """
    print("Step 1: Preparing the data")
    stimulus_table = get_stimulus_table(session)
    image_start_times, image_end_times, total_bins, bins_per_image = calculate_bins(stimulus_table, timesteps_per_frame)
    spike_matrix = process_all_neurons(spike_times, image_start_times, image_end_times, total_bins, bins_per_image)
    spike_df = create_and_prepare_spike_dataframe(spike_matrix, spike_times, stimulus_table, timesteps_per_frame)
    
    print("Step 2: Cleaning the data")
    normalized_firing_rates = normalize_firing_rates(spike_df)
    filtered_normalized_firing_rates = filter_and_save_neurons(normalized_firing_rates, highest_value, lowest_value, session_number, output_dir)
    
    print("Step 3: Data Validation and Final Saving")
    nan_count = filtered_normalized_firing_rates.isna().sum().sum()
    print(f"Number of NaN values in the final DataFrame: {nan_count}")
    
    print("Data processing and saving complete.")
    filtered_data_file = save_and_count_spike_dataframe(filtered_normalized_firing_rates, session_number, output_dir, timesteps_per_frame)
    return filtered_data_file


def get_session_ids(cache):
    """
    Get all available session IDs from the cache.
    
    Parameters:
    cache (EcephysProjectCache): The cache object.
    
    Returns:
    list: A list of all session IDs.
    """
    session_table = cache.get_session_table()
    return session_table.index.tolist()

def process_session_batch(cache, session_batch, output_dir, timesteps_per_frame=5, timeout=300):
    """
    Process a batch of sessions.
    
    Parameters:
    cache (EcephysProjectCache): The cache object.
    session_batch (list): List of session IDs to process.
    output_dir (str): The directory where data should be saved.
    timesteps_per_frame (int): Number of timesteps per frame for binning.
    timeout (int): Timeout for fetching session data.
    
    Returns:
    dict: A dictionary with session IDs as keys and processed DataFrames as values.
    """
    processed_data = {}
    
    for session_number in session_batch:
        print(f"\nProcessing session {session_number}")
        spike_df = master_function(session_number, output_dir, timesteps_per_frame, timeout)
        
        if spike_df is not None:
            processed_data[session_number] = spike_df
            
            # Save the spike_df as a pickle file
            pkl_file_path = os.path.join(output_dir, f'spike_trains_with_stimulus_session_{session_number}_{timesteps_per_frame}.pkl')
            with open(pkl_file_path, 'wb') as f:
                pickle.dump(spike_df, f)
            print(f"Saved pickle file for session {session_number}")
        
        gc.collect()
    
    return processed_data

def process_all_sessions_in_batches(cache, output_dir, batch_size=5, timesteps_per_frame=5, timeout=300):
    """
    Process all available sessions in batches.
    
    Parameters:
    cache (EcephysProjectCache): The cache object.
    output_dir (str): The directory where data should be saved.
    batch_size (int): Number of sessions to process in each batch.
    timesteps_per_frame (int): Number of timesteps per frame for binning.
    timeout (int): Timeout for fetching session data.
    
    Returns:
    dict: A dictionary with session IDs as keys and processed DataFrames as values.
    """
    session_ids = get_session_ids(cache)
    processed_data = {}
    
    print(f"Total number of sessions: {len(session_ids)}")
    
    for i in range(0, len(session_ids), batch_size):
        batch = session_ids[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1} of {len(session_ids)//batch_size + 1}")
        batch_results = process_session_batch(cache, batch, output_dir, timesteps_per_frame, timeout)
        processed_data.update(batch_results)
        
        gc.collect()
        
        #Add delay between batches to prevent kernal timeout
        time.sleep(10)
    
    return processed_data



def master_function(session_number, output_dir, timesteps_per_frame=10, timeout=300):
    """
    Master function to execute the entire workflow from data extraction to saving.

    Parameters:
    session_number (int): The session number.
    output_dir (str): The directory where data should be saved. Default is 'output'.
    timesteps_per_frame (int, optional): Number of timesteps per frame for binning. Default is 10.

    Returns:
    str: The name of the saved filtered data file.
    """
    
    output_dir = '/proj/STOR/pipiras/Neuropixel/output'
    
    start_time = time.time()

    print("Updated version 3!")
    print("Initializing workflow...")
    
    # Paths for input files
    spike_trains_file_path = os.path.join(output_dir, f'spike_trains_with_stimulus_session_{session_number}_{timesteps_per_frame}.pkl')
    # normalized_firing_rates_file_path = os.path.join(output_dir, f'normalized_firing_rates_{session_number}_{timesteps_per_frame}.pkl')

    # Check if the files already exist
    if os.path.exists(spike_trains_file_path):
        print("Loading existing datasets...")
        
        # Load the spike trains 
        with open(spike_trains_file_path, 'rb') as f:
            spike_df = pickle.load(f)
        print(f"Loaded spike trains dataset: {type(spike_df)}")

        print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")
        return spike_df

    else:
        try:
            # Create data if non-existent
            print("Creating directory and manifest path...")
            output_dir, manifest_path = create_directory_and_manifest(directory_name=output_dir)

            print("Initializing EcephysProjectCache and session table...")
            cache, session_table = create_cache_get_session_table(manifest_path)
            session = cache.get_session_data(session_number)

            print("Fetching session data and spike times...")
            spike_times, stimulus_table = pick_session_and_pull_data(cache, session_number, timeout)

            print("Filtering valid spike times...")
            valid_spike_times = filter_valid_spike_times(spike_times, session)
            gc.collect()

            print("Calculating bins and processing neurons...")
            image_start_times, image_end_times, total_bins, bins_per_image = calculate_bins(stimulus_table, timesteps_per_frame)
            spike_matrix = process_all_neurons(valid_spike_times, image_start_times, image_end_times, total_bins, bins_per_image)
            gc.collect()

            print("Preparing spike DataFrame...")
            spike_df = create_and_prepare_spike_dataframe(spike_matrix, valid_spike_times, stimulus_table, timesteps_per_frame)
            gc.collect()

            print("Normalizing firing rates...")
            normalized_firing_rates = normalize_firing_rates(spike_df)

            print("Save raw and normalized firing rate dataframes as a pickle file.")
            with open(spike_trains_file_path, 'wb') as f:
                pickle.dump(spike_df, f)
            with open(normalized_firing_rates_file_path, 'wb') as f:
                pickle.dump(normalized_firing_rates, f)

            print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")

        except Exception as e:
            #logging.error(f"Error: An unexpected error occurred - {str(e)}")
            return None, None

    return spike_df, normalized_firing_rates
