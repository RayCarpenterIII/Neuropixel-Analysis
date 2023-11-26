import os
import pickle
import pandas as pd

def check_file_exists(session_id, output_dir="output", file_prefix="filtered_normalized_pickle", timesteps_per_frame=10):
    """
    Check if the pickle file exists in the specified directory.
    
    Parameters:
    session_id (int): The ID of the session.
    output_dir (str): The directory where the pickle file is expected to be found.
    file_prefix (str): The prefix of the pickle file.
    
    Returns:
    bool: True if the file exists, False otherwise.
    """
    pickle_path = os.path.join(output_dir, f"{file_prefix}_{session_id}_{timesteps_per_frame}.pkl")
    return os.path.exists(pickle_path)

def load_pickle_file(pickle_path):
    """
    Load a pickle file from the given path.
    
    Parameters:
    pickle_path (str): Path to the pickle file.
    
    Returns:
    object: The object loaded from the pickle file, or None if loading fails.
    """
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"No file found at {pickle_path}")
        return None

def save_pickle_file(data, pickle_path):
    """
    Save data to a pickle file at the given path.
    
    Parameters:
    data (object): The data to save.
    pickle_path (str): Path to save the pickle file.
    """
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {pickle_path}")

def clean_avg_firing_rates(df, highest_value, lowest_value):
    """
    Clean the average firing rates in the given DataFrame.
    
    Parameters:
    df (DataFrame): The DataFrame with the firing rates data.
    highest_value (float): Upper z-score threshold for filtering neurons.
    lowest_value (float): Lower z-score threshold for filtering neurons.
    
    Returns:
    DataFrame: The cleaned DataFrame.
    """
    df_copy = df.drop('frame', axis=1)
    normalized_firing_rates = (df_copy - df_copy.mean()) / df_copy.std()
    normalized_firing_rates.insert(0, 'frame', df['frame'])

    selected_neurons_mask = (~(normalized_firing_rates > highest_value).any(axis=0)) & ((normalized_firing_rates > lowest_value).any(axis=0))
    filtered_normalized_firing_rates = normalized_firing_rates.loc[:, ['frame'] + selected_neurons_mask[selected_neurons_mask].index.tolist()]
    
    return filtered_normalized_firing_rates

def master_cleaning_and_saving(session_id, output_dir='output', timesteps_per_frame=10, original_pickle_prefix='filtered_normalized_pickle', new_pickle_prefix='spike_trains_with_stimulus_session', highest_value=100, lowest_value=0):
    """
    Master function to check, clean, and save the firing rate data within an 'output' directory.
    
    Parameters:
    session_id (int): The ID of the session to be processed.
    output_dir (str): The directory where the pickle file is located.
    original_pickle_prefix (str): The prefix of the original pickle file.
    new_pickle_prefix (str): The prefix for the new cleaned pickle file.
    highest_value (float): The threshold to remove neurons with a high z-score.
    lowest_value (float): The threshold to remove neurons with a z-score never exceeding this value.
    """
    # Check if the original pickle file exists
    if check_file_exists(session_id, output_dir, original_pickle_prefix, timesteps_per_frame):
        # Construct the path to the original pickle file
        original_pickle_path = os.path.join(output_dir, f'{original_pickle_prefix}_{session_id}_{timesteps_per_frame}.pkl')

        # Load the original data
        spike_df = load_pickle_file(original_pickle_path)
        
        # Clean the average firing rates
        filtered_normalized_firing_rates = clean_avg_firing_rates(spike_df, highest_value, lowest_value)

        # Drop rows where the frame is -1
        filtered_normalized_firing_rates = filtered_normalized_firing_rates[filtered_normalized_firing_rates['frame'] != -1]

        # Construct the path to the new pickle file
        new_pickle_path = os.path.join(output_dir, f'{new_pickle_prefix}_{session_id}.pkl')

        # Save the cleaned data
        save_pickle_file(filtered_normalized_firing_rates, new_pickle_path)
        print(f"Original data file for session {session_id} has been cleaned and saved.")
        
    else:
        print(f"Original data file for session {session_id} does not exist in '{output_dir}' directory.")

    return filtered_normalized_firing_rates

# Example usage:
# master_cleaning_and_saving(session_id=721123822)

