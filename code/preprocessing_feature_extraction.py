import os
import numpy as np
import pandas as pd
import itertools
from scipy import io
import matplotlib.pyplot as plt
import mne_connectivity
from mne_connectivity.viz import plot_connectivity_circle
import warnings
# Suppress FutureWarning for deprecated frame.append method
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_data(directory, output_fig):
    '''
    This function is responsible for data loading and preprocessing for multiple conditions, specifically Desktop and VR.
    
    Preprocessing Steps:
    - Extracting flight simulation timestamps.
    - Segmenting the time series based on the extracted timestamps.
    - Applying data normalization.

    Feature Extraction:
    - Computing the Phase Locking Value (PLV) values.

    Returns:
    - PLV values for all samples and conditions.
    '''
    
    Desktop_plv, VR_plv = [], []
    
    files = sorted(os.listdir(directory))[1:] # index from 1 because Mac has .DS_Store file
    
    for i in range(0, len(files), 2):  # Iterate through the files with a step of 2
        if i + 1 < len(files):  # Ensure there are two more files available            
            Desktop = files[i]
            VR = files[i + 1]

            # load the .mat files
            data_Desktop = io.loadmat(os.path.join(directory, Desktop))
            data_VR = io.loadmat(os.path.join(directory, VR))
            
            # Desktop condition
            baseline_latency, test_latency = extract_events(data_Desktop)
            data_baseline, data_trial = extract_data(baseline_latency, test_latency, data_Desktop)
            data_Desktop = normalize_data(data_baseline, data_trial)

            # retrieve filename
            split_filename = Desktop.split("_")
            filename = split_filename[1] + "_" + split_filename[-1].split(".")[0]
            
            # save PLV values
            plv_alpha, plv_beta, plv_theta = plv_function(data_Desktop, filename, output_fig)
            Desktop_plv.append([plv_alpha, plv_beta, plv_theta])
            
            # VR condition
            baseline_latency, test_latency = extract_events(data_VR)
            data_baseline, data_trial = extract_data(baseline_latency, test_latency, data_VR)
            data_VR = normalize_data(data_baseline, data_trial)
            
            # retrieve filename
            split_filename = Desktop.split("_")
            filename = split_filename[1] + "_" + split_filename[-1].split(".")[0]
            
            # save PLV values
            plv_alpha, plv_beta, plv_theta = plv_function(data_VR, filename, output_fig)
            VR_plv.append([plv_alpha, plv_beta, plv_theta])
    
    return Desktop_plv, VR_plv

def extract_events(data):
    
    # Provided EEG.event object
    event_latencies = data['EEG'][0][0]['event'][0]

    # Extract the event latencies for "start baseline" and "start test"
    baseline_latency = next(int(event[0][0][0]) for event in event_latencies if event[1][0] == 'start baseline') # retrieve baseline frequency
    test_latency = next(int(event[0][0][0]) for event in event_latencies if event[1][0] == 'start test') # start simulation
    
    return baseline_latency, test_latency

def extract_data(baseline_latency, test_latency, data):
    
    # Provided EEG.data object (channels, timeseries)
    data = data['EEG'][0][0]['data']

    # Slice the columns for each event
    data_baseline = data[:, baseline_latency:test_latency]
    data_trial = data[:, test_latency:]
    
    return data_baseline, data_trial
    
def normalize_data(data_baseline, data_trial):
    
    # mean subtraction method to normalize the data for each sample
    data = data_trial - np.mean(data_baseline)

    return data

def plv_function(data, filename, output_fig):
    '''
    Retrieving preprocessed and normalized EEG recording data of the flight simulation.
    
    Functionality:
    - Calculates the Phase Locking Value (PLV) features for each sample.
    - Plots the PLV features for each frequency band and for each sample.

    Returns:
    - PLV features for each sample.
    '''
    
    # Extract the time series data from the frontal and parietal regions
    data = np.concatenate((data[0:5, ], data[8:13, ]), axis=0)

    # Reshape data to have a single epoch dimension
    data = data.reshape(1, data.shape[0], data.shape[1])

    # Define the frequency band ranges
    fmin_theta = 4  # Lower freq of theta band
    fmax_theta = 8  # Upper freq of theta band
    fmin_alpha = 8  # Lower freq of alpha band
    fmax_alpha = 13  # Upper freq of alpha band
    fmin_beta = 13  # Lower freq of beta band
    fmax_beta = 30  # Upper freq of beta band

    sfreq = 256
    method='plv'
    
    # Set freqs parameter for each bands
    freqs_theta = np.arange(fmin_theta, fmax_theta + 1)
    freqs_alpha = np.arange(fmin_alpha, fmax_alpha + 1)
    freqs_beta = np.arange(fmin_beta, fmax_beta + 1)

    # Call the spectral_connectivity_time function for each frequency band
    con_theta = mne_connectivity.spectral_connectivity_time(data, freqs_theta, method=method, sfreq=sfreq, fmin=fmin_theta, fmax=fmax_theta, mode='cwt_morlet', average=True, faverage=True)
    con_alpha = mne_connectivity.spectral_connectivity_time(data, freqs_alpha, method=method, sfreq=sfreq, fmin=fmin_alpha, fmax=fmax_alpha, mode='cwt_morlet', average=True, faverage=True)
    con_beta = mne_connectivity.spectral_connectivity_time(data, freqs_beta, method=method, sfreq=sfreq, fmin=fmin_beta, fmax=fmax_beta, mode='cwt_morlet', average=True, faverage=True)

    # Get the connectivity data for each frequency band
    data_theta = con_theta.get_data('raveled')
    data_alpha = con_alpha.get_data('raveled')
    data_beta = con_beta.get_data('raveled')

    # Remove zero entries from the connectivity data
    data_theta = data_theta[data_theta != 0]
    data_alpha = data_alpha[data_alpha != 0]
    data_beta = data_beta[data_beta != 0]
    
    # Number of nodes or channels
    n_nodes = data.shape[1]
    
    # Generate all possible combinations of nodes
    combinations = list(itertools.combinations(range(n_nodes), 2))

    # Extract "in" and "out" indices using zip() and list comprehension
    in_indices, out_indices = zip(*combinations)

    # Convert to numpy arrays
    in_indices = np.array(in_indices)
    out_indices = np.array(out_indices)

    # # Select the desired indices
    desired_indices = [4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
    in_indices = in_indices[desired_indices]
    out_indices = out_indices[desired_indices]

    # Filter the connectivity data to include only the desired indices
    filtered_data_theta = data_theta[desired_indices]
    filtered_data_alpha = data_alpha[desired_indices]
    filtered_data_beta = data_beta[desired_indices]

    # Specify indices as a tuple
    indices = (in_indices, out_indices)

    # Define the filtered data for each plot
    plot_data = [filtered_data_theta, filtered_data_alpha, filtered_data_beta]
    
    # frontal and parietal electrodes
    node_names = ['F7', 'F3', 'Fz', 'F4', 'F8', 'P3', 'Pz', 'P4', 'PO7', 'PO8']
    # red is for the Frontal channels and blue is for the Parietal channels
    node_colors=['red', 'red', 'red', 'red', 'red', 'blue', 'blue', 'blue', 'blue', 'blue']

    # Define the data and titles for each plot
    plot_titles = ['Theta - PLV', 'Alpha - PLV', 'Beta - PLV']
    fig_names = [f'{filename}_plv_theta', f'{filename}_plv_alpha', f'{filename}_plv_beta']

    # Iterate over the subplots
    for i in range(3):
        # Run plot_connectivity_circle() for each subplot
        # n_lines parameter can be added, shows n of highest connectivity lines
        plot_connectivity_circle(plot_data[i], node_names, node_colors=node_colors, indices=indices, fontsize_names=15, 
                                 node_edgecolor='w', textcolor='w', node_height=2, facecolor='k', linewidth=2, colormap='plasma', 
                                 fontsize_colorbar=10, colorbar_size=0.75, colorbar_pos=(0, 0.7), fontsize_title=25, title=plot_titles[i], 
                                 interactive=True, show=False, vmin=0, vmax=1)
        
        # Save the figure as a PNG file
        fig_name = f'{fig_names[i]}.png'
        fig_path = os.path.join(output_fig, fig_name)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        
        # Clear the current figure and release the memory
        plt.clf()
    
    return data_alpha, data_beta, data_theta

def read_FrequencySpectrumFeatures(directory):
    
    baseline_Desktop, baseline_VR, trial_Desktop, trial_VR = [], [], [], []
    
    # loading the spectral features
    for subdir in sorted(os.listdir(directory))[1:]: # index from 1 because Mac has .DS_Store file
        for file in sorted(os.listdir(os.path.join(directory, subdir))):
            df = pd.read_csv(os.path.join(directory, subdir, file))
                        
            if subdir == 'Baseline' and file.startswith('Desktop'):                
                baseline_Desktop.append(df)
            elif subdir == 'Baseline' and file.startswith('VR'):
                baseline_VR.append(df)
            elif subdir == 'Trial' and file.startswith('Desktop'):
                trial_Desktop.append(df)
            else:
                trial_VR.append(df)
    
    # also baseline normalization for the absolute or relative spectral power features
    normalized_Desktop = [normalize_data(baseline, trial) for baseline, trial in zip(baseline_Desktop, trial_Desktop)]
    normalized_VR = [normalize_data(baseline, trial) for baseline, trial in zip(baseline_VR, trial_VR)]
    
    return normalized_Desktop, normalized_VR

def combine_features(Desktop_freq, VR_freq, Desktop_plv, VR_plv):
    
    features_Desktop, features_VR = [], []

    # Combine all the PLV features and Spectral features together
    for plv_Desktop, plv_VR, alpha_Desktop, alpha_VR, beta_Desktop, beta_VR, theta_Desktop, theta_VR in zip(Desktop_plv, VR_plv, 
                                                                                                            Desktop_freq[0].iterrows(), VR_freq[0].iterrows(),
                                                                                                            Desktop_freq[1].iterrows(), VR_freq[1].iterrows(),
                                                                                                            Desktop_freq[2].iterrows(), VR_freq[2].iterrows()):

        # 135 PLV features + 42 spectral features
        features_Desktop.append(np.concatenate((plv_Desktop[0], plv_Desktop[1], plv_Desktop[2], np.array(alpha_Desktop[1].tolist()), np.array(beta_Desktop[1].tolist()), np.array(theta_Desktop[1].tolist()))))
        features_VR.append(np.concatenate((plv_VR[0], plv_VR[1], plv_VR[2], np.array(alpha_VR[1].tolist()), np.array(beta_VR[1].tolist()), np.array(theta_VR[1].tolist()))))
        
    return features_Desktop, features_VR

def binarizing_labels(Desktop, VR, labels, output_dir):
    '''
    Transforming continous labels to binary labels. 0 = low workload, 1 = high workload.
    '''
    
    # extracting the workload labels for Desktop and VR
    # index row player 45 dropped, because player 45 misses the 'start trial' event latency, so is dropt from the data
    Desktop_labels = pd.read_csv(labels, sep=',')['NASA-TLX DESKTOP'].drop(44)
    VR_labels = pd.read_csv(labels, sep=',')['NASA-TLX VR'].drop(44)

    # Threshold for workload: low workload <= 50, high workload > 50
    thresh = 50
    # Assign binary labels to each matrix in the list
    data_Desktop = [{'X': data, 'label': np.array([0])} if label <= thresh else {'X': data, 'label': np.array([1])} for data, label in zip(Desktop, Desktop_labels)]
    data_VR = [{'X': data, 'label': np.array([0])} if label <= thresh else {'X': data, 'label': np.array([1])} for data, label in zip(VR, VR_labels)]
    
    for participant, Desktop, VR in zip(range(1,52), data_Desktop, data_VR):
        # assign the original names to the files, because participant 45 got deleted
        if participant >= 45:
            filename_Desktop = f'P{participant+1}_Desktop.npz'
            filename_VR = f'P{participant+1}_VR.npz'
        else:
            filename_Desktop = f'P{participant}_Desktop.npz'
            filename_VR = f'P{participant}_VR.npz'

        # Save the data to a NPZ file
        np.savez(os.path.join(output_dir, 'DESKTOP', filename_Desktop), X=Desktop['X'], Y=Desktop['label'])
        np.savez(os.path.join(output_dir, 'VR', filename_VR), X=VR['X'], Y=VR['label'])

directory = '/Users/basverkennis/Desktop/Flight-Sim-Cognitive-Workload-EEG-Prediction/raw/Preprocessed_EEG_files'
labels = '/Users/basverkennis/Desktop/Flight-Sim-Cognitive-Workload-EEG-Prediction/raw/Table_logsandqrs.csv'
directory_frequency_bands = '/Users/basverkennis/Desktop/Flight-Sim-Cognitive-Workload-EEG-Prediction/raw/Absolute_EEG_power_values'
output_dir = '/Users/basverkennis/Desktop/Flight-Sim-Cognitive-Workload-EEG-Prediction/data'
output_fig = '/Users/basverkennis/Desktop/Flight-Sim-Cognitive-Workload-EEG-Prediction/results/connectivity_plots_per_sample'

# directory = '/Flight-Sim-Cognitive-Workload-EEG-Prediction/raw/Preprocessed_EEG_files'
# labels = '/Flight-Sim-Cognitive-Workload-EEG-Prediction/raw/Table_logsandqrs.csv'
# directory_frequency_bands = '/Flight-Sim-Cognitive-Workload-EEG-Prediction/raw/Absolute_EEG_power_values'
# output_dir = '/Flight-Sim-Cognitive-Workload-EEG-Prediction/data'
# output_fig = '/Flight-Sim-Cognitive-Workload-EEG-Prediction/results/connectivity_plots_per_sample'

# loading the preprocessed EEG recording data to calculate the plv values
Desktop_plv, VR_plv = load_data(directory, output_fig)

# load the absolute or relative spectral power features
normalized_Desktop, normalized_VR = read_FrequencySpectrumFeatures(directory_frequency_bands)

# combine both sets of features
Desktop, VR = combine_features(normalized_Desktop, normalized_VR, Desktop_plv, VR_plv)

# binarize the labels
binarizing_labels(Desktop, VR, labels, output_dir)