import os
import numpy as np
import pandas as pd
import itertools
from scipy import io
import matplotlib.pyplot as plt
import mne_connectivity
from mne_connectivity.viz import plot_connectivity_circle
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
# Suppress FutureWarning for deprecated frame.append method
warnings.simplefilter(action='ignore', category=FutureWarning)

def LoadData_ProcessData_FeatureExtraction(directory, output_fig):
    '''
    Data loading and preprocessing for conditions: Desktop & VR.

    Preprocessing:
    - Extracting preprocessed flight simulation EEG time series.
    - Segmenting the time series, according to pre-defined marked timestamps (baseline and test phases in th flight simulation).
    - Applying baseline normalization with mean of trial phase EEG time series.

    Feature Extraction:
    - Computing the Connectivity Phase Locking Value (PLV) values.

    Returns:
    - Connectivity (PLV) values for all samples and conditions.
    '''
    
    Desktop_plv, VR_plv = [], []
    files = sorted(os.listdir(directory))
    for i in range(0, len(files), 2): 
        if i + 1 < len(files):
            
            # Desktop condition
            Desktop = files[i]
            print('Desktop: ', Desktop.split('_')[1])
            data_Desktop = io.loadmat(os.path.join(directory, Desktop))
            baseline_latency, test_latency = extract_events(data_Desktop)
            data_baseline, data_trial = extract_data(baseline_latency, test_latency, data_Desktop)
            
            filename = split_filename(Desktop) + '_baseline'
            baseline_plv_alpha, baseline_plv_beta, baseline_plv_theta = plv_function(data_baseline, filename, output_fig)
            filename = split_filename(Desktop) + '_trial'
            trial_plv_alpha, trial_plv_beta, trial_plv_theta = plv_function(data_trial, filename, output_fig)
            data_Desktop_alpha = normalize_data(baseline_plv_alpha, trial_plv_alpha)
            data_Desktop_beta = normalize_data(baseline_plv_beta, trial_plv_beta)
            data_Desktop_theta = normalize_data(baseline_plv_theta, trial_plv_theta)
            
            Desktop_plv.append([data_Desktop_alpha, data_Desktop_beta, data_Desktop_theta])
            
            # VR condition
            VR = files[i + 1]
            print('VR: ', VR.split('_')[1])
            data_VR = io.loadmat(os.path.join(directory, VR))
            baseline_latency, test_latency = extract_events(data_VR)
            data_baseline, data_trial = extract_data(baseline_latency, test_latency, data_VR)
            
            filename = split_filename(VR) + '_baseline'
            baseline_plv_alpha, baseline_plv_beta, baseline_plv_theta = plv_function(data_baseline, filename, output_fig)
            filename = split_filename(VR) + '_trial'
            baseline_plv_alpha, baseline_plv_beta, baseline_plv_theta = plv_function(data_baseline, filename, output_fig)
            data_VR_alpha = normalize_data(baseline_plv_alpha, trial_plv_alpha)
            data_VR_beta = normalize_data(baseline_plv_beta, trial_plv_beta)
            data_VR_theta = normalize_data(baseline_plv_theta, trial_plv_theta)
            
            VR_plv.append([data_VR_alpha, data_VR_beta, data_VR_theta])
    
    return Desktop_plv, VR_plv

def split_filename(filename):
    return filename.split("_")[1] + "_" + filename.split("_")[-1].split(".")[0]

def extract_events(data):
    event_latencies = data['EEG'][0][0]['event'][0]
    baseline_latency = next(int(event[0][0][0]) for event in event_latencies if event[1][0] == 'start baseline')
    test_latency = next(int(event[0][0][0]) for event in event_latencies if event[1][0] == 'start test')
    return baseline_latency, test_latency

def extract_data(baseline_latency, test_latency, data):
    data = data['EEG'][0][0]['data']
    data_baseline = data[:, baseline_latency:test_latency]
    data_trial = data[:, test_latency:]
    return data_baseline, data_trial
    
def normalize_data(data_baseline, data_trial):
    data = data_trial - np.mean(np.array([data_trial, data_baseline]), axis=0)
    return data

def plv_function(data, filename, output_fig):
    '''
    Compute the Connectivity (PLV) values for each sample. 
    
    Input: 
    - Preprocessed (normalized) EEG timeseries of flight simulation.
    
    Functionality:
    - Calculates the Phase Locking Value (PLV) features for each sample in three frequency bands (alpha, beta, theta).
    - Connectivity plot for each sample and all three frequency bands.

    Returns:
    - Connectivity (PLV) features for each sample.
    '''
    
    data = np.concatenate((data[0:5, ], data[8:13, ]), axis=0)
    data = data.reshape(1, data.shape[0], data.shape[1])

    fmin_theta, fmax_theta = 4, 8
    fmin_alpha, fmax_alpha = 8, 13
    fmin_beta, fmax_beta = 13, 30
    sfreq, method = 256, 'plv'
    freqs_theta = np.arange(fmin_theta, fmax_theta + 1)
    freqs_alpha = np.arange(fmin_alpha, fmax_alpha + 1)
    freqs_beta = np.arange(fmin_beta, fmax_beta + 1)
    
    con_theta = mne_connectivity.spectral_connectivity_time(data, freqs_theta, method=method, sfreq=sfreq, fmin=fmin_theta, fmax=fmax_theta, mode='cwt_morlet', average=True, faverage=True, verbose=False)
    con_alpha = mne_connectivity.spectral_connectivity_time(data, freqs_alpha, method=method, sfreq=sfreq, fmin=fmin_alpha, fmax=fmax_alpha, mode='cwt_morlet', average=True, faverage=True, verbose=False)
    con_beta = mne_connectivity.spectral_connectivity_time(data, freqs_beta, method=method, sfreq=sfreq, fmin=fmin_beta, fmax=fmax_beta, mode='cwt_morlet', average=True, faverage=True, verbose=False)
    
    data_theta = con_theta.get_data('raveled')
    data_alpha = con_alpha.get_data('raveled')
    data_beta = con_beta.get_data('raveled')
    data_theta = data_theta[data_theta != 0]
    data_alpha = data_alpha[data_alpha != 0]
    data_beta = data_beta[data_beta != 0]
    
    desired_indices = [4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
    n_nodes = data.shape[1]
    combinations = list(itertools.combinations(range(n_nodes), 2))
    in_indices, out_indices = zip(*combinations)
    in_indices = np.array(in_indices)
    out_indices = np.array(out_indices)
    
    in_indices = in_indices[desired_indices]
    out_indices = out_indices[desired_indices]
    indices = (in_indices, out_indices)
    
    filtered_data_theta = data_theta[desired_indices]
    filtered_data_alpha = data_alpha[desired_indices]
    filtered_data_beta = data_beta[desired_indices]
    plot_data = [filtered_data_theta, filtered_data_alpha, filtered_data_beta]
    
    node_names = ['F7', 'F3', 'Fz', 'F4', 'F8', 'P3', 'Pz', 'P4', 'PO7', 'PO8']
    node_colors=['red', 'red', 'red', 'red', 'red', 'blue', 'blue', 'blue', 'blue', 'blue']
    plot_titles = ['Theta - PLV', 'Alpha - PLV', 'Beta - PLV']
    fig_names = [f'{filename}_plv_theta', f'{filename}_plv_alpha', f'{filename}_plv_beta']

    for i in range(3):
        plot_connectivity_circle(plot_data[i], node_names, node_colors=node_colors, indices=indices, fontsize_names=15, 
                                 node_edgecolor='w', textcolor='w', node_height=2, facecolor='k', linewidth=2, colormap='plasma', 
                                 fontsize_colorbar=10, colorbar_size=0.75, colorbar_pos=(0, 0.7), fontsize_title=25, title=plot_titles[i], 
                                 interactive=True, show=False, vmin=0, vmax=1)
        
        fig_name = f'{fig_names[i]}.png'
        fig_path = os.path.join(output_fig, fig_name)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.clf()
    
    return data_alpha, data_beta, data_theta

def Load_SpectralPowerFeatures(directory):
    
    '''
    Loads and normalizes spectral power features for Desktop and VR conditions.
    '''
    
    participants_to_drop = [2, 8, 37]
    baseline_Desktop, baseline_VR, trial_Desktop, trial_VR = [], [], [], []
    for subdir in sorted(os.listdir(directory))[1:]:
    # for subdir in sorted(os.listdir(directory))[1:]:
        for file in sorted(os.listdir(os.path.join(directory, subdir))):
            
            print('Subdir: ', subdir) 
            print('File: ', file.split('_')[0], file.split('_')[1])
            
            df = pd.read_csv(os.path.join(directory, subdir, file))
            df = df.drop(participants_to_drop)   
                          
            if subdir == 'Baseline' and file.startswith('Desktop'):                
                baseline_Desktop.append(df)
            elif subdir == 'Baseline' and file.startswith('VR'):
                baseline_VR.append(df)
            elif subdir == 'Trial' and file.startswith('Desktop'):
                trial_Desktop.append(df)
            else:
                trial_VR.append(df)
    
    # baseline normalization for the (relative) spectral power features
    normalized_Desktop = [normalize_data(baseline, trial) for baseline, trial in zip(baseline_Desktop, trial_Desktop)]
    normalized_VR = [normalize_data(baseline, trial) for baseline, trial in zip(baseline_VR, trial_VR)]

    return normalized_Desktop, normalized_VR

def CombineFeatures(Desktop_freq, VR_freq, Desktop_plv, VR_plv):
    
    '''
    Combines spectral power features and Phase Locking Value (PLV) features, returning combined feature sets for Desktop and VR conditions per participant.
    '''
    
    features_Desktop, features_VR = [], []
    for plv_Desktop, plv_VR, alpha_Desktop, alpha_VR, beta_Desktop, beta_VR, theta_Desktop, theta_VR in zip(Desktop_plv, 
                                                                                                            VR_plv, 
                                                                                                            Desktop_freq[0].iterrows(), 
                                                                                                            VR_freq[0].iterrows(),
                                                                                                            Desktop_freq[1].iterrows(), 
                                                                                                            VR_freq[1].iterrows(),
                                                                                                            Desktop_freq[2].iterrows(), 
                                                                                                            VR_freq[2].iterrows()):

        # 135 PLV features + 42 spectral features
        features_Desktop.append(np.concatenate((plv_Desktop[0], 
                                                plv_Desktop[1], 
                                                plv_Desktop[2], 
                                                np.array(alpha_Desktop[1].tolist()), 
                                                np.array(beta_Desktop[1].tolist()), 
                                                np.array(theta_Desktop[1].tolist())
                                                )))
        features_VR.append(np.concatenate((plv_VR[0], 
                                           plv_VR[1], 
                                           plv_VR[2], 
                                           np.array(alpha_VR[1].tolist()), 
                                           np.array(beta_VR[1].tolist()), 
                                           np.array(theta_VR[1].tolist())
                                           )))
    
    return features_Desktop, features_VR

def Residuals_NASATLXSubset_BinarizingLabels_SaveFile(Desktop, VR, labels, labels_Desktop, labels_VR, output_dir):
    '''
    - Transforms continuous NASA-TLX scores into binary labels (0 = low workload, 1 = high workload).
    - Computes residuals based on a mixed-effects model, accounting for condition, task, and participant effects.
    - Saves features and binary labels per participant and condition.
    '''
    Desktop_labels = pd.read_csv(labels_Desktop, sep=',', header=None) # dropped participants = [2, 8, 37]
    VR_labels = pd.read_csv(labels_VR, sep=',', header=None) # dropped participants = [2, 8, 37]
    
    # Aggregation of Subscale ratings NASA-TLX workload
    temp_df_Desktop = np.mean([Desktop_labels[0], Desktop_labels[1], Desktop_labels[3], Desktop_labels[4]], axis=0)
    temp_df_VR = np.mean([VR_labels[0], VR_labels[1], VR_labels[3], VR_labels[4]], axis=0)
    new_df = pd.concat([pd.DataFrame(temp_df_Desktop), pd.DataFrame(temp_df_VR)], axis=0).reset_index(drop=True)
    new_df.columns = ['Workload']

    participants_to_drop = [2, 8, 37]
    df = pd.read_csv(labels, sep=',').drop(participants_to_drop)
    tasks = df['ppNumber'].str.split('_')
    task_order = tasks.apply(lambda x: x[2])
    task = task_order.apply(lambda x: x[2:])
    task = pd.DataFrame({'flight_task': task.reset_index(drop=True)})

    replacement = {'01': '02', '02': '01'}
    task2 = pd.DataFrame()
    task2['flight_task'] = task['flight_task'].replace(replacement)
    mapping = {'01': 'speed change', '02': 'medium turn'}

    new_df['Condition'] = ['DESKTOP'] * 49 + ['VR'] * 49
    new_df['Participant'] = list(range(1, 50)) * 2
    temp_df = pd.DataFrame()
    temp_df = pd.concat((task['flight_task'].map(mapping), task2['flight_task'].map(mapping)),axis=0).reset_index(drop=True)
    new_df['Flight_Task'] = temp_df
    
    # Fit a mixed-effects model with random effects of Participant
    model_random = smf.mixedlm("Workload ~ Condition + Flight_Task", new_df, groups=new_df["Participant"], re_formula="~1").fit()
    new_df['Residuals'] = model_random.resid
    
    median = np.median(new_df['Residuals'])
    
    print(median)
    
    Desktop_labels = new_df['Residuals'].iloc[:48]
    VR_labels = new_df['Residuals'].iloc[48:]
    data_Desktop = [{'X': data, 'label': np.array([0])} if label <= median else {'X': data, 'label': np.array([1])} for data, label in zip(Desktop, Desktop_labels)]
    data_VR = [{'X': data, 'label': np.array([0])} if label <= median else {'X': data, 'label': np.array([1])} for data, label in zip(VR, VR_labels)]
    
    for file_nr, Desktop, VR in zip(range(1,52-len(participants_to_drop)), data_Desktop, data_VR):
        filename_Desktop = f'{file_nr}_Desktop.npz'
        filename_VR = f'{file_nr}_VR.npz'
        np.savez(os.path.join(output_dir, 'DESKTOP', filename_Desktop), X=Desktop['X'], Y=Desktop['label'])
        np.savez(os.path.join(output_dir, 'VR', filename_VR), X=VR['X'], Y=VR['label'])

directory = '/Flight-Sim-Cognitive-Workload-EEG-Prediction/raw/Preprocessed_EEG_files'
labels = '/Flight-Sim-Cognitive-Workload-EEG-Prediction/raw/Table_logsandqrs.csv'
labels_Desktop = '/Flight-Sim-Cognitive-Workload-EEG-Prediction/raw/subworkload_Desktop.csv'
labels_VR = '/Flight-Sim-Cognitive-Workload-EEG-Prediction/raw/subworkload_VR.csv'
directory_frequency_bands = '/Flight-Sim-Cognitive-Workload-EEG-Prediction/raw/Absolute_EEG_power_values'
output_dir = '/Flight-Sim-Cognitive-Workload-EEG-Prediction/data'
output_fig = '/Flight-Sim-Cognitive-Workload-EEG-Prediction/results/connectivity_plots_per_sample'

# loading the (preprocessed) EEG time series data, processing it, and (Connectivity/PLV) feature extraction
Desktop_plv, VR_plv = LoadData_ProcessData_FeatureExtraction(directory, output_fig) 

# load the (absolute) spectral power features and output (relative) spectral power features
normalized_Desktop, normalized_VR = Load_SpectralPowerFeatures(directory_frequency_bands)

# combine both sets of features (Connectivity (PLV) features + (relative) Spectral Power features)
Desktop, VR = CombineFeatures(normalized_Desktop, normalized_VR, Desktop_plv, VR_plv)

Residuals_NASATLXSubset_BinarizingLabels_SaveFile(Desktop, VR, labels, labels_Desktop, labels_VR, output_dir)