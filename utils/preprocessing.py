import scipy
import copy
import os
import numpy as np

import database.neuralynx_extract.combinato_tools as ct
import database.neuralynx_extract.nlxio as nlxio
from statsmodels.stats import multitest

import utils.helper_func as hf

#variables used in functions
notch_freq = (50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0)
quality_factor = 30.0
fs_downs = 1024

def downsample(ncs_dir, patient_id, save_dir, channels, up, down):
    print('Create downsampled data file')
    for i in range(len(channels)):
        ch = channels[i]
        ncs_path = f'{ncs_dir}/{patient_id}/{ch}.ncs'
        
        with hf.suppress_stdout():
            print ("You cannot see this")
    
            # Create the ncsobj, which pulls and structures the information from the ncs file
            ncsobj = ct.ExtractNcsFile(ncs_path)

            # Parse out the data
            data, timestamps, timestep = ncsobj.read()

            downsampled_data = scipy.signal.resample_poly(data, up, down)
            np.save(f'{save_dir}/{patient_id}_channel-{ch}_downsampled_data.npy', downsampled_data)
        
    timestamps_downs_patient = np.linspace(timestamps[0], timestamps[-1], downsampled_data.shape[0])
    np.save(f'{save_dir}/{patient_id}_timestamps_downs.npy', timestamps_downs_patient)
    
    return
    
    
def filterSignal(x, fs, low, high, order):
    """
        Filter raw signal
        y = filterSignal(x, Fs, low, high) filters the signal x. Each column in x is one
        recording channel. Fs is the sampling frequency. low and high specify the passband in Hz.
        The filter delay is compensated in the output y.
    """
    if low == 0:
        # Nyquist frequency is the highest freq that the sampled signal can unambiguously represent
        nyq = 0.5 * fs
        # cut-off freq expressed as the fraction of the Nyquist freq
        high_cut = high / nyq

        b, a = scipy.signal.butter(order, high_cut, btype='lowpass')

        y = copy.deepcopy(x)

        # forward-backward filter. a linear filter that achieves zero phase delay by applying 
        # an IIR filter to a signal twice, once forwards and once backwards
        y = scipy.signal.filtfilt(b, a, x)

        return y
    
    else: 
        # Nyquist frequency is the highest freq that the sampled signal can unambiguously represent
        nyq = 0.5 * fs
        # cut-off freq expressed as the fraction of the Nyquist freq
        low_cut = low / nyq
        high_cut = high / nyq

        b, a = scipy.signal.butter(order, [low_cut, high_cut], btype='bandpass')

        y = copy.deepcopy(x)

        # forward-backward filter. a linear filter that achieves zero phase delay by applying 
        # an IIR filter to a signal twice, once forwards and once backwards
        y = scipy.signal.filtfilt(b, a, x)

        return y


def notch_filter(downsampled_data, patient_id, ch, save_dir=False):
    ### add **kwargs
    for freq in notch_freq:
        b_notch, a_notch = scipy.signal.iirnotch(freq, quality_factor, fs_downs)
        # Apply notch filter using signal.filtfilt
        downsampled_data = scipy.signal.filtfilt(b_notch, a_notch, downsampled_data) 

    if save_dir == True:    
        np.save(f'{save_dir}/{patient_id}_channel-{ch}_notch_filtered.npy', downsampled_data)
        return downsampled_data

    else:
        return downsampled_data


def prewhitening(x_array):
    """
    Prewhitening mean zero mean and unit standard deviation
    http://hosting.astro.cornell.edu/~cordes/A6523/Prewhitening.pdf
    """
    x_whiten = (x_array - x_array.mean()) / x_array.std()
    
    return x_whiten


class Switcher(object):
    def filtering(self, downsampled_data, filtering_type):
        """Select filtering function"""
        #method_name = 'month_' + str(argument)
        method_name = filtering_type
        print(method_name)
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: "Invalid filtering")
        # Call the method as we return it
        return method(downsampled_data)

    def broadband(self, downsampled_data):
        return downsampled_data

    def slow_gamma(self, downsampled_data):
        filtered_data = filterSignal(downsampled_data, fs_downs, 25, 55, order=3)
        return filtered_data

    def fast_gamma(self, downsampled_data):
        filtered_data = filterSignal(downsampled_data, fs_downs, 60, 100, order=3)
        return filtered_data

    def theta(self, downsampled_data):
        filtered_data = filterSignal(downsampled_data, fs_downs, 4, 8, order=3)
        return filtered_data

    def lowpass_20(self, downsampled_data):
        filtered_data = filterSignal(downsampled_data, fs_downs, 0, 20, order=3)
        return filtered_data

    
def find_epochs_idx(timestamps_downs, stim_onsets_downs, df_stim_info, st, epoch_len):   
    idx = np.where(timestamps_downs == hf.find_nearest(timestamps_downs, df_stim_info.loc[st,'time']))[0][0]
    idx1 = np.where(timestamps_downs == timestamps_downs[idx])[0][0]-int(np.round(epoch_len/2))
    idx2 = np.where(timestamps_downs == timestamps_downs[idx])[0][0]+int((np.round(epoch_len/2)+1))
    return idx, idx1, idx2
    
    
def create_epochs_array(downsampled_data, timestamps_downs, df_stim_info, patient_id, ch, save_dir, epoch_len, filtering_type='broadband', notch_filtering=True):
    if notch_filtering == True:
        #filter out 50 Hz - time 0.1 sec
        #print('Do notch filtering')
        downsampled_data = notch_filter(downsampled_data, patient_id, ch, save_dir=False)
    
    #a = Switcher()
    #filtered_data = a.filtering(downsampled_data, filtering_type)
    
    if filtering_type == 'broadband':
        filtered_data = downsampled_data
    elif filtering_type == 'slow_gamma':
        filtered_data = filterSignal(downsampled_data, fs_downs, 25, 55, order=3)
    elif filtering_type == 'fast_gamma':
        filtered_data = filterSignal(downsampled_data, fs_downs, 60, 100, order=3)
    elif filtering_type == 'theta':
        filtered_data = filterSignal(downsampled_data, fs_downs, 4, 8, order=3)
    elif filtering_type == 'lowpass_20':
        filtered_data = filterSignal(downsampled_data, fs_downs, 0, 20, order=3)
    
    stim_onsets_downs = np.zeros(len(df_stim_info['stim_index']))
    ofset_ms = np.zeros(len(df_stim_info['stim_index']))
    epochs = np.zeros((len(df_stim_info['stim_index']), epoch_len))
    
    for st in df_stim_info['stim_index']:
        #idx = np.where(timestamps_downs == hf.find_nearest(timestamps_downs, df_stim_info.loc[st,'time']))[0][0]
        #idx1 = np.where(timestamps_downs == timestamps_downs[idx])[0][0]-int(np.round(epoch_len/2))
        #idx2 = np.where(timestamps_downs == timestamps_downs[idx])[0][0]+int((np.round(epoch_len/2)+1))       
        
        idx, idx1, idx2 = find_epochs_idx(timestamps_downs, stim_onsets_downs, df_stim_info, st, epoch_len)
        ofset_ms[st] = df_stim_info.loc[st,'time'] - timestamps_downs[idx] 
        stim_onsets_downs[st] = timestamps_downs[idx]
        
        epochs[st,:] = filtered_data[idx1:idx2]
    
    np.save(f'{save_dir}/{patient_id}_epochs_channel-{ch}.npy', epochs)    
    np.save(f'{save_dir}/{patient_id}_stim_onsets_downs.npy', stim_onsets_downs)
    np.save(f'{save_dir}/{patient_id}_ofset_ms.npy', ofset_ms)
    
    return epochs, stim_onsets_downs, ofset_ms

