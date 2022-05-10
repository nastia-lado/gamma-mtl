import scipy
import copy
import os
import numpy as np
from statsmodels.stats import multitest
from contextlib import contextmanager
import sys, os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne


top_dir = '/home/anastasia/epiphyte/anastasia/output'

#500 or 900 here
fs = 32768
up = 1
down=32
fs_downs = (up/down)*fs   #The resulting sample rate is up / down times the original sample rate.
dt = 1/fs_downs           # sampling period/time/interval or time resolution, often denoted as T
epoch_start = 512


def create_folder_structure(parent_dir, df_patient_info):
    for i in range(len(df_patient_info['channel_name'])):
        ch = df_patient_info.loc[i,'channel_name']
        ch_site = df_patient_info.loc[i,'recording_site']
        directory = f'{ch}_{ch_site}'
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
        
    return


def get_brain_area_info(df_patient_info, brain_area):
    """Return information for brain area"""
    selected_areas = df_patient_info.loc[df_patient_info['brain_area']==brain_area]
    return selected_areas


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
    
    
def find_nearest(array,value):
    """
        Find the neasrest values in the array
        Used for finding timepoints
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    
    return array[idx]


def create_epoch_array(indices, epochs, epoch_start):
    epoch = []
    for i in range(len(indices)):
        e = epochs[indices[i], epoch_start:]
        epoch.append(e)
    epochs_return = np.array(epoch)
    return epochs_return


def compute_baseline(sp_array, time_vec, t1, t2):
    idx1 = np.where(time_vec == find_nearest(time_vec, t1))[0][0]
    idx2 = np.where(time_vec == find_nearest(time_vec, t2))[0][0]        
    baseline = np.mean(sp_array[:,:,idx1:idx2], axis=2)
    
    return baseline
   

def find_significant_cells(p_values, alpha):
    if True in (p_values<=alpha):
        string = f'p<={alpha} significant results'
        a1, a2 = np.where(p_values<=alpha)
        #print(string)
        return string, a1, a2
    else:
        string = f"no significant results"
        #print(string)
        return np.array([]), np.array([]), np.array([])
    
    
def compute_baseline_per_channel(ch, ch_site, df_stim_info, sp_dir, estim, all_stim, time_vec, t1, t2, sp_type):    
    baseline_pre = np.zeros((len(all_stim),10,estim[0]))
    baseline_post = np.zeros((len(all_stim),10,estim[0]))
    
    for st in all_stim:
        pre = df_stim_info.loc[(df_stim_info['position']=='pre') & (df_stim_info['stim_id']==st)]
        pre = pre.reset_index(drop=True)
        
        current_stim_index = pre.loc[0,'stim_id']
        current_stim_name = pre.loc[0,'stim_name']
        current_stim_paradigm = pre.loc[0,'paradigm']
        
        if sp_type == 'raw':
            pre_sp_array = np.load(f'{sp_dir}/raw/{ch}_{ch_site}_{current_stim_index}_{current_stim_name}_pre.npy')
            post_sp_array = np.load(f'{sp_dir}/raw/{ch}_{ch_site}_{current_stim_index}_{current_stim_name}_post.npy')
            baseline_pre[st,:,:] = compute_baseline(pre_sp_array,time_vec,t1,t2)
            baseline_post[st,:,:] = compute_baseline(post_sp_array,time_vec, t1,t2)
        
        elif sp_type == 'log':
            pre_sp_array = np.load(f'{sp_dir}/log/{ch}_{ch_site}_{current_stim_index}_{current_stim_name}_pre.npy')
            post_sp_array = np.load(f'{sp_dir}/log/{ch}_{ch_site}_{current_stim_index}_{current_stim_name}_post.npy')
            baseline_pre[st,:,:] = compute_baseline(pre_sp_array,time_vec,t1,t2)
            baseline_post[st,:,:] = compute_baseline(post_sp_array,time_vec,t1,t2)
        
        else:
            print('Wrong spectrogram type')
            
        #np.save(f'{top_dir}/anastasia/output/05-spectrogram_FFT/baseline/{ch}_{ch_site}_pre.npy', baseline_pre)
        #np.save(f'{top_dir}/anastasia/output/05-spectrogram_FFT/baseline/{ch}_{ch_site}_post.npy', baseline_post)
        #np.save(f'{top_dir}/anastasia/output/05-spectrogram_FFT/baseline/log/{ch}_{ch_site}_pre.npy', baseline_pre_log)
        #np.save(f'{top_dir}/anastasia/output/05-spectrogram_FFT/baseline/log/{ch}_{ch_site}_post.npy', baseline_post_log)
    
    return baseline_pre, baseline_post
    
    
def db_normalize(sp_array, baseline):
    baseline_mean = np.mean(np.mean(baseline, axis=1), axis=0)
    ##dB power 10*np.log10
    norm_sp = 10*np.log10(sp_array/baseline_mean[None,:,None])
    
    return norm_sp
    
    
def baseline_zscore(sp_array, baseline_log):
    baseline_log_mean = np.mean(np.mean(baseline_log, axis=1), axis=0)
    baseline_log = baseline_log.reshape((baseline_log.shape[0]*baseline_log.shape[1]), baseline_log.shape[2])
    sp_array_zscore = (sp_array-baseline_log_mean[None,:,None])/np.std(baseline_log, axis=0)[None,:,None]
    
    return sp_array_zscore


def custom_ttest(pre_sp_array, post_sp_array, alpha, method, test_type = 'wilcoxon'):
    ps_pre_flat = np.zeros((10,pre_sp_array.shape[1]* pre_sp_array.shape[2]))
    ps_post_flat = np.zeros((10,pre_sp_array.shape[1]* pre_sp_array.shape[2]))
    for i in range(10):
        ps_pre_flat[i] = pre_sp_array[i,:,:].flatten()
        ps_post_flat[i] = post_sp_array[i,:,:].flatten()
        
    if test_type == 'ttest_rel':
        t_val, p_values = scipy.stats.ttest_rel(ps_pre_flat, ps_post_flat, 0)
    
    elif test_type == 'wilcoxon':
        t_values = []
        p_values = []

        for i in range(ps_pre_flat.shape[1]):
            t_val, p_val = scipy.stats.wilcoxon(ps_pre_flat[:,i], ps_post_flat[:,i])
            t_values.append(t_val)
            p_values.append(p_val)
        p_values = np.array(p_values)
        
    elif test_type == 'perm_rel':
        data = np.concatenate((ps_pre_flat, ps_post_flat), axis=0)
        T0, pval_corrected, H0 = mne.stats.permutation_t_test(data, 10000, n_jobs=6)
        pval_corrected = pval_corrected.reshape(pre_sp_array.shape[1],pre_sp_array.shape[2])
        return T0, pval_corrected, H0 
    
    else:
        print('Wrong test_type')
        return
    
    output = multitest.multipletests(p_values, alpha=alpha, method=method)
    reject = output[0]
    pval_corrected = output[1]
    reject = reject.reshape(pre_sp_array.shape[1],pre_sp_array.shape[2])
    p_values = p_values.reshape(pre_sp_array.shape[1],pre_sp_array.shape[2])
    pval_corrected = pval_corrected.reshape(pre_sp_array.shape[1],pre_sp_array.shape[2])
    return p_values, pval_corrected, reject


def compute_spectrogram(df_stim_info, st, estim, epochs, fs_downs, nperseg, nfft, noverlap):
    #for st in all_stim:
    pre = df_stim_info.loc[(df_stim_info['position']=='pre') & (df_stim_info['stim_id']==st)]
    pre = pre.reset_index(drop=True)
    post = df_stim_info.loc[(df_stim_info['position']=='post') & (df_stim_info['stim_id']==st)]
    pre_index = np.array(pre['stim_index'])
    post_index = np.array(post['stim_index'])

    current_stim_index = pre.loc[0,'stim_id']
    current_stim_name = pre.loc[0,'stim_name']
    current_stim_paradigm = pre.loc[0,'paradigm']

    pre_sp_array = np.zeros((len(pre_index),estim[0],estim[1]))
    post_sp_array = np.zeros((len(post_index),estim[0],estim[1]))
    for i in range(len(pre_index)):
        pre_epoch = epochs[pre_index[i], epoch_start:]
        post_epoch = epochs[post_index[i], epoch_start:]
        
        #Selects between computing the power spectral density (‘density’) where Sxx has units of V**2/Hz and computing the 
        #power spectrum (‘spectrum’) where Sxx has units of V**2, if x is measured in V and fs is measured in Hz. Defaults to ‘density’.
        freq,time_sp,pwr_pre = scipy.signal.spectrogram(pre_epoch,fs_downs, window='hanning', nperseg=nperseg,
                                                   nfft=nfft, noverlap=noverlap, detrend=False, scaling='density')
        
        freq,_,pwr_post = scipy.signal.spectrogram(post_epoch,fs_downs, window='hanning', nperseg=nperseg,
                                                   nfft=nfft, noverlap=noverlap, detrend=False, scaling='density')

        pre_sp_array[i,:,:] = pwr_pre
        post_sp_array[i,:,:] = pwr_post

    return pre_sp_array, post_sp_array, freq, time_sp, current_stim_index, current_stim_name


def compute_wavelet_spectrogram(df_stim_info, st, estim, epochs, widths, w):
    #for st in all_stim:
    pre = df_stim_info.loc[(df_stim_info['position']=='pre') & (df_stim_info['stim_id']==st)]
    pre = pre.reset_index(drop=True)
    post = df_stim_info.loc[(df_stim_info['position']=='post') & (df_stim_info['stim_id']==st)]
    pre_index = np.array(pre['stim_index'])
    post_index = np.array(post['stim_index'])

    current_stim_index = pre.loc[0,'stim_id']
    current_stim_name = pre.loc[0,'stim_name']
    current_stim_paradigm = pre.loc[0,'paradigm']

    cwtm_pre_array = np.zeros((len(pre_index),estim[0],estim[1]))
    cwtm_post_array = np.zeros((len(post_index),estim[0],estim[1]))
    
    for i in range(len(pre_index)):
        pre_epoch = epochs[pre_index[i], epoch_start:]
        post_epoch = epochs[post_index[i], epoch_start:]

        cwtm_pre = scipy.signal.cwt(pre_epoch, scipy.signal.morlet2, widths, w=w)
        cwtm_post = scipy.signal.cwt(post_epoch, scipy.signal.morlet2, widths, w=w)
        #cwtm_pre, freq = pywt.cwt(pre_epoch, 100, "morl", 1000)
        
        cwtm_pre_array[i,:,:] = np.abs(cwtm_pre)**2
        cwtm_post_array[i,:,:] = np.abs(cwtm_post)**2
 
    return cwtm_pre_array, cwtm_post_array, current_stim_index, current_stim_name


def make_hilbert_transform(epochs, pre_index, post_index, fs_downs):
    analytic_signal_pre = []
    amplitude_envelope_pre = []
    phase_envelope_pre = []
    instantaneous_phase_pre = []
    instantaneous_frequency_pre = []
    pre_epochs = []

    analytic_signal_post = []
    amplitude_envelope_post = []
    phase_envelope_post = []
    instantaneous_phase_post = []
    instantaneous_frequency_post = []
    post_epochs = []

    for i in range(len(pre_index)):
        pre_epoch = epochs[pre_index[i],epoch_start:]
        post_epoch = epochs[post_index[i],epoch_start:]
        pre_epochs.append(pre_epoch)
        post_epochs.append(post_epoch)

        analytic_signal_pre = scipy.signal.hilbert(pre_epoch)
        ampl_envelope_pre = np.abs(analytic_signal_pre)**2
        ph_envelope_pre = np.angle(analytic_signal_pre)
        inst_phase_pre = np.unwrap(np.angle(analytic_signal_pre))
        inst_frequency_pre = (np.diff(inst_phase_pre)/(2.0*np.pi) * fs_downs)

        amplitude_envelope_pre.append(ampl_envelope_pre)
        phase_envelope_pre.append(ph_envelope_pre)
        instantaneous_phase_pre.append(inst_phase_pre)
        instantaneous_frequency_pre.append(inst_frequency_pre)

        analytic_signal_post = scipy.signal.hilbert(post_epoch)
        ampl_envelope_post = np.abs(analytic_signal_post)**2
        ph_envelope_post = np.angle(analytic_signal_post)
        inst_phase_post = np.unwrap(np.angle(analytic_signal_post))
        inst_frequency_post = (np.diff(inst_phase_post)/(2.0*np.pi) * fs_downs)

        amplitude_envelope_post.append(ampl_envelope_post)
        phase_envelope_post.append(ph_envelope_post)
        instantaneous_phase_post.append(inst_phase_post)
        instantaneous_frequency_post.append(inst_frequency_post)
    
    amplitude_envelope_pre = np.array(amplitude_envelope_pre)
    instantaneous_frequency_pre = np.array(instantaneous_frequency_pre)
    pre_epochs = np.array(pre_epochs)
    amplitude_envelope_post = np.array(amplitude_envelope_post)
    instantaneous_frequency_post = np.array(instantaneous_frequency_post)
    post_epochs = np.array(post_epochs)
    phase_envelope_pre = np.array(phase_envelope_pre)
    phase_envelope_post = np.array(phase_envelope_post)
    
    return amplitude_envelope_pre, amplitude_envelope_post, instantaneous_frequency_pre, instantaneous_frequency_post, pre_epochs, post_epochs, phase_envelope_pre, phase_envelope_post
