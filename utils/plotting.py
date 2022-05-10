import scipy
import copy
import os
import numpy as np
from statsmodels.stats import multitest
from contextlib import contextmanager
import sys, os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import utils.helper_func as hf

top_dir = '/home/anastasia/epiphyte/anastasia/output'
fs = 32768
#The resulting sample rate is up / down times the original sample rate.
up = 1
down=32
fs_downs = (up/down)*fs
dt = 1/fs_downs      # sampling period/time/interval or time resolution, often denoted as T
#for 1.5 sec
epoch_len = np.int((1.5*fs_downs)+1)
t = np.linspace(-500, 1000, epoch_len)
time_vec = np.linspace(-1000, 1000, 2049)
epoch_start = np.where(time_vec == hf.find_nearest(time_vec, -500))[0][0]

def center_output_pcolormesh(t, freq):
    X,Y = np.meshgrid(t,freq)
    x = X[0,:]
    y = Y[:,0]
    dx=x[1]-x[0]
    dy=y[1]-y[0]
    xedge = np.arange(x[0]-0.5*dx, x[-1]+dx, dx)
    yedge = np.arange(y[0]-0.5*dy, y[-1]+dy, dy)
    return xedge, yedge


def make_one_spectrogram_plot(ax, t, freq, sp_array_mean, title):
    xedge, yedge = center_output_pcolormesh(t, freq)
    im = ax.pcolormesh(xedge, yedge, sp_array_mean, shading='auto', cmap='jet') #vmin=-1, vmax=1,
    #im = ax.pcolormesh(t,freq,sp_array_mean)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    ax.axis(ymin=0, ymax=200)
    return im


def create_colorbar(im, fig):
    #cax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im,shrink=0.2) #cax=cax,
    cbar.set_label('Power [dB]') #, rotation=270
    return

    
def make_one_epoch_plot(ax, epoch, t, title):
    #plot many line plots on one
    for a in range(epoch.shape[0]):
        ax.plot(t,epoch[a,:],color='darkgrey')
    ax.plot(t,np.mean(epoch, axis=0),color='k')
    ax.axvline(x=0, color='r')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (microvolts)")
    ax.set_title(title)
    return
    

def plot_epochs(df_patient_info, patient_id, df_stim_info, all_stim):
    for i in range(len(df_patient_info['channel_name'])):
        ch = df_patient_info.loc[i,'channel_name']
        ch_site = df_patient_info.loc[i,'recording_site']
        epochs = np.load(f'{top_dir}/01-preprocessed/epochs/lowpass_20_2_sec/{patient_id}_epochs_channel-{ch}.npy')
        
        figures_path = f'{top_dir}/01-preprocessed/epochs/lowpass_figures/{ch}_{ch_site}'
        if not os.path.exists(figures_path):
            os.makedirs(figures_path)
        
        for st in all_stim:
            pre = df_stim_info.loc[(df_stim_info['position']=='pre') & (df_stim_info['stim_id']==st)]
            pre = pre.reset_index(drop=True)
            post = df_stim_info.loc[(df_stim_info['position']=='post') & (df_stim_info['stim_id']==st)]
            pre_index = np.array(pre['stim_index'])
            post_index = np.array(post['stim_index'])
                      
            current_stim_index = pre.loc[0,'stim_id']
            current_stim_name = pre.loc[0,'stim_name']
            current_stim_paradigm = pre.loc[0,'paradigm']            

            pre_epoch = hf.create_epoch_array(pre_index, epochs, epoch_start)
            post_epoch = hf.create_epoch_array(post_index, epochs, epoch_start)

            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 5))
            title = f'channel: {ch}, recording site: {ch_site}, stim_index:{current_stim_index}, stim_name:{current_stim_name}, paradigm:{current_stim_paradigm}'
            file_name = f"{{'channel' : '{ch}', 'recording_site' : '{ch_site}', 'stim_index':'{current_stim_index}', 'stim_name':'{current_stim_name}', 'paradigm':'{current_stim_paradigm}'}}"
            fig.suptitle(title)
            
            make_one_epoch_plot(ax1, pre_epoch, t, title="pre movie")
            
            make_one_epoch_plot(ax2, post_epoch, t, title="post movie")
            
            fig.savefig(f'{figures_path}/{file_name}.png', facecolor='white', transparent=False)
            plt.close()
            
    return


#separate into 2 func to make more reusable
def plot_epochs_organized(df_patient_info, patient_id, df_stim_info, all_stim_name):
    for i in range(len(df_patient_info['channel_name'])):
        ch = df_patient_info.loc[i,'channel_name']
        ch_site = df_patient_info.loc[i,'recording_site']
        epochs = np.load(f'{top_dir}/01-preprocessed/epochs/lowpass_20_2_sec/{patient_id}_epochs_channel-{ch}.npy')
        
        figures_path = f'{top_dir}/01-preprocessed/epochs/lowpass_figures_organized/{ch}_{ch_site}'
        if not os.path.exists(figures_path):
            os.makedirs(figures_path)

        for j in range(len(all_stim_name)):
            all_st_type = df_stim_info.loc[df_stim_info['stim_name'] == all_stim_name[j]]
            all_st_type = np.unique(all_st_type['stim_id'])

            if len(all_st_type) == 1:
                fig = plt.figure(figsize=(15,4))
            elif len(all_st_type) == 6:
                fig = plt.figure(figsize=(15,14))
            else:
                fig = plt.figure(figsize=(15,10))
                
            outer = gridspec.GridSpec(len(all_st_type), 1, figure=fig, wspace=0.5, hspace=0.5)

            for k in range(len(all_st_type)):
                st = all_st_type[k]
                pre = df_stim_info.loc[(df_stim_info['position']=='pre') & (df_stim_info['stim_id']==st)]
                pre = pre.reset_index(drop=True)
                post = df_stim_info.loc[(df_stim_info['position']=='post') & (df_stim_info['stim_id']==st)]
                pre_index = np.array(pre['stim_index'])
                post_index = np.array(post['stim_index'])
           
                current_stim_index = pre.loc[0,'stim_id']
                current_stim_name = pre.loc[0,'stim_name']
                current_stim_paradigm = pre.loc[0,'paradigm']            

                pre_epoch = hf.create_epoch_array(pre_index, epochs, epoch_start)
                post_epoch = hf.create_epoch_array(post_index, epochs, epoch_start)

                inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[k], wspace=0.1, hspace=0.1)

                title = f'channel: {ch}, recording site: {ch_site}, stim_name:{current_stim_name}, paradigm:{current_stim_paradigm}'
                file_name = f"{{'channel'! '{ch}', 'recording_site'! '{ch_site}', 'stim_name'!'{current_stim_name}', 'paradigm'!'{current_stim_paradigm}'}}"
                fig.suptitle(title)

                ax1 = fig.add_subplot(inner[0])
                make_one_epoch_plot(ax1, pre_epoch, t, title=f'pre movie, stim_index: {current_stim_index}')

                ax2 = fig.add_subplot(inner[1], sharey=ax1)
                make_one_epoch_plot(ax2, post_epoch, t, title=f'post movie, stim_index: {current_stim_index}')
                plt.setp(ax2.get_yticklabels(), visible=False)

            fig.savefig(f'{figures_path}/{file_name}.png', facecolor='white', transparent=False)
            plt.close()
            #plt.show()
    return


def plot_organized_spectrograms(df_stim_info, all_stim_name, t, freq, idx, ch, ch_site, folder, norm_type, alpha, method, test_type):
    list_strings = []
    
    figures_path = f'{top_dir}/{folder}/plots/{norm_type}/{ch}_{ch_site}'
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)
    
    for j in range(len(all_stim_name)):
        all_st_type = df_stim_info.loc[df_stim_info['stim_name'] == all_stim_name[j]]
        all_st_type = np.unique(all_st_type['stim_id'])

        if len(all_st_type) == 1:
            fig = plt.figure(figsize=(20,3.5))
        elif len(all_st_type) == 4:
            fig = plt.figure(figsize=(20,13))
        elif len(all_st_type) == 6:
            fig = plt.figure(figsize=(20,16))
        else:
            fig = plt.figure(figsize=(20,12))

        outer = gridspec.GridSpec(len(all_st_type), 1, figure=fig, wspace=0.3, hspace=0.3)

        k=0
        for st in all_st_type:
            pre = df_stim_info.loc[(df_stim_info['position']=='pre') & (df_stim_info['stim_id']==st)]
            pre = pre.reset_index(drop=True)
            post = df_stim_info.loc[(df_stim_info['position']=='post') & (df_stim_info['stim_id']==st)]
            pre_index = np.array(pre['stim_index'])
            post_index = np.array(post['stim_index'])

            current_stim_index = pre.loc[0,'stim_id']
            current_stim_name = pre.loc[0,'stim_name']
            current_stim_paradigm = pre.loc[0,'paradigm']

            pre_sp_array = np.load(f'{top_dir}/{folder}/spectrograms/normalized/{norm_type}/{ch}_{ch_site}_{current_stim_index}_{current_stim_name}_pre_{norm_type}.npy')
            post_sp_array = np.load(f'{top_dir}/{folder}/spectrograms/normalized/{norm_type}/{ch}_{ch_site}_{current_stim_index}_{current_stim_name}_post_{norm_type}.npy')

            #we will work only with frequencies below 200
            pre_sp_array = pre_sp_array[:,0:idx+1,:]
            post_sp_array = post_sp_array[:,0:idx+1,:]

            #take mean
            pre_sp_array_mean = np.mean(pre_sp_array, axis=0)
            post_sp_array_mean = np.mean(post_sp_array, axis=0)
            diff = post_sp_array_mean - pre_sp_array_mean
            
            #t-test
            p_values, pval_corrected, reject = hf.custom_ttest(pre_sp_array, post_sp_array, alpha, method, test_type=test_type)      
            string, a1, a2 = hf.find_significant_cells(pval_corrected, alpha)
            if a1.size > 0:
                str = (ch, ch_site, current_stim_index, current_stim_name, current_stim_paradigm, string, a1, a2)
                list_strings.append(str)
                #print(str)
            del string, a1, a2
            # gridspec inside gridspec    
            inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[k], wspace=0.1, hspace=0.1)

            title = f'Spectrogram channel: {ch}, recording_site: {ch_site}, stim_name:{current_stim_name}'
            file_name = f"{{'spectrogram_channel'!'{ch}', 'recording_site'!'{ch_site}', 'stim_name'!'{current_stim_name}'}}"
            fig.suptitle(title)  

            ax1 = fig.add_subplot(inner[0])
            im = make_one_spectrogram_plot(ax1, t, freq, pre_sp_array_mean, title='pre')
            cbar = fig.colorbar(im) #shrink=0.2
            #cbar.set_label('Power [dB]') #, rotation=270
            ax2 = fig.add_subplot(inner[1])
            im = make_one_spectrogram_plot(ax2, t, freq, post_sp_array_mean, title=f'post, stim_id: {current_stim_index}')
            ax2.contour(t, freq, pval_corrected<=alpha, 0, colors='black')
            cbar = fig.colorbar(im) #shrink=0.2
            #cbar.set_label('Power [dB]') #, rotation=270
            ax3 = fig.add_subplot(inner[2])
            im = make_one_spectrogram_plot(ax3, t, freq, diff, title=f'diff, paradigm:{current_stim_paradigm}')
            ax3.contour(t, freq, pval_corrected<=alpha, 0, colors='black')
            #plotting.create_colorbar(im, fig)
            #cax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
            cbar = fig.colorbar(im) #shrink=0.2
            #cbar.set_label('Power [dB]') #, rotation=270

            plt.setp(ax2.get_yticklabels(), visible=False)
            plt.setp(ax2.set_ylabel('Freq'), visible=False)
            plt.setp(ax3.get_yticklabels(), visible=False)
            plt.setp(ax3.set_ylabel('Freq'), visible=False)

            k=k+1    

        fig.savefig(f'{figures_path}/{file_name}.png', facecolor='white', transparent=False)
        plt.close()
        
    return list_strings


def plot_all_spectrograms_separately(df_stim_info, all_stim_name, t, freq, idx, ch, ch_site, folder, norm_type, alpha, method):
    list_strings = []
    
    figures_path = f'{top_dir}/{folder}/separate_plots/{norm_type}/{ch}_{ch_site}'
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)
    
    for st in all_stim:
        pre = df_stim_info.loc[(df_stim_info['position']=='pre') & (df_stim_info['stim_id']==st)]
        pre = pre.reset_index(drop=True)
        post = df_stim_info.loc[(df_stim_info['position']=='post') & (df_stim_info['stim_id']==st)]
        
        current_stim_index = pre.loc[0,'stim_id']
        current_stim_name = pre.loc[0,'stim_name']
        current_stim_paradigm = pre.loc[0,'paradigm']
        
        pre_sp_array = np.load(f'{top_dir}/{folder}/spectrograms/normalized/{norm_type}/{ch}_{ch_site}_{current_stim_index}_{current_stim_name}_pre_{norm_type}.npy')
        post_sp_array = np.load(f'{top_dir}/{folder}/spectrograms/normalized/{norm_type}/{ch}_{ch_site}_{current_stim_index}_{current_stim_name}_post_{norm_type}.npy')
        
        #we will work only with frequencies below 200
        pre_sp_array = pre_sp_array[:,0:idx+1,:]
        post_sp_array = post_sp_array[:,0:idx+1,:]
        
        #take mean
        pre_sp_array_mean = np.mean(pre_sp_array, axis=0)
        post_sp_array_mean = np.mean(post_sp_array, axis=0)
        diff = post_sp_array_mean - pre_sp_array_mean
        
        #t-test
        p_values, pval_corrected, reject = hf.custom_ttest(pre_sp_array_db, post_sp_array_db, alpha, method)      
        
        string, a1, a2 = hf.find_significant_cells(p_values, alpha)
        if a1.size > 0:
            list_strings_db.append((ch, ch_site, current_stim_index, current_stim_name, current_stim_paradigm, string, a1, a2))
            
        file_name = f"{{'spectrogram_channel'! '{ch}', 'recording_site'! '{ch_site}', 'stim_index'!'{current_stim_index}', 'stim_name'!'{current_stim_name}', 'paradigm'!'{current_stim_paradigm}'}}"
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 3.5))
        title = f'Spectrogram channel: {ch}, recording_site: {ch_site}, stim_index:{current_stim_index}, stim_name:{current_stim_name}, paradigm:{current_stim_paradigm}'
        fig.suptitle(title)  
        
        im = plotting.make_one_spectrogram_plot(ax1, t, freq, pre_sp_array_mean)
        cbar = fig.colorbar(im)
        im = plotting.make_one_spectrogram_plot(ax2, t, freq, post_sp_array_mean)
        cbar = fig.colorbar(im)
        ax2.contour(t, freq, p_values<=0.001, 0, colors='black')
        im = plotting.make_one_spectrogram_plot(ax3, t, freq, diff)
        #plotting.create_colorbar(im, fig)
        cbar = fig.colorbar(im)
        
        fig.savefig(f'{figures_path}/{file_name}.png', facecolor='white', transparent=False)
        plt.close()
    return list_strings


"""
Functions for plotting the screening results. 
Adapted from Johannes Niediek's original standalone code to interface withe database 
"""

import numpy as np
import matplotlib.pyplot as mpl
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

# local imports
from database.db_setup import *

from visualization.scr_utils import *
import analysis.stats_code.compute_pvalues as cp

# constants 

EMPTY = np.array([])
OUTFOLDER = 'pvalues'

PVAL_CUTOFF = .1
MIN_ACTIVE_TRIALS = 3

FIGSIZE = (12, 12)
T_PRE = 1000
T_POST = 2000
HIST_BIN_WIDTH = 100
HIST_BINS = np.arange(-T_PRE, T_POST + 1, HIST_BIN_WIDTH)

POSITIONS = ('pre', 'post')
COLORS = ('blue', 'green')

DPI = 200


#################################
## Screening Plotting Functions #
#################################


def plot_one_raster(plot, plot_hist, sp_times, event_data, pvalues):
    
    sep = .2
    hist_max = 0
    time_data = []
    color_data = []
    hist_data = defaultdict(list)

    for name, color in zip(POSITIONS, COLORS):
        events = event_data.loc[event_data.position == name, 'time'].values
        for event in events:
            idx_sp = (sp_times >= event - T_PRE) & (sp_times <= event + T_POST)
            if idx_sp.any():
                temp = sp_times[idx_sp] - event

                time_data.append(temp)
                t_hist, _ = np.histogram(temp, HIST_BINS)

                hist_data[name].append(t_hist * 1000 / HIST_BIN_WIDTH)
            else:
                time_data.append([-2*T_PRE])   # hack of eventplot

            color_data.append(color)

    # immediately plot the histogram
    for name, color in zip(POSITIONS, COLORS):
        if len(hist_data[name]):
            hist = np.vstack(hist_data[name]).mean(0)
            hist_max = np.max((hist.max(), hist_max))
            plot_hist.bar(HIST_BINS[:-1], hist, width=HIST_BIN_WIDTH, facecolor=color,
                lw=0, alpha=.5)


    plot.eventplot(time_data, color=color_data, lw=1)
    # plot.invert_yaxis()

    plot_hist.set_xticks((0, 1000))

    for pos in ('bottom', 'top', 'right'):
        plot.spines[pos].set_visible(False)
        plot_hist.spines[pos].set_visible(False)

    plot.axis('off')

    for pos in (0, 1000):
        for pl in (plot, plot_hist):
            pl.axvline(pos, ls='--', color='k', lw=1, alpha=.8)

    plot.set_xlim((-T_PRE, T_POST))
    plot.set_ylim((-.5, len(time_data) + .5))
    plot.set_yticks([])
    plot.set_xticks((0, 1000))
    plot.set_xticklabels([])

    has_response = {}

    for name, namecolor in zip(POSITIONS, COLORS):
        has_response[name] = False
        idx = pvalues.position == name
        #print(pvalues)
        #print(pvalues.position)
        #print(name)
        #print(idx)
        assert idx.sum() == 1

        if name == 'pre':
            xpos = -.05
            ha = 'left'
        elif name == 'post':
            xpos = 1.05
            ha = 'right'

        active_t = pvalues.loc[idx, 'active_trials'].values[0]

        #### here, modify the pvalues shown. 
        ## before doing so, make sure it'd be worth it by comparing 
        ## the bwsr and the perm pvalue results
        for short_name, value_name, ypos in zip(('BW', 'SCR', 'A'),
                ('pval_bwsr', 'pval_scr', 'active_trials'),
                (1, 1 + sep, 1 + 2 * sep)):
            value = pvalues.loc[idx, value_name].values[0]
            color = 'k'
            if short_name in ('BW', 'SCR'):
                my_format = '.4f'
                # print(value, active_t)
                if (value < PVAL_CUTOFF) and (active_t > MIN_ACTIVE_TRIALS):
                    color = 'g'
                    has_response[name] = True
            else:
                my_format = 'd'

            title = format(value, my_format)

            if name == 'pre':
                title = short_name[0] + ' ' + title

            plot.text(xpos, ypos, title, color=color, transform=plot.transAxes, ha=ha, va='bottom', size=7)

        plot.text(xpos, 1 + 3 * sep, name, ha=ha, va='bottom', size=9, transform=plot.transAxes, color=namecolor)

    ret_val = False
    if has_response['pre'] or has_response['post']:
        if has_response['pre'] and has_response['post']:
            color = 'green'
        elif has_response['pre'] and not has_response['post']:
            color = 'blue'
        elif not has_response['pre'] and has_response['post']:
            color = 'red'
            ret_val = True

        rect = Rectangle((0, 0), 1, 1, edgecolor=color, transform=plot.transAxes,
            lw=4, facecolor='none')
        plot.add_patch(rect)

    # return whether this is an interesting example (additional response)
    return ret_val, hist_max


def plot_one_unit(fig, grid, title, stim_frame, spikes, unit_pvals, stim_data):
    """
    stim_data is a filename -> (a, b, c, d) dictionary, where
    a: stim_num
    b: stim_name
    c: paradigm
    d: image
    """

    all_hists = []
    times = spikes
    save_unit = False

    info_plot = fig.add_axes([0, .9, .8, .1])
    info_plot.axis('off')

#     cluster_plot = fig.add_axes([.9, .92, .095, .078])
#    # spu.spike_heatmap(cluster_plot, spikes)
#     cluster_plot.set_xticks([])
#     cluster_plot.set_xticklabels([])
#     cluster_plot.set_ylabel('µV')
    num_stimuli = len(np.unique(stim_frame["filename"]))
    
    if num_stimuli == 42:
        img_order = IMAGE_ORDER_ALL
    elif num_stimuli == 35:
        img_order = IMAGE_ORDER_NO_TEXT
    else: 
        raise Exception("Irregular number of stimuli for patient.")
    
    for i_row, row in enumerate(img_order):
        for i_col, img_fname in enumerate(row):
            stim_num, stim_name, paradigm, image = stim_data[img_fname]
            # idx_pval_stim = unit_pvals.stim_num == stim_num

            plot = fig.add_subplot(grid[3 * i_row, i_col])
            plot.axis('off')

            # show the image
            if paradigm == 'scr' :
                plot.imshow(image)
                plot.text(.5, 1, stim_name, transform=plot.transAxes,
                    va='bottom', ha='center', size=6)

            else:
                plot.text(.5, .5, stim_name, transform=plot.transAxes,
                    va='center', ha='center')

            # make the raster and histogram
            plot = fig.add_subplot(grid[3 * i_row + 1, i_col])
            plot_hist = fig.add_subplot(grid[3 * i_row + 2, i_col])

            if i_col > 0:
                plot_hist.set_yticklabels([])

            else:
                plot_hist.set_ylabel('Hz')

            if i_row + 1 < len(img_order):
                plot_hist.set_xticklabels([])
            
            # generate the raster plot for one stimulus
            # plot_one_raster(plot, plot_hist, sp_times, event_data, pvalues)
            is_interesting, hist_max = plot_one_raster(plot, plot_hist,
                    times, stim_frame.loc[stim_frame.stim_num == stim_num, ['position', 'time']],
                    unit_pvals.loc[unit_pvals.stim_num == stim_num, :])

            if is_interesting:
                save_unit = True

            all_hists.append(plot_hist)

    info_plot.text(.5, .5, title, va='center', ha='center', size=12)
    if save_unit:
        rect = Rectangle((0, .25), .1, .5, facecolor='red', edgecolor='none',
            transform=info_plot.transAxes, alpha=.5)
        info_plot.add_patch(rect)

    # rescale the maximum
    for plot in all_hists:
        plot.set_ylim((0, hist_max * 1.1))
        plot.set_yticks((0, round(hist_max)))

    return save_unit

def run_one_channel(fig, save_folder, grid, frame, channel, patient_id, session_nr, stim_data):
    """
    load the pvalues and units for one channel, one sorting, and iterate over the units
    """
    
    # get all units from a single channel
    channel_units = get_unit_ids_in_channel(patient_id, session_nr, channel)
    
    for unit in channel_units:
        print("    Running unit {}..".format(unit))


        ### calculate pvalues for a unit 
        spikes = get_spiking_activity(patient_id, session_nr, unit)
        
        stim_index, eventtimes = get_scr_eventtimes(patient_id, session_nr)

        region = get_brain_region(patient_id, unit)
        unit_type = get_unit_type(patient_id, session_nr, unit)
    
        title = "{:03d}mv1 Unit #{}, Channel: {} ({})".format(patient_id, unit, channel, unit_type)

        unit_pvals = get_scr_stats_as_df(patient_id, session_nr, unit)
        
        print("    Total spikes: {}".format(len(spikes)))
        
        fig.clf()
                 
        #print(frame)
        is_interesting = plot_one_unit(fig, grid, title, frame, spikes, unit_pvals, stim_data)


        fname = '{:03d}mv1_CSC{:02d}_{}_unit{:03d}.jpeg'.format(patient_id,
                channel, unit_type, unit)

        fig.savefig(os.path.join(save_folder, fname), dpi=DPI, transparent=False)

        if is_interesting:
            with open(os.path.join(save_folder, 'interesting.txt'), 'a') as fid:
                fid.write(fname + '\n')
            fid.close()
            
def run_session(patient_id):
    """
    load stimulus frame and channel list for one session
    """  
    session_nr = get_session_info(patient_id)
    
    assert isinstance(session_nr, int), "More than one session for patient {}. Code currently not set up for automatically running multiple sessions from a single patient.".format(patient_id)
    
    position, stim_id, filename, stim_name, is_500_days, paradigm, time = get_screening_data(patient_id, session_nr)
    frame = cp.make_dataframe(position, stim_id, filename, stim_name, is_500_days, paradigm, time)

    stim_data = {}
    stim_nums = frame.stim_num.unique()

    for stim_num in stim_nums:
        # setting up to get the stimulus image
        meta = frame.loc[frame.stim_num == stim_num, :].iloc[0]
        stim_fname = meta["filename"]
        stim_data[stim_fname] = (stim_num,
                                meta["stim_name"],
                                meta["paradigm"],
                                mpl.imread(os.path.join(PATH_TO_IMAGES, stim_fname))) ## read image file into an array

    fig = mpl.figure(figsize=FIGSIZE)
    grid = GridSpec(18, 7, left=.06, right=.98, top=.9, bottom=.04)

    # init saving directory 
    save_folder = os.path.join(PATH_TO_PLOTS,
        'screenings', '{:03d}'.format(patient_id))
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    all_channels = get_cscs_for_patient(patient_id, session_nr)
        
    for channel in all_channels:
        print("Running channel {}...".format(channel))
        run_one_channel(fig, save_folder, grid, frame, channel, patient_id, session_nr, stim_data)

    mpl.close(fig)