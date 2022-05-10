import scipy
import os
import sys
import copy
import glob
import time 
import warnings
import sklearn
import numpy as np
import pandas as pd
from statsmodels.stats import multitest
from contextlib import contextmanager
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import RandomUnderSampler, NearMiss, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN
from imblearn.under_sampling import CondensedNearestNeighbour, OneSidedSelection, NeighbourhoodCleaningRule, InstanceHardnessThreshold
from sklearn.linear_model import LogisticRegression
    

top_dir = '/home/anastasia/epiphyte/anastasia/output'
n_channels = 80 #different for different patients
n_stim_present = 10
n_stim = 42

def create_data_df(df_stim_info, df_patient_info, all_stim, filtering_type, time_chunks, time_chunks_dict, columns_data_array, folder, n_pca_comp, scr_type, pca_mode=True):
    stim_list = []
    row_list = []

    for st in all_stim:
        df_part = df_stim_info.loc[(df_stim_info['position']==scr_type) & (df_stim_info['stim_id']==st)]
        df_part = df_part.reset_index(drop=True)
        current_stim_index = df_part.loc[0,'stim_id']
        current_stim_name = df_part.loc[0,'stim_name']
        current_stim_paradigm = df_part.loc[0,'paradigm']
        is_500_days = df_part.loc[0,'is_500_days']
        ls = [current_stim_index, current_stim_name, current_stim_paradigm, is_500_days]
        for g in range(10):
            row_list.append(ls)

        data_list = []
        #columns_data_array = []
        for j in filtering_type:
            for i in range(len(df_patient_info['channel_name'])):
                ch = df_patient_info.loc[i,'channel_name']
                ch_site = df_patient_info.loc[i,'recording_site']

                envelope_post = np.load(f'{top_dir}/{folder}/{j}/power/zscore/{ch}_{ch_site}/{ch}_{ch_site}_{current_stim_index}_{current_stim_name}_amplitude_envelope_zscore_{scr_type}.npy')

                for tm in time_chunks:
                    #columns_data_array.append(f'{j}_{ch}_{tm}_mean')
                    #columns_data_array.append(f'{j}_{ch}_{tm}_median')
                    #columns_data_array.append(f'{j}_{ch}_{tm}_variance')
                    ep_st= time_chunks_dict[f'{tm}_st']
                    ep_end= time_chunks_dict[f'{tm}_end']
                    
                    for k in range(10):
                        ep = np.mean(envelope_post[k, ep_st:ep_end])
                        data_list.append(ep)
                    for k in range(10):
                        ep = np.median(envelope_post[k, ep_st:ep_end])
                        data_list.append(ep)
                    for k in range(10):
                        ep = np.var(envelope_post[k, ep_st:ep_end])
                        data_list.append(ep)
                    for k in range(10):
                        ep = np.sqrt(np.mean(envelope_post[k, ep_st:ep_end]**2))
                        data_list.append(ep)                        
                        
                if pca_mode == True:
                    pca = PCA(n_components=n_pca_comp).fit(envelope_post[:,500:])
                    pca = pca.transform(envelope_post[:,500:])
                    for n in range(n_pca_comp):
                        for k in range(10):
                            data_list.append(pca[k,n])

        data_array = np.array(data_list)
        if pca_mode == True:
            size = n_channels*len(filtering_type)*len(time_chunks)+n_channels*len(filtering_type)*n_pca_comp+n_channels*len(filtering_type)*len(time_chunks)+n_channels*len(filtering_type)*len(time_chunks)+n_channels*len(filtering_type)*len(time_chunks)
            #size = n_channels*len(filtering_type)*len(time_chunks)+n_channels*len(filtering_type)*n_pca_comp
        else:
            size = n_channels*len(filtering_type)*len(time_chunks)+n_channels*len(filtering_type)
        data_array = data_array.reshape(size, n_stim_present)
        data_array = data_array.transpose()
        stim_list.append(data_array)

    stim_array = np.array(stim_list)
    stim_array = stim_array.reshape(n_stim*n_stim_present, size)


    df_stim_data = pd.DataFrame(stim_array, columns = columns_data_array)
    columns = ['stim_index', 'stim_name', 'stim_paradigm', 'is_500_days']
    df_stimuli = pd.DataFrame(row_list, columns=columns)

    df_data = pd.concat([df_stimuli, df_stim_data], axis=1, sort=False)
    
    return df_data, df_stimuli

def custom_SVM(x_train, y_train, K, combinations, kernel, random_seeds):
    time_start = time.time()
    avg_loss = []
    losses = []
    
    xs_folds = np.array_split(x_train, K)
    ys_folds = np.array_split(y_train, K)
    
    # Use a K-fold cross validation with different parameters
    # for all choices of parameters combinations
    for combination in combinations:
        # for number of folds
        errors = np.array([])

        for k in range(K):
            # Build new training set and train with the parameter combination
            xs_training_set = np.array([])
            ys_training_set = np.array([])

            for index in range(K):
                if index == k:
                    xs_test_set = xs_folds[index]
                    ys_test_set = ys_folds[index]
                if xs_training_set.size == 0:
                    xs_training_set = xs_folds[index]
                    ys_training_set = ys_folds[index]
                else:
                    xs_training_set = np.append(xs_training_set, xs_folds[index], axis=0)
                    ys_training_set = np.append(ys_training_set, ys_folds[index], axis=0)

            if kernel == 'linear':
                if (combination[3]=='l1') and (combination[2]=='squared_hinge'):
                    status = False
                elif (combination[3]=='l2') and (combination[2]=='hinge'):                        
                    status = True
                else:
                    status = False
                classifier = svm.LinearSVC(
                             random_state=combination[0],
                             C=combination[1],                                 
                             loss=combination[2],
                             penalty=combination[3],
                             dual = status
                                 ).fit(xs_training_set, ys_training_set)
            else:
                classifier = svm.SVC(C=combination[1],
                             kernel = kernel,
                             #gamma = combination[2],                                     
                             random_state=combination[0],        
                             decision_function_shape='ovo',
                             class_weight = 'balanced'
                                 ).fit(xs_training_set, ys_training_set)

            # Compute the validation error for this fold
            predictions = classifier.predict(xs_test_set)
            # Use 0-1-loss
            loss = np.sum(np.abs(predictions - ys_test_set))    
            errors = np.append(errors, loss)
            test_acc = classifier.score(xs_test_set, ys_test_set)
            test_acc_balanced = sklearn.metrics.balanced_accuracy_score(ys_test_set, predictions)
            f1 = f1_score(ys_test_set, predictions, average='weighted')
            cm = sklearn.metrics.confusion_matrix(ys_test_set, predictions)
            # Model Precision: what percentage of positive tuples are labeled as such?
            precision = sklearn.metrics.precision_score(ys_test_set, predictions)
            # Model Recall: what percentage of positive tuples are labelled as such?
            recall = sklearn.metrics.recall_score(ys_test_set, predictions)
            #Cohen's Kappa: clf's accuracy normalized by the imbalance of classes
            kappa = sklearn.metrics.cohen_kappa_score(ys_test_set, predictions)
            losses.append([test_acc, loss, f1, cm, precision, recall, kappa, test_acc_balanced, kernel, combination[1:]])

        # Compute the average validation error
        avg_loss.append([np.sum(errors) / K, combination])
        
    time_end = time.time()
    #print("To run this cell, we need {} seconds.".format(time_end - time_start))
    return losses, avg_loss, classifier

# Select the parameter combination that leads to the lowest loss
def find_params_with_lowest_loss(avg_loss, kernel):
    best_loss = float("inf")
    index_best_loss = -1
    for i in range(len(avg_loss)):
        if avg_loss[i][0] < best_loss:
            best_loss = avg_loss[i][0]
            index_best_loss = i

    best_params = avg_loss[index_best_loss][1]
    if kernel == 'linear':
        best_params = {'random_state': best_params[0],
                       'C': best_params[1],
                       'loss': best_params[2],
                       'penalty': best_params[3]}
    else:
        best_params = {'random_state': best_params[0],
                       'C': best_params[1],
                       #'gamma': best_params[3]
                      }
    return best_params

def find_classifier_with_lowest_loss(avg_loss):
    best_loss = float("inf")
    index_best_loss = -1
    for i in range(len(avg_loss)):
        if avg_loss[i][0] < best_loss:
            best_loss = avg_loss[i][0]
            index_best_loss = i

    best_params = avg_loss[index_best_loss][1:]
    return best_params

def plot_coefficients(classifier, feature_names, top_features=10):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    fig, ax1 = plt.subplots(1, 1, figsize=(12,10))
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    ax1.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    ax1.set_xticks(np.arange(1, 1 + 2 * top_features))
    ax1.set_xticklabels(feature_names[top_coefficients], rotation=60, ha='right')
    return fig

def plot_coefficients_nonlinear(classifier, feature_names, X_test, y_test, top=-1):
    features = np.array(feature_names)
    perm_importance = permutation_importance(classifier, X_test, y_test)
    sorted_idx = perm_importance.importances_mean.argsort()
    fig, ax1 = plt.subplots(1, 1, figsize=(12,10))
    ax1.bar(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
    ax1.set_xticks(np.arange(1, 1 + len(feature_names)))
    ax1.set_xticklabels(features[sorted_idx], rotation=60, ha='right')
    return fig

#print avg accuracies
def print_avg_accuracies_per_area(all_losses, comb_linear, comb_rest, kernels, brain_areas, movie_stim_percentage, fig_save_path):
    mean_accuracies = []
    comb_linear_labels = [str(i) for i in comb_linear]
    comb_rest_labels = [str(i) for i in comb_rest]
    for brain_area in brain_areas:
        one_area = []
        fig, axes = plt.subplots(2, 2, figsize =(14,12))
        axes = axes.reshape(-1)
        string = f'Accuracies of best classifiers for {brain_area}'
        fig.suptitle(string)
        for sublist in all_losses:
            if sublist[1] == brain_area:
                one_area.append(sublist[0])
        one_area = [item for sublist in one_area for item in sublist]
        for i in range(len(kernels)):
            kernel = kernels[i]
            one_area_one_clf = []
            mean_acc_kernel = []
            for sublist in one_area:
                if sublist[7] == kernel:
                    one_area_one_clf.append(sublist)
            if kernel == 'linear':
                comb = comb_linear 
                comb_labels = comb_linear_labels
            else:
                comb = comb_rest
                comb_labels = comb_rest_labels
            accuracies = []
            for c in comb:
                acc = []
                for sublist in one_area_one_clf:
                    if sublist[-1] == c:
                        acc.append(sublist[0])
                accuracies.append(acc)
            mean_accuracies.append(np.mean(acc))
            # Creating plot  
            axes[i].boxplot(accuracies, labels=comb_labels)
            axes[i].set_title(f'{kernel} kernel')
            axes[i].set_ylim(0,1)
            axes[i].axhline(movie_stim_percentage/100, c='r', linestyle='--')
            axes[i].set_xticklabels(labels=comb_labels, rotation=90)
        plt.tight_layout()
        fig.savefig(f'{fig_save_path}/{brain_area}_all_clfs_accuracies.png', facecolor='white', transparent=False)
        plt.close()
    
    return 


def print_avg_balanced_accuracies_per_area(all_losses, comb_linear, comb_rest, kernels, brain_areas, movie_stim_percentage, fig_save_path):
    mean_accuracies = []
    comb_linear_labels = [str(i) for i in comb_linear]
    comb_rest_labels = [str(i) for i in comb_rest]
    for brain_area in brain_areas:
        one_area = []
        fig, axes = plt.subplots(2, 2, figsize =(14,12))
        axes = axes.reshape(-1)
        string = f'Accuracies of best classifiers for {brain_area}'
        fig.suptitle(string)
        for sublist in all_losses:
            if sublist[1] == brain_area:
                one_area.append(sublist[0])
        one_area = [item for sublist in one_area for item in sublist]
        for i in range(len(kernels)):
            kernel = kernels[i]
            one_area_one_clf = []
            mean_acc_kernel = []
            for sublist in one_area:
                if sublist[7] == kernel:
                    one_area_one_clf.append(sublist)
            if kernel == 'linear':
                comb = comb_linear 
                comb_labels = comb_linear_labels
            else:
                comb = comb_rest
                comb_labels = comb_rest_labels
            accuracies = []
            for c in comb:
                acc = []
                for sublist in one_area_one_clf:
                    if sublist[-1] == c:
                        acc.append(sublist[-3])
                accuracies.append(acc)
            mean_accuracies.append(np.mean(acc))
            # Creating plot  
            axes[i].boxplot(accuracies, labels=comb_labels)
            axes[i].set_title(f'{kernel} kernel')
            axes[i].set_ylim(0,1)
            #axes[i].axhline(movie_stim_percentage/100, c='r', linestyle='--')
            axes[i].set_xticklabels(labels=comb_labels, rotation=90)
        plt.tight_layout()
        fig.savefig(f'{fig_save_path}/{brain_area}_all_clfs_accuracies.png', facecolor='white', transparent=False)
        plt.close()
    
    return 


def print_avg_kappas_per_area(all_losses, comb_linear, comb_rest, kernels, brain_areas, movie_stim_percentage, fig_save_path):
    mean_kappas = []
    comb_linear_labels = [str(i) for i in comb_linear]
    comb_rest_labels = [str(i) for i in comb_rest]
    for brain_area in brain_areas:
        one_area = []
        fig, axes = plt.subplots(2, 2, figsize =(14,12))
        axes = axes.reshape(-1)
        string = f'Cohen kappa scores of best classifiers for {brain_area}'
        fig.suptitle(string)
        for sublist in all_losses:
            if sublist[1] == brain_area:
                one_area.append(sublist[0])
        one_area = [item for sublist in one_area for item in sublist]
        for i in range(len(kernels)):
            kernel = kernels[i]
            one_area_one_clf = []
            mean_acc_kernel = []
            for sublist in one_area:
                if sublist[7] == kernel:
                    one_area_one_clf.append(sublist)
            if kernel == 'linear':
                comb = comb_linear 
                comb_labels = comb_linear_labels
            else:
                comb = comb_rest
                comb_labels = comb_rest_labels
            kappas = []
            for c in comb:
                kappa = []
                for sublist in one_area_one_clf:
                    if sublist[-1] == c:
                        kappa.append(sublist[6])
                kappas.append(kappa)
            mean_kappas.append(np.mean(kappa))
            # Creating plot
            axes[i].boxplot(kappas, labels=comb_labels)
            axes[i].set_title(f'{kernel} kernel')
            axes[i].set_ylim(0,1)
            axes[i].axhline(movie_stim_percentage/100, c='r', linestyle='--')
            axes[i].set_xticklabels(labels=comb_labels, rotation=90)
        plt.tight_layout()
        fig.savefig(f'{fig_save_path}/{brain_area}_all_clfs_kappas.png', facecolor='white', transparent=False)
        plt.close()

    return 

def print_avg_f1s_per_area(all_losses, comb_linear, comb_rest, kernels, brain_areas, movie_stim_percentage, fig_save_path):
    mean_f1s = []
    comb_linear_labels = [str(i) for i in comb_linear]
    comb_rest_labels = [str(i) for i in comb_rest]
    for brain_area in brain_areas:
        one_area = []
        fig, axes = plt.subplots(2, 2, figsize =(14,12))
        axes = axes.reshape(-1)
        string = f'F1 scores of best classifiers for {brain_area}'
        fig.suptitle(string)
        for sublist in all_losses:
            if sublist[1] == brain_area:
                one_area.append(sublist[0])
        one_area = [item for sublist in one_area for item in sublist]
        for i in range(len(kernels)):
            kernel = kernels[i]
            one_area_one_clf = []
            mean_acc_kernel = []
            for sublist in one_area:
                if sublist[7] == kernel:
                    one_area_one_clf.append(sublist)
            if kernel == 'linear':
                comb = comb_linear 
                comb_labels = comb_linear_labels
            else:
                comb = comb_rest
                comb_labels = comb_rest_labels
            f1s = []
            for c in comb:
                f1 = []
                for sublist in one_area_one_clf:
                    if sublist[-1] == c:
                        f1.append(sublist[2])
                f1s.append(f1)
            mean_f1s.append(np.mean(f1))
            # Creating plot
            axes[i].boxplot(f1s, labels=comb_labels)
            axes[i].set_title(f'{kernel} kernel')
            axes[i].set_ylim(0,1)
            axes[i].axhline(movie_stim_percentage/100, c='r', linestyle='--')
            axes[i].set_xticklabels(labels=comb_labels, rotation=90)
        plt.tight_layout()
        fig.savefig(f'{fig_save_path}/{brain_area}_all_clfs_f1s.png', facecolor='white', transparent=False)
        plt.close()

    return 


def print_avg_precisions_per_area(all_losses, comb_linear, comb_rest, kernels, brain_areas, movie_stim_percentage, fig_save_path):
    mean_f1s = []
    comb_linear_labels = [str(i) for i in comb_linear]
    comb_rest_labels = [str(i) for i in comb_rest]
    for brain_area in brain_areas:
        one_area = []
        fig, axes = plt.subplots(2, 2, figsize =(14,12))
        axes = axes.reshape(-1)
        string = f'F1 scores of best classifiers for {brain_area}'
        fig.suptitle(string)
        for sublist in all_losses:
            if sublist[1] == brain_area:
                one_area.append(sublist[0])
        one_area = [item for sublist in one_area for item in sublist]
        for i in range(len(kernels)):
            kernel = kernels[i]
            one_area_one_clf = []
            mean_acc_kernel = []
            for sublist in one_area:
                if sublist[7] == kernel:
                    one_area_one_clf.append(sublist)
            if kernel == 'linear':
                comb = comb_linear 
                comb_labels = comb_linear_labels
            else:
                comb = comb_rest
                comb_labels = comb_rest_labels
            f1s = []
            for c in comb:
                f1 = []
                for sublist in one_area_one_clf:
                    if sublist[-1] == c:
                        f1.append(sublist[4])
                f1s.append(f1)
            mean_f1s.append(np.mean(f1))
            # Creating plot
            axes[i].boxplot(f1s, labels=comb_labels)
            axes[i].set_title(f'{kernel} kernel')
            axes[i].set_ylim(0,1)
            axes[i].axhline(movie_stim_percentage/100, c='r', linestyle='--')
            axes[i].set_xticklabels(labels=comb_labels, rotation=90)
        plt.tight_layout()
        fig.savefig(f'{fig_save_path}/{brain_area}_all_clfs_f1s.png', facecolor='white', transparent=False)
        plt.close()

    return 


def print_avg_recalls_per_area(all_losses, comb_linear, comb_rest, kernels, brain_areas, movie_stim_percentage, fig_save_path):
    mean_f1s = []
    comb_linear_labels = [str(i) for i in comb_linear]
    comb_rest_labels = [str(i) for i in comb_rest]
    for brain_area in brain_areas:
        one_area = []
        fig, axes = plt.subplots(2, 2, figsize =(14,12))
        axes = axes.reshape(-1)
        string = f'F1 scores of best classifiers for {brain_area}'
        fig.suptitle(string)
        for sublist in all_losses:
            if sublist[1] == brain_area:
                one_area.append(sublist[0])
        one_area = [item for sublist in one_area for item in sublist]
        for i in range(len(kernels)):
            kernel = kernels[i]
            one_area_one_clf = []
            mean_acc_kernel = []
            for sublist in one_area:
                if sublist[7] == kernel:
                    one_area_one_clf.append(sublist)
            if kernel == 'linear':
                comb = comb_linear 
                comb_labels = comb_linear_labels
            else:
                comb = comb_rest
                comb_labels = comb_rest_labels
            f1s = []
            for c in comb:
                f1 = []
                for sublist in one_area_one_clf:
                    if sublist[-1] == c:
                        f1.append(sublist[5])
                f1s.append(f1)
            mean_f1s.append(np.mean(f1))
            # Creating plot
            axes[i].boxplot(f1s, labels=comb_labels)
            axes[i].set_title(f'{kernel} kernel')
            axes[i].set_ylim(0,1)
            axes[i].axhline(movie_stim_percentage/100, c='r', linestyle='--')
            axes[i].set_xticklabels(labels=comb_labels, rotation=90)
        plt.tight_layout()
        fig.savefig(f'{fig_save_path}/{brain_area}_all_clfs_f1s.png', facecolor='white', transparent=False)
        plt.close()

    return 

#select the best classifier with lowest loss

def select_clf_lowest_loss(all_losses, comb_linear, comb_rest, kernels, brain_areas):
    best_clfs = []
    for brain_area in brain_areas:
        print(f'brain area: {brain_area}')
        one_area = []
        mean_losses_per_area = []
        smallest_loss_per_clf = []
        dict1 = {}
        for sublist in all_losses:
            if sublist[1] == brain_area:
                one_area.append(sublist[0])
        one_area = [item for sublist in one_area for item in sublist]

        for i in range(len(kernels)):
            kernel = kernels[i]
            one_area_one_clf = []
            mean_losses_kernel = []
            for sublist in one_area:
                if sublist[7] == kernel:
                    one_area_one_clf.append(sublist)
            if kernel == 'linear':
                comb = copy.deepcopy(comb_linear)
            else:
                comb = copy.deepcopy(comb_rest)
            losses_per_clf = []
            for j in range(len(comb)):
                c = comb[j]
                lss = []
                labels = []
                for sublist in one_area_one_clf:
                    if sublist[-1] == c:
                        lss.append(sublist[1])
                losses_per_clf.append(lss)
                mean_losses_kernel.append(np.mean(losses_per_clf[j]))

            dictionary = dict(zip(mean_losses_kernel, comb))

            for key in dictionary:
                dictionary[key].append(kernel)

            dict1.update(dictionary)
            mean_losses_per_area.append(mean_losses_kernel)

        min_loss = min(min(mean_losses_per_area, key=min))
        best_clf = dict1[min_loss]
        best_clfs.append(best_clf)
        print(f'best classifier: {best_clf}')
        print(f'loss: {min_loss}')
        
    return best_clfs

#print best clfs per area
def print_best_clfs_per_area(all_losses, best_clfs, brain_areas, feature_type, stim_percentage, fig_save_path):
    accuracies = []
    mean_accuracies = []
    std_accuracies = []
    bal_accuracies = []
    mean_bal_accuracies = []
    std_bal_accuracies = []    
    f1s = []
    mean_f1s = []
    std_f1s = []
    kappas = []
    mean_kappas = []
    std_kappas = []
    recalls = []
    mean_recalls = []
    std_recalls = []
    precisions = []
    mean_precisions = []
    std_precisions = []
    for i in range(len(brain_areas)):
        brain_area = brain_areas[i]
        print(f'brain area: {brain_area}')
        best_clf = best_clfs[i]
        one_area = []

        for sublist in all_losses:
            if sublist[1] == brain_area:
                one_area.append(sublist[0])
        one_area = [item for sublist in one_area for item in sublist]

        acc = []
        bal_acc = []
        f1 = []
        kappa = []
        recall = []
        precision = []
        for sublist in one_area:
            #print(sublist)
            if (sublist[-2] == best_clf[-1]) and (sublist[-1] == best_clf[:-1]):
                acc.append(sublist[0])
                bal_acc.append(sublist[7])
                recall.append(sublist[5])
                precision.append(sublist[4])                
                f1.append(sublist[2])
                kappa.append(sublist[6])
        accuracies.append(acc)
        bal_accuracies.append(bal_acc)
        f1s.append(f1)
        kappas.append(kappa)
        precisions.append(precision)
        recalls.append(recall)
        mean_accuracies.append(np.mean(acc))
        std_accuracies.append(np.std(acc))
        mean_bal_accuracies.append(np.mean(bal_acc))
        std_bal_accuracies.append(np.std(bal_acc))        
        mean_recalls.append(np.mean(recall))
        std_recalls.append(np.std(recall))
        mean_precisions.append(np.mean(precision))
        std_precisions.append(np.std(precision))
        mean_f1s.append(np.mean(f1))
        std_f1s.append(np.std(f1))
        mean_kappas.append(np.mean(kappa))
        std_kappas.append(np.std(kappa))


        print(f'best classifier: {best_clf}')
        print(f'mean accuracy: {round(mean_accuracies[i], 2)}\u00B1{round(std_accuracies[i], 2)}')
        print(f'mean balanced accuracy: {round(mean_bal_accuracies[i], 2)}\u00B1{round(std_bal_accuracies[i], 2)}')
        print(f'mean recall: {round(mean_recalls[i], 2)}\u00B1{round(std_recalls[i], 2)}')
        print(f'mean precision: {round(mean_precisions[i], 2)}\u00B1{round(std_precisions[i], 2)}')
        print(f'mean F1: {round(mean_f1s[i], 2)}\u00B1{round(std_f1s[i], 2)}')
        print(f'mean kappa: {round(mean_kappas[i], 2)}\u00B1{round(std_kappas[i], 2)}')   

    #plot
    fig = plt.figure(figsize =(10, 7))
    string = f'Accuracies of best classifiers for {feature_type} feature type'
    fig.suptitle(string)
    plt.boxplot(accuracies, labels=brain_areas)
    plt.ylim(0,1)
    plt.axhline(stim_percentage/100, c='r', linestyle='--')
    plt.xticks(rotation=90)
    plt.show()
    fig.savefig(f'{fig_save_path}/Accuracies_of_best_classifiers_for_{feature_type}_feature_type.png', facecolor='white', transparent=False)
    
    fig = plt.figure(figsize =(10, 7))
    string = f'Accuracies of best classifiers for {feature_type} feature type'
    fig.suptitle(string)
    plt.boxplot(bal_accuracies, labels=brain_areas)
    plt.ylim(0,1)
    #plt.axhline(stim_percentage/100, c='r', linestyle='--')
    plt.xticks(rotation=90)
    plt.show()
    fig.savefig(f'{fig_save_path}/Accuracies_of_best_classifiers_for_{feature_type}_feature_type.png', facecolor='white', transparent=False)
    
    fig = plt.figure(figsize =(10, 7))
    string = f'Precision scores of best classifiers for {feature_type} feature type'
    fig.suptitle(string)
    plt.boxplot(precisions, labels=brain_areas)
    plt.ylim(0,1)
    #plt.axhline(stim_percentage/100, c='r', linestyle='--')
    plt.xticks(rotation=90)
    plt.show()
    fig.savefig(f'{fig_save_path}/precisions_of_best_classifiers_for_{feature_type}_feature_type.png', facecolor='white', transparent=False)
    
    fig = plt.figure(figsize =(10, 7))
    string = f'Recall scores of best classifiers for {feature_type} feature type'
    fig.suptitle(string)
    plt.boxplot(recalls, labels=brain_areas)
    plt.ylim(0,1)
    #plt.axhline(stim_percentage/100, c='r', linestyle='--')
    plt.xticks(rotation=90)
    plt.show()
    fig.savefig(f'{fig_save_path}/recalls_of_best_classifiers_for_{feature_type}_feature_type.png', facecolor='white', transparent=False)
    
    fig = plt.figure(figsize =(10, 7))
    string = f'F1 scores of best classifiers for {feature_type} feature type'
    fig.suptitle(string)
    plt.boxplot(f1s, labels=brain_areas)
    plt.ylim(0,1)
    #plt.axhline(stim_percentage/100, c='r', linestyle='--')
    plt.xticks(rotation=90)
    plt.show()
    fig.savefig(f'{fig_save_path}/F1s_of_best_classifiers_for_{feature_type}_feature_type.png', facecolor='white', transparent=False)
    
    fig = plt.figure(figsize =(10, 7))
    string = f'Cohen\'s kappa scores of best classifiers for {feature_type} feature type'
    fig.suptitle(string)
    plt.boxplot(kappas, labels=brain_areas)
    plt.ylim(-1,1)
    #plt.axhline(stim_percentage/100, c='r', linestyle='--')
    plt.xticks(rotation=90)
    plt.show()
    fig.savefig(f'{fig_save_path}/Kappas_of_best_classifiers_for_{feature_type}_feature_type.png', facecolor='white', transparent=False)   
    
    
def custom_SVM_one_run(X, y, combinations, kernel, rand_seed):
    time_start = time.time()
    avg_loss = []
    losses = []
    
    np.random.seed(rand_seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    
    for combination in combinations:
        errors = np.array([])

        if kernel == 'linear':
            if (combination[3]=='l1') and (combination[2]=='squared_hinge'):
                status = False
            elif (combination[3]=='l2') and (combination[2]=='hinge'):                        
                status = True
            else:
                status = False
            classifier = svm.LinearSVC(
                         random_state=combination[0],
                         C=combination[1],                                 
                         loss=combination[2],
                         penalty=combination[3],
                         dual = status
                             ).fit(X_train, y_train)
            #cv = CountVectorizer()
            #cv.fit(X_train)
            #fig = decoding.plot_coefficients(best_estimator, cv.get_feature_names(), top_features=20)
        else:
            classifier = svm.SVC(C=combination[1],
                         kernel = kernel,
                         #gamma = combination[2],                                     
                         random_state=combination[0],        
                         decision_function_shape='ovo',
                         class_weight = 'balanced'
                             ).fit(X_train, y_train)
            #perm_importance = permutation_importance(best_estimator, X_test, y_test)
            #cv = CountVectorizer()
            #cv.fit(X_train)
            #feature_names = cv.get_feature_names()
            #fig = decoding.plot_coefficients_nonlinear(best_estimator, feature_names, X_test, y_test)

        # Compute the validation error for this fold
        predictions = classifier.predict(X_test)
        # Use 0-1-loss
        loss = np.sum(np.abs(predictions - y_test))    
        errors = np.append(errors, loss)
        test_acc = classifier.score(X_test, y_test)
        test_acc_balanced = sklearn.metrics.balanced_accuracy_score(ys_test_set, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        cm = sklearn.metrics.confusion_matrix(y_test, predictions)
        # Model Precision: what percentage of positive tuples are labeled as such?
        precision = sklearn.metrics.precision_score(y_test, predictions)
        # Model Recall: what percentage of positive tuples are labelled as such?
        recall = sklearn.metrics.recall_score(y_test, predictions)
        #Cohen's Kappa: clf's accuracy normalized by the imbalance of classes
        kappa = sklearn.metrics.cohen_kappa_score(y_test, predictions)
        
        losses.append([test_acc, loss, f1, cm, precision, recall, kappa, test_acc_balanced, kernel, combination[1:]])

    time_end = time.time()
    #print("To run this cell, we need {} seconds.".format(time_end - time_start))
    return losses, classifier

def custom_SVM_one_run_resampling(X, y, combinations, kernel, rand_seed, sampling):
    time_start = time.time()
    avg_loss = []
    losses = []
    
    np.random.seed(rand_seed)
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, train_size=0.8)
    
    if sampling == 'oversampling':
        rs = RandomOverSampler(random_state=0)
    elif sampling == 'SMOTE':
        rs = SMOTE(random_state=0)
    elif sampling == 'SMOTEENN':
        rs = SMOTEENN(random_state=0)
    elif sampling == 'SMOTETomek':
        rs = SMOTEENN(random_state=0)      
    elif sampling == 'undersampling':
        rs = RandomUnderSampler(random_state=0)
    elif sampling == 'NearMiss':
        rs = NearMiss(version=1)
    elif sampling == 'NNmode':
        rs = EditedNearestNeighbours(kind_sel='mode')
    elif sampling == 'NNall':
        rs = EditedNearestNeighbours(kind_sel='all')
    elif sampling == 'RepeatedNN':
        rs = RepeatedEditedNearestNeighbours()
    elif sampling == 'AllKNN':
        rs = AllKNN()
    elif sampling == 'CondensedNN':
        rs = CondensedNearestNeighbour(random_state=0) 
    elif sampling == 'OneSidedSelection':
        rs = OneSidedSelection(random_state=0)
    elif sampling == 'NeighbourhoodCleaningRule':
        rs = NeighbourhoodCleaningRule(sampling_strategy='all')
    elif sampling == 'InstanceHardnessThreshold':
        rs = InstanceHardnessThreshold(random_state=0,estimator=LogisticRegression(solver='lbfgs', multi_class='auto'))
        
    X_train, y_train = rs.fit_resample(X_tr, y_tr)
    
    for combination in combinations:
        errors = np.array([])

        if kernel == 'linear':
            if (combination[3]=='l1') and (combination[2]=='squared_hinge'):
                status = False
            elif (combination[3]=='l2') and (combination[2]=='hinge'):                        
                status = True
            else:
                status = False
            classifier = svm.LinearSVC(
                         random_state=combination[0],
                         C=combination[1],                                 
                         loss=combination[2],
                         penalty=combination[3],
                         dual = status
                             ).fit(X_train, y_train)
            #cv = CountVectorizer()
            #cv.fit(X_train)
            #fig = decoding.plot_coefficients(best_estimator, cv.get_feature_names(), top_features=20)
        else:
            classifier = svm.SVC(C=combination[1],
                         kernel = kernel,
                         #gamma = combination[2],                                     
                         random_state=combination[0],        
                         decision_function_shape='ovo',
                         class_weight = 'balanced'
                             ).fit(X_train, y_train)
            #perm_importance = permutation_importance(best_estimator, X_test, y_test)
            #cv = CountVectorizer()
            #cv.fit(X_train)
            #feature_names = cv.get_feature_names()
            #fig = decoding.plot_coefficients_nonlinear(best_estimator, feature_names, X_test, y_test)

        # Compute the validation error for this fold
        predictions = classifier.predict(X_test)
        # Use 0-1-loss
        loss = np.sum(np.abs(predictions - y_test))    
        errors = np.append(errors, loss)
        test_acc = classifier.score(X_test, y_test)
        test_acc_balanced = sklearn.metrics.balanced_accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        cm = sklearn.metrics.confusion_matrix(y_test, predictions)
        # Model Precision: what percentage of positive tuples are labeled as such?
        precision = sklearn.metrics.precision_score(y_test, predictions)
        # Model Recall: what percentage of positive tuples are labelled as such?
        recall = sklearn.metrics.recall_score(y_test, predictions)
        #Cohen's Kappa: clf's accuracy normalized by the imbalance of classes
        kappa = sklearn.metrics.cohen_kappa_score(y_test, predictions)
        
        losses.append([test_acc, loss, f1, cm, precision, recall, kappa, test_acc_balanced, kernel, combination[1:]])

    time_end = time.time()
    #print("To run this cell, we need {} seconds.".format(time_end - time_start))
    return losses, classifier