import os
import pywt
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt


import os
import sys
import tqdm

import wfdb
from wfdb import rdsamp,rdann
#from utils import qrs_detect, comp_cosEn, save_dict


def character_label(num):
    sample_path = all_id.iloc[num,0]
    sig, fields = wfdb.rdsamp(sample_path)
    length = len(sig)
    if all_id['label'][num] == 'normal':
        labels = length * [0]
    elif all_id['label'][num] == 'AFf':
        labels = length * [1]
    elif all_id['label'][num] == 'AFp':
        annotation = wfdb.rdann(sample_path, 'atr').aux_note
        R_index = wfdb.rdann(sample_path, 'atr').sample
        if R_index[-1]>length:
            R_index[-1] = length
            print('modified!')
        annotation_valid = []
        labels = length * [0]
        for a in range(len(annotation)):
            if (annotation[a] != ''):
                annotation_valid.append(a)
        if len(annotation_valid) % 2 != 0 :
            raise
        num_pair = round(len(annotation_valid)/2)
        for i in range(num_pair):
            left = R_index[annotation_valid[2*i]]
            right = R_index[annotation_valid[2*i+1]]
            labels[left:right] = [1] * (right-left)
        assert len(labels) == length
        if annotation[0] != '':
            labels[0:R_index[0]] = [1] * R_index[0]
        if annotation[-1] != '':
            labels[R_index[-1]:] = [1] * (length - R_index[-1])
        if sum(labels) == 0:
            raise
    if len(labels) != length:
        raise
    return labels

if __name__ == '__main__':
    train_id = pd.read_csv('new_train_IDs.csv',index_col = 0).index.tolist()
    valid_id = pd.read_csv('new_valid_IDs.csv',index_col = 0).index.tolist()
    test_id = pd.read_csv('new_test_IDs.csv',index_col = 0).index.tolist()
    all_id = pd.read_csv('all_IDs.csv',index_col = 0)
    all_id['data_type'] = 0
    all_id.loc[valid_id,'data_type'] = 1
    all_id.loc[test_id,'data_type'] = 2


    proc_ecg_sen = {}
    sen_len = 1500
    for num in tqdm.tqdm(range(0,all_id.shape[0])):
        file = all_id.iloc[num,0]
        sig, fields = wfdb.rdsamp(file)
        length = len(sig)
        fs = fields['fs']

        sig = sig[:,1]
        c_label = character_label(num)
        data_type = all_id.iloc[num,5]
        if data_type == 0:
            n_sentence = round(np.floor(length/sen_len))
        else:
            n_sentence = round(np.ceil(length/sen_len))
        for i in range(n_sentence):
            key = str(num) + "_" + str(i) + "_" + "1"
            value = sig[i*sen_len:i*sen_len+sen_len]
            label = c_label[i*sen_len:i*sen_len+sen_len]
            length = len(value)
            assert length == len(label)
            if all_id['label'][num] == 'normal':
                series_label = 0
            elif all_id['label'][num] == 'AFf':
                series_label = 1
            elif all_id['label'][num] == 'AFp':
                if sum(label) == 0:
                    series_label = 2
                elif sum(label) == len(label):
                    series_label = 3
                elif sum(label)/len(label) <= 0.5:
                    series_label = 4
                elif sum(label)/len(label) > 0.5:
                    series_label = 5
            
            if data_type == 0:
                assert length == sen_len
            if min(value) == max(value):
                continue
                
            proc_ecg_sen[key] = (value,label,length,num,series_label,np.nan,data_type,1,i)

    np.save('proc_ecg_sentence_1500.npy',proc_ecg_sen)        
