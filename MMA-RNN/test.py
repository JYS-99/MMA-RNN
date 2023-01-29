import torch
import torch.nn as nn
import torch.optim as optim
import data
import models
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from sklearn.metrics import confusion_matrix as cm
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wfdb
from wfdb import rdsamp,rdann
from utils import qrs_detect,comp_cosEn, save_dict
import json
import shutil
import sys
import joblib
from sklearn.neural_network import MLPClassifier
import scipy.io as sio


log_name = 'log.txt'
path_name = '.'
data_dir = './data'

R = np.array([[1, -1, -.5], [-2, 1, 0], [-1, 0, 1]])

class RefInfo():
    def __init__(self, sample_path):
        self.sample_path = sample_path
        self.fs, self.len_sig, self.beat_loc, self.af_starts, self.af_ends, self.class_true = self._load_ref()
        self.endpoints_true = np.dstack((self.af_starts, self.af_ends))[0, :, :]
        # self.endpoints_true = np.concatenate((self.af_starts, self.af_ends), axis=-1)
        if self.class_true == 1 or self.class_true == 2:
            self.onset_score_range, self.offset_score_range = self._gen_endpoint_score_range()
        else:
            self.onset_score_range, self.offset_score_range = None, None
        self.endpoints_diy = []
        for i in range(len(self.af_starts)):
            self.endpoints_diy.append([self.beat_loc[self.af_starts[i]], self.beat_loc[self.af_ends[i]]])

    def _load_ref(self):
        sig, fields = wfdb.rdsamp(self.sample_path)
        ann_ref = wfdb.rdann(self.sample_path, 'atr')

        fs = fields['fs']
        length = len(sig)
        sample_descrip = fields['comments']

        beat_loc = np.array(ann_ref.sample)  # r-peak locations
        ann_note = np.array(ann_ref.aux_note)  # rhythm change flag

        af_start_scripts = np.where((ann_note == '(AFIB') | (ann_note == '(AFL'))[0]
        af_end_scripts = np.where(ann_note == '(N')[0]

        if 'non atrial fibrillation' in sample_descrip:
            class_true = 0
        elif 'persistent atrial fibrillation' in sample_descrip:
            class_true = 1
        elif 'paroxysmal atrial fibrillation' in sample_descrip:
            class_true = 2
        else:
            print('Error: the recording is out of range!')

            return -1

        return fs, length, beat_loc, af_start_scripts, af_end_scripts, class_true

    def _gen_endpoint_score_range(self):
        """

        """
        onset_range = np.zeros((self.len_sig + 1,), dtype=np.float)
        offset_range = np.zeros((self.len_sig + 1,), dtype=np.float)
        for i, af_start in enumerate(self.af_starts):
            if self.class_true == 2:
                if max(af_start - 1, 0) == 0:
                    onset_range[: self.beat_loc[af_start + 2]] += 1
                elif max(af_start - 2, 0) == 0:
                    onset_range[self.beat_loc[af_start - 1]: self.beat_loc[af_start + 2]] += 1
                    onset_range[: self.beat_loc[af_start - 1]] += .5
                else:
                    onset_range[self.beat_loc[af_start - 1]: self.beat_loc[af_start + 2]] += 1
                    onset_range[self.beat_loc[af_start - 2]: self.beat_loc[af_start - 1]] += .5
                onset_range[self.beat_loc[af_start + 2]: self.beat_loc[af_start + 3]] += .5
            elif self.class_true == 1:
                onset_range[: self.beat_loc[af_start + 2]] += 1
                onset_range[self.beat_loc[af_start + 2]: self.beat_loc[af_start + 3]] += .5
        for i, af_end in enumerate(self.af_ends):
            if self.class_true == 2:
                if min(af_end + 1, len(self.beat_loc) - 1) == len(self.beat_loc) - 1:
                    offset_range[self.beat_loc[af_end - 2]:] += 1
                elif min(af_end + 2, len(self.beat_loc) - 1) == len(self.beat_loc) - 1:
                    offset_range[self.beat_loc[af_end - 2]: self.beat_loc[af_end + 1]] += 1
                    offset_range[self.beat_loc[af_end + 1]:] += 0.5
                else:
                    offset_range[self.beat_loc[af_end - 2]: self.beat_loc[af_end + 1]] += 1
                    offset_range[self.beat_loc[af_end + 1]: min(self.beat_loc[af_end + 2], self.len_sig - 1)] += .5
                offset_range[self.beat_loc[af_end - 3]: self.beat_loc[af_end - 2]] += .5
            elif self.class_true == 1:
                offset_range[self.beat_loc[af_end - 2]:] += 1
                offset_range[self.beat_loc[af_end - 3]: self.beat_loc[af_end - 2]] += .5

        return onset_range, offset_range


def load_ans(ans_file):
    endpoints_pred = []
    if ans_file.endswith('.json'):
        json_file = open(ans_file, "r")
        ans_dic = json.load(json_file)
        endpoints_pred = np.array(ans_dic['predict_endpoints'])

    elif ans_file.endswith('.mat'):
        ans_struct = sio.loadmat(ans_file)
        endpoints_pred = ans_struct['predict_endpoints'] - 1

    return endpoints_pred


def ue_calculate(endpoints_pred, endpoints_true, onset_score_range, offset_score_range):
    score = 0
    ma = len(endpoints_true)
    mr = len(endpoints_pred)
    for [start, end] in endpoints_pred:
        score += onset_score_range[int(start)]
        score += offset_score_range[int(end)]

    score *= (ma / max(ma, mr))

    return score


def ur_calculate(class_true, class_pred):
    score = R[int(class_true), int(class_pred)]

    return score


def score(data_path, ans_path):
    # AF burden estimation
    SCORE = []
    UR_SCORE = []
    UE_SCORE = []

    def is_mat_or_json(file):
        return (file.endswith('.json')) + (file.endswith('.mat'))

    ans_set = filter(is_mat_or_json, os.listdir(ans_path))
    for i, ans_sample in enumerate(ans_set):
        sample_nam = ans_sample.split('.')[0]
        sample_path = os.path.join(data_path, sample_nam)
        endpoints_pred = load_ans(os.path.join(ans_path, ans_sample))
        TrueRef = RefInfo(sample_path)

        if len(endpoints_pred) == 0:
            class_pred = 0
        elif len(endpoints_pred) == 1 and np.diff(endpoints_pred)[-1] == TrueRef.len_sig - 1:
            class_pred = 1
        else:
            class_pred = 2

        ur_score = ur_calculate(TrueRef.class_true, class_pred)

        if TrueRef.class_true == 1 or TrueRef.class_true == 2:
            ue_score = ue_calculate(endpoints_pred, TrueRef.endpoints_true, TrueRef.onset_score_range,
                                    TrueRef.offset_score_range)
        else:
            ue_score = 0

        u = ur_score + ue_score
        SCORE.append(u)
        UR_SCORE.append(ur_score)
        UE_SCORE.append(ue_score)

    return SCORE, UR_SCORE, UE_SCORE


all_ids = pd.read_csv(data_dir + 'all_IDs.csv',index_col = 0)

def trans_y(y):
    if y == 'normal':
        return 0
    elif y == 'AFf':
        return 1
    elif y == 'AFp':
        return 2
    else:
        raise
def cnt_012(a):
    cnt_0 = 0
    cnt_1 = 0
    cnt_2 = 0
    for i in a:
        if i == 0:
            cnt_0 += 1
        elif i == 1:
            cnt_1 += 1
        elif i == 2:
            cnt_2 += 1
    return np.array([cnt_0,cnt_1,cnt_2])
def my_score(t,p):
    R = np.array([[1, -1, -.5], [-2, 1, 0], [-1, 0, 1]])
    assert len(t) == len(p)
    s = 0
    for i in range(len(t)):
        s += R[int(t[i]), int(p[i])]
    return s/len(t)
def get_X_y(p):
    X = np.zeros([len(p.keys()),3])
    y = np.zeros([len(p.keys()),1])
    idx = 0
    for i,j in p.items():
        X[idx] = cnt_012(j)
        y[idx] = trans_y(all_ids.iloc[i,1])
        idx += 1
    return X,y

def revise(x, length, mode="same"):
    l = len(x)
    if sum(x) / l <= 0.01:
        x[::] = 0
    elif sum(x) / l >= 0.99:
        x[::] = 1
    else:
        x[:75] = np.argmax(x[:75])
        x[-75:] = np.argmax(x[-75:])
        x = np.convolve(x, np.ones((length,)) / length, mode=mode)
        x = list(map(lambda x: x >= 0.5 and 1 or 0, x))
    return x[:l]

def challenge_entry(sample_path, revise_window, pred_1, pred_2):
    """
    This is our method.
    """
    num = all_ids[all_ids['sample'] == sample_path].index.values[0]
    if pred_1[num] == 0:
        return {'predict_endpoints': []}
    elif pred_1[num] == 1:
        return {'predict_endpoints': [[0, len(pred_2[num]) - 1]]}

    predict = pred_2[num]  # head 2 output
    if revise_window != 0:
        predict = revise(predict, revise_window)
    length = len(predict)
    if all_ids.iloc[num, 0] != sample_path:
        print("check!")
        raise
    sta = list()
    end = list()
    for i in range(length):
        if i == 0:
            if predict[i] and predict[i + 1]:
                sta.append(i)
        elif i == length - 1:
            if predict[i - 1] and predict[i]:
                end.append(i)
        else:
            if not predict[i - 1] and predict[i] and predict[i + 1]:
                sta.append(i)
            elif predict[i - 1] and predict[i] and not predict[i + 1]:
                end.append(i)
    assert len(sta) == len(end), 'unequal lengths of beginnings and ends'
    final_end_points = [[sta[idx], end[idx]] for idx in range(len(sta))]

    pred_dcit = {'predict_endpoints': final_end_points}

    return pred_dcit

def score_calculate_test(test_seq_preds,test_char_preds,clf_model):

    X_test, y_test = get_X_y(test_seq_preds)
    clf = clf_model

    idx = 0
    test_pred_revised = {}
    test_mlp = clf.predict(X_test)
    for i, j in test_seq_preds.items():
        test_pred_revised[i] = test_mlp[idx]
        idx += 1

    RESULT_PATH =  path_name + '/' + 'res'
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    else:
        shutil.rmtree(RESULT_PATH)
        os.makedirs(RESULT_PATH)
    for ind in test_pred_revised.keys():
        ind = int(ind)
        sample_path = all_ids.iloc[:, [0]].iloc[ind, :].values[0]
        pred_dict = challenge_entry(sample_path, 1200, test_pred_revised, test_char_preds)
        sample = sample_path.split("/")[1]
        save_dict(os.path.join(RESULT_PATH, sample + '.json'), pred_dict)
    TESTSET_PATH = data_dir +'/Training_set/Training_set_2'
    RESULT_PATH = path_name + '/' + 'res'
    score_avg = score(TESTSET_PATH, RESULT_PATH)
    test_scores = [round(np.mean(score_avg[0]), 4), round(sum(score_avg[1]) / len(score_avg[1]), 4),
              round(sum(score_avg[2]) / len(score_avg[2]), 4)]

    return test_scores

def test_model(model, test_loader):

    model.train(False)
    preds = {}
    mid_results = {}
    all_preds = []
    all_labels = []
    all_seq_preds = []
    seq_preds_list = {}
    total_correct = 0
    total_correct_seq = 0
    total_seq = 0
    total_samples = 0
    with torch.no_grad():
        for (idx, batch) in enumerate(test_loader):

            ids, ords, inputs, targets, seq_targets, lengths = batch

            inputs = [input.to(device) for input in inputs]
            labels = [target.to(device) for target in targets]
            seq_labels = seq_targets.to(device)
            lengths = lengths.to(device)
            char_outputs, seq_outputs, seq_length, hidden, _, _ = model(inputs, lengths)

            seq_length = seq_length.int().view(-1).cpu().numpy().tolist()

            char_preds = []
            for i in range(char_outputs.size(0)):
                char_preds.append(char_outputs[i, :min(seq_length[i], char_outputs.size(1)), :])
                preds[ids[i]] = preds.get(ids[i],[])+[(ords[i],torch.max(char_outputs[i, :min(seq_length[i], char_outputs.size(1)), :],1)[1])]
                seq_preds_list[ids[i]] = seq_preds_list.get(ids[i],[])+[(ords[i],torch.max(seq_outputs[i], 0)[1])]
            char_preds = torch.concat(char_preds, dim=0)



            labels = torch.concat(labels, dim=0)
            all_labels.append(labels)


            _, predictions = torch.max(char_preds, 1)
            all_preds.append(predictions)

            total_correct += torch.sum(predictions == labels)
            total_samples += (predictions.size(0))

            _, seq_predictions = torch.max(seq_outputs,1)
            all_seq_preds.append(seq_predictions)

    epoch_acc = total_correct.double() / total_samples
    all_preds = torch.concat(all_preds,dim=0)
    all_seq_preds = torch.concat(all_seq_preds,dim=0)
    all_labels = torch.concat(all_labels,dim=0)
    return epoch_acc.item(), all_preds, preds,seq_preds_list,all_seq_preds,all_labels

def collate_fn(batch):
    ids = [item['id'] for item in batch]
    ords = [item['ord'] for item in batch]
    data = [torch.FloatTensor(item['dataMat']) for item in batch]
    label = [torch.LongTensor(item['labelMat']) for item in batch]
    seqlabel = torch.LongTensor([item['seqLabelMat'] for item in batch])
    length = torch.FloatTensor([item['lenMat'] for item in batch])
    return ids, ords, data, label, seqlabel, length

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    split = 'test'
    save_dir = path_name

    print('---load data---')
    proc_ecg_sen = np.load(data_dir + "proc_ecg_sentence_1500.npy", allow_pickle=True).item()

    train_ecg_sen = {}
    valid_ecg_sen = {}
    test_ecg_sen = {}
    for i, j in proc_ecg_sen.items():
        # if j[6] == 0:
        #     train_ecg_sen[i] = j
        # elif j[6] == 1:
        #     valid_ecg_sen[i] = j
        if j[6] == 2:
            test_ecg_sen[i] = j # test set

    seq_index_test = list(test_ecg_sen.keys())


    def min_max_norm(s):
        if np.max(s, axis=0) == np.min(s, axis=0):
            return int(0)
        return (s - np.min(s, axis=0)) / (np.max(s, axis=0) - np.min(s, axis=0))


    test_data_Mat = {}
    test_label_Mat = {}
    test_length_Mat = {}
    test_seq_lable_Mat = {}
    test_ord_Mat = {}
    test_num_Mat = {}
    seq_index_test_sub = []
    for num in seq_index_test:

        if isinstance(min_max_norm(test_ecg_sen[num][0]), int):
            continue
        seq_index_test_sub.append(num)
        test_num_Mat[num] = test_ecg_sen[num][3]
        test_data_Mat[num] = min_max_norm(test_ecg_sen[num][0])
        test_label_Mat[num] = test_ecg_sen[num][1]
        test_length_Mat[num] = test_ecg_sen[num][2]
        test_ord_Mat[num] = test_ecg_sen[num][6]
        if test_ecg_sen[num][4] == 0 or test_ecg_sen[num][4] == 1:
            test_seq_lable_Mat[num] = test_ecg_sen[num][4]
        else:
            test_seq_lable_Mat[num] = 2
    test_dataset = data.MMECG(dataMat=test_data_Mat, labelMat=test_label_Mat, LenMat=test_length_Mat,
                               SeqLabelMat=test_seq_lable_Mat, OrdMat=test_ord_Mat,
                              numMat=test_num_Mat, index_list=seq_index_test_sub)

    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=40, collate_fn=collate_fn)
    model_dir = '/model/best_model_MMECG.pt'
    model = torch.load(model_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_acc, all_preds, preds, seq_preds, all_seq_preds,all_labels = test_model(model,
                                                                      test_loader)

    model_mlp = joblib.load(path_name + '/model/' + 'mlp.model')

    final_preds = {}
    for key, val in preds.items():
        val.sort(key=lambda x: x[0])
        preds_i = []
        for i, p in val:
            preds_i.extend(p.int().cpu().numpy().tolist())
        final_preds[key] = np.array(preds_i)

    final_seq_preds = {}
    for key, val in seq_preds.items():
        val.sort(key=lambda x: x[0])
        preds_i = []
        for i, p in val:
            preds_i.append(p.int().cpu().numpy())
        final_seq_preds[key] = np.array(preds_i)

    np.save(save_dir + 'final_test_preds.npy', final_preds)
    np.save(save_dir + 'final_cls_preds_test.npy', final_seq_preds)

    test_scores = score_calculate_test(final_seq_preds, final_preds, model_mlp)
    with open(path_name + '/log/' + log_name, 'a') as file_log:
        print('test scores:{:.4f} {:.4f} {:.4f}'.format(test_scores[0], test_scores[1], test_scores[2]), file=file_log)

    print('---- beat predictions ---- ')
    test_label = all_labels.cpu().numpy()
    all_preds = all_preds.cpu().numpy()
    with open(path_name + '/log/' + log_name, 'a') as file_log:
        print(metrics.accuracy_score(test_label, all_preds), file=file_log)
        print(cm(test_label, all_preds), file=file_log)
        print(metrics.classification_report(test_label, all_preds), file=file_log)
