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
from torch.utils.data import DataLoader
import wfdb
from utils import qrs_detect,comp_cosEn, save_dict
import json
import shutil
import joblib
from sklearn.neural_network import MLPClassifier
import scipy.io as sio
from copy import deepcopy


log_name = 'log.txt'
path_name = './model'
data_dir = './data'

R = np.array([[1, -1, -.5], [-2, 1, 0], [-1, 0, 1]])

class RefInfo():
    def __init__(self, sample_path):
        self.sample_path = sample_path
        self.fs, self.len_sig, self.beat_loc, self.af_starts, self.af_ends, self.class_true = self._load_ref()
        self.endpoints_true = np.dstack((self.af_starts, self.af_ends))[0, :, :]
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

    predict = deepcopy(pred_2[num])  # head 2 output
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

def score_calculate(seq_preds,char_preds,valid_seq_preds,valid_char_preds,epoch):

    X_train, y_train = get_X_y(seq_preds)
    X_valid, y_valid = get_X_y(valid_seq_preds)
    clf = MLPClassifier(hidden_layer_sizes=(100), random_state=0, max_iter=10000).fit(X_train, y_train)
    print(my_score(y_train, clf.predict(X_train)))
    joblib.dump(clf, './model/mlp_model'+'mlp'+str(epoch)+'.model')

    idx = 0
    valid_pred_revised = {}
    valid_mlp = clf.predict(X_valid)
    for i, j in valid_seq_preds.items():
        valid_pred_revised[i] = valid_mlp[idx]
        idx += 1

    RESULT_PATH = './' + 'res'
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    else:
        shutil.rmtree(RESULT_PATH)
        os.makedirs(RESULT_PATH)
    for ind in valid_pred_revised.keys():
        ind = int(ind)
        sample_path = all_ids.iloc[:, [0]].iloc[ind, :].values[0]
        pred_dict = challenge_entry(sample_path, 1200, valid_pred_revised, valid_char_preds)
        sample = sample_path.split("/")[1]
        save_dict(os.path.join(RESULT_PATH, sample + '.json'), pred_dict)
    TESTSET_PATH = './data/Training_set/Training_set_2'
    RESULT_PATH = './' + 'res'
    score_avg = score(TESTSET_PATH, RESULT_PATH)
    # 总分，第一问，第二问
    valid_scores = [round(np.mean(score_avg[0]), 4), round(sum(score_avg[1]) / len(score_avg[1]), 4),
              round(sum(score_avg[2]) / len(score_avg[2]), 4)]

    return valid_scores

def score_calculate_test(test_seq_preds,test_char_preds,clf_model):

    X_test, y_test = get_X_y(test_seq_preds)
    clf = clf_model

    idx = 0
    test_pred_revised = {}
    test_mlp = clf.predict(X_test)
    for i, j in test_seq_preds.items():
        test_pred_revised[i] = test_mlp[idx]
        idx += 1

    RESULT_PATH = './' + 'res'
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
    TESTSET_PATH = './data/Training_set/Training_set_2'
    RESULT_PATH = './' + 'res'
    score_avg = score(TESTSET_PATH, RESULT_PATH)
    # 总分，第一问，第二问
    test_scores = [round(np.mean(score_avg[0]), 4), round(sum(score_avg[1]) / len(score_avg[1]), 4),
              round(sum(score_avg[2]) / len(score_avg[2]), 4)]

    return test_scores

def test_model(model, test_loader):

    model.train(False)
    preds = {}
    mid_results = {}
    all_preds = []
    all_seq_preds = []
    seq_preds_list = {}
    total_correct = 0
    total_correct_seq = 0
    total_seq = 0
    total_samples = 0
    with torch.no_grad():
        for (idx, batch) in enumerate(test_loader):

            ids, ords, inputs, targets, seq_mfcc, sen_mfcc, seq_targets, lengths = batch  # seq_targets,  seq_features,

            inputs = [input.to(device) for input in inputs]
            labels = [target.to(device) for target in targets]
            seq_mfcc = seq_mfcc.to(device)
            sen_mfcc = sen_mfcc.to(device)
            seq_labels = seq_targets.to(device)
            lengths = lengths.to(device)
            char_outputs, seq_outputs, seq_length, hidden, _, _ = model(inputs, lengths, seq_mfcc, sen_mfcc)  # mask,

            seq_length = seq_length.int().view(-1).cpu().numpy().tolist()

            char_preds = []
            for i in range(char_outputs.size(0)):
                char_preds.append(char_outputs[i, :min(seq_length[i], char_outputs.size(1)), :])
                preds[ids[i]] = preds.get(ids[i],[])+[(ords[i],torch.max(char_outputs[i, :min(seq_length[i], char_outputs.size(1)), :],1)[1])]
                seq_preds_list[ids[i]] = seq_preds_list.get(ids[i],[])+[(ords[i],torch.max(seq_outputs[i], 0)[1])]
            char_preds = torch.concat(char_preds, dim=0)

            labels = torch.concat(labels, dim=0)


            _, predictions = torch.max(char_preds, 1)
            all_preds.append(predictions)

            total_correct += torch.sum(predictions == labels)
            total_samples += (predictions.size(0))

            _, seq_predictions = torch.max(seq_outputs,1)
            all_seq_preds.append(seq_predictions)

    epoch_acc = total_correct.double() / total_samples
    all_preds = torch.concat(all_preds,dim=0)
    all_seq_preds = torch.concat(all_seq_preds,dim=0)
    return epoch_acc.item(), all_preds, preds,seq_preds_list,all_seq_preds

def train_model(model, train_loader, valid_loader, criterion, seq_criterion, optimizer, scheduler, save_dir, valid_trues, num_epochs=20):
    def train(model, train_loader, optimizer, scheduler, criterion, seq_criterion):
        model.train(True)
        total_loss = 0.0
        total_correct = 0
        total_correct_seq = 0
        total_seq = 0
        total_samples = 0
        losses = []
        total_beat_loss= 0.0
        total_seq_loss = 0.0

        preds = {}
        seq_preds_list = {}

        cnt_batch = 0
        for (idx,batch) in enumerate(train_loader):

            ids, ords, inputs, targets, seq_mfcc, sen_mfcc, seq_targets, lengths = batch  # seq_targets,  seq_features,
            cnt_batch += 1
            inputs = [input.to(device) for input in inputs]
            labels = [target.to(device) for target in targets]
            seq_mfcc = seq_mfcc.to(device)
            sen_mfcc = sen_mfcc.to(device)
            seq_labels = seq_targets.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad()
            char_outputs, seq_outputs, seq_length,  hidden, _ , _ = model(inputs, lengths, seq_mfcc, sen_mfcc) #mask,
            # beat_outputs, seq_length = model(inputs, others, seq_features,) #seq_features,

            seq_length = seq_length.int().view(-1).cpu().numpy().tolist()

            char_preds = []
            for i in range(char_outputs.size(0)):
                char_preds.append(char_outputs[i, :min(seq_length[i], char_outputs.size(1)), :])
                preds[ids[i]] = preds.get(ids[i], []) + [
                    (ords[i], torch.max(char_outputs[i, :min(seq_length[i], char_outputs.size(1)), :], 1)[1])]
                seq_preds_list[ids[i]] = seq_preds_list.get(ids[i], []) + [(ords[i], torch.max(seq_outputs[i], 0)[1])]
            char_preds = torch.concat(char_preds, dim=0)


            labels = torch.concat(labels, dim=0)

            char_loss = criterion(char_preds, labels)
            seq_loss = seq_criterion(seq_outputs, seq_labels.view(-1))
            _, predictions = torch.max(char_preds, 1)

            loss = 40/1500 * char_loss + seq_loss

            loss.backward()

            nn.utils.clip_grad_norm(model.parameters(), 0.1)
            optimizer.step()
            scheduler.step()

            losses.append(loss.item()*len(inputs))
            total_loss += loss.item() * len(inputs)
            total_beat_loss += char_loss.item() * len(inputs)
            total_seq_loss += seq_loss.item() * len(inputs)
            total_correct += torch.sum(predictions == labels)
            total_samples += (predictions.size(0))

            _, seq_preds = torch.max(seq_outputs, 1)
            total_correct_seq += torch.sum(seq_preds == seq_labels.view(-1))
            total_seq += len(inputs)
            if cnt_batch % 100 == 0:
                print('%d Batch, Acc %9.4f' % (cnt_batch, total_correct.double()/total_samples))

        epoch_loss = total_loss / total_samples
        print('Epoch, Beat Loss %9.4f Seq Loss %9.4f' % (total_beat_loss / total_samples, total_seq_loss/total_samples))
        epoch_acc = total_correct.double() / total_samples
        return epoch_loss, epoch_acc.item(),losses, preds, seq_preds_list

    def valid(model, valid_loader, criterion, seq_criterion):
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_correct_seq = 0
        total_seq = 0

        preds = {}
        seq_preds_list = {}


        for (idx, batch) in enumerate(valid_loader):

            ids, ords, inputs, targets, seq_targets, lengths = batch  # seq_targets,  seq_features,
            inputs = [input.to(device) for input in inputs]
            labels = [target.to(device) for target in targets]
            seq_labels = seq_targets.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad()
            char_outputs, seq_outputs, seq_length, hidden, char_alpha, word_alpha = model(inputs, lengths) #mask,

            seq_length = seq_length.int().view(-1).cpu().numpy().tolist()


            char_preds = []
            for i in range(char_outputs.size(0)):
                char_preds.append(char_outputs[i, :min(seq_length[i], char_outputs.size(1)), :])
                preds[ids[i]] = preds.get(ids[i], []) + [
                    (ords[i], torch.max(char_outputs[i, :min(seq_length[i], char_outputs.size(1)), :], 1)[1])]
                seq_preds_list[ids[i]] = seq_preds_list.get(ids[i], []) + [(ords[i], torch.max(seq_outputs[i], 0)[1])]
            char_preds = torch.concat(char_preds, dim=0)

            labels = torch.concat(labels, dim=0)

            char_loss = criterion(char_preds, labels)  # labels.view(-1),, alpha = 0.75, reduction='mean'
            seq_loss = seq_criterion(seq_outputs, seq_labels.view(-1))
            _, predictions = torch.max(char_preds, 1)

            loss = 40/1500 * char_loss +  seq_loss

            total_loss += loss.item() * len(inputs)
            total_correct += torch.sum(predictions == labels)
            total_samples += (predictions.size(0))

            _, seq_preds = torch.max(seq_outputs, 1)
            total_correct_seq += torch.sum(seq_preds == seq_labels.view(-1))
            total_seq += len(inputs)


        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct.double() / total_samples
        epoch_seq_acc = total_correct_seq.double()/total_seq

        return epoch_loss, epoch_acc.item() ,epoch_seq_acc.item(),  preds,seq_preds_list

    best_score = 0.0
    dir_log = save_dir + '/log/' + log_name

    train_losses, valid_losses = [],[]
    for epoch in tqdm(range(1, num_epochs+1)):
        infos = f'epoch:{epoch:d}/{num_epochs:d}' +\
                    f'lr: {scheduler.get_lr()[0]:.2e} '
        with open('./log/'+log_name, 'a') as file_log:
            print(infos,file = file_log)
            print('*' * 100,file = file_log)
            train_loss, train_acc, train_loss_l, train_preds, train_seq_preds = train(model, train_loader, optimizer, scheduler, criterion, seq_criterion)
            print("training: {:.5f}, {:.5f}".format(train_loss, train_acc),file = file_log)
            valid_loss, valid_acc, valid_seq_acc, valid_preds, valid_seq_preds = valid(model, valid_loader, criterion, seq_criterion)
            print("validation: {:.4f}, {:.4f} {:.4f}".format(valid_loss, valid_acc, valid_seq_acc),file = file_log)

            final_train_preds = {}
            for key, val in train_preds.items():
                val.sort(key=lambda x: x[0])
                preds_i = []
                for i, p in val:
                    preds_i.extend(p.int().cpu().numpy().tolist())
                final_train_preds[key] = np.array(preds_i)

            final_train_seq_preds = {}
            for key, val in train_seq_preds.items():
                val.sort(key=lambda x: x[0])
                preds_i = []
                for i, p in val:
                    preds_i.append(p.int().cpu().numpy())
                final_train_seq_preds[key] = np.array(preds_i)

            final_valid_preds = {}
            for key, val in valid_preds.items():
                val.sort(key=lambda x: x[0])
                preds_i = []
                for i, p in val:
                    preds_i.extend(p.int().cpu().numpy().tolist())
                final_valid_preds[key] = np.array(preds_i)

            final_valid_seq_preds = {}
            for key, val in valid_seq_preds.items():
                val.sort(key=lambda x: x[0])
                preds_i = []
                for i, p in val:
                    preds_i.append(p.int().cpu().numpy())
                final_valid_seq_preds[key] = np.array(preds_i)

            valid_scores = score_calculate(final_train_seq_preds,final_train_preds,final_valid_seq_preds,final_valid_preds,epoch)
            print("validation scores: {:.4f}, {:.4f} {:.4f}".format(valid_scores[0],valid_scores[1],valid_scores[2]), file=file_log)

            if valid_scores[0] > best_score:
                best_score = valid_scores[0]
                best_model = model
                torch.save(best_model, os.path.join(save_dir, 'best_model_MMECG.pt'))
                np.save(save_dir + '/final_cls_valid.npy', final_valid_seq_preds)
                np.save(save_dir + '/final_preds_valid.npy', final_valid_preds)


            train_losses.extend(train_loss_l)
            valid_losses.append(valid_loss)
            print("best validation score:{:.4f}".format(best_score), file = file_log)

    return train_losses, valid_losses


def collate_fn(batch):
    ids = [item['id'] for item in batch]
    ords = [item['ord'] for item in batch]
    data = [torch.FloatTensor(item['dataMat']) for item in batch]
    label = [torch.LongTensor(item['labelMat']) for item in batch]
    seqlabel = torch.LongTensor([item['seqLabelMat'] for item in batch])
    length = torch.FloatTensor([item['lenMat'] for item in batch])
    return ids, ords, data, label, seqlabel, length

if __name__ == '__main__':

    os.chdir(r"./")
    data_dir = './data'
    save_dir = '.' + path_name
    if not os.path.exists(save_dir + '/log'):
        os.mkdir(save_dir + '/log')
    if not os.path.exists(save_dir + '/mlp_model'):
        os.mkdir(save_dir + '/mlp_model')
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


    parser = argparse.ArgumentParser(description='ecg')
    parser.add_argument('--save_dir', type=str,  default=save_dir, help='model save path')
    parser.add_argument('--seed', type=int, default=0,
                        help='set random seed')
    args,_ = parser.parse_known_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    print('---load data---')
    proc_ecg_sen = np.load(data_dir + "proc_ecg_sentence_1500.npy", allow_pickle=True).item()

    train_ecg_sen = {}
    valid_ecg_sen = {}
    test_ecg_sen = {}
    for i, j in proc_ecg_sen.items():
        if j[6] == 0:
            train_ecg_sen[i] = j
        elif j[6] == 1:
            valid_ecg_sen[i] = j
        elif j[6] == 2:
            test_ecg_sen[i] = j
        else:
            raise

    seq_index_train = list(train_ecg_sen.keys())
    np.random.shuffle(seq_index_train)

    seq_index_valid = list(valid_ecg_sen.keys())
    np.random.shuffle(seq_index_valid)

    seq_index_test = list(test_ecg_sen.keys())

    def min_max_norm(s):
        if np.max(s,axis = 0) == np.min(s,axis = 0):
            return int(0)
        return (s - np.min(s,axis = 0))/(np.max(s,axis = 0)-np.min(s,axis = 0))

    train_data_Mat = {}
    train_label_Mat = {}
    train_length_Mat = {}
    train_seq_lable_Mat = {}
    train_ord_Mat = {}
    train_num_Mat = {}
    seq_index_train_sub = []
    for num in seq_index_train:

        if train_ecg_sen[num][7] == 1:
            if isinstance(min_max_norm(train_ecg_sen[num][0]), int):
                continue
            seq_index_train_sub.append(num)
            train_num_Mat[num] = train_ecg_sen[num][3]
            train_data_Mat[num] = min_max_norm(train_ecg_sen[num][0])
            train_label_Mat[num] = train_ecg_sen[num][1]
            train_length_Mat[num] = train_ecg_sen[num][2]
            train_ord_Mat[num] = train_ecg_sen[num][8]
            if train_ecg_sen[num][4] == 0 or train_ecg_sen[num][4] == 1:
                train_seq_lable_Mat[num] = train_ecg_sen[num][4]
            else:
                train_seq_lable_Mat[num] = 2

    valid_data_Mat = {}
    valid_label_Mat = {}
    valid_length_Mat = {}
    valid_seq_lable_Mat = {}
    valid_ord_Mat = {}
    valid_num_Mat = {}
    seq_index_valid_sub = []
    for num in seq_index_valid:
        if valid_ecg_sen[num][7] == 1:
            if isinstance(min_max_norm(valid_ecg_sen[num][0]), int):
                continue
            seq_index_valid_sub.append(num)
            valid_num_Mat[num] = valid_ecg_sen[num][3]
            valid_data_Mat[num] = min_max_norm(valid_ecg_sen[num][0])
            valid_label_Mat[num] = valid_ecg_sen[num][1]
            valid_length_Mat[num] = valid_ecg_sen[num][2]
            valid_ord_Mat[num] = valid_ecg_sen[num][8]
            if valid_ecg_sen[num][4] == 0 or valid_ecg_sen[num][4] == 1:
                valid_seq_lable_Mat[num] = valid_ecg_sen[num][4]
            else:
                valid_seq_lable_Mat[num] = 2

    test_data_Mat = {}
    test_label_Mat = {}
    test_length_Mat = {}
    test_seq_lable_Mat = {}
    test_ord_Mat = {}
    test_num_Mat = {}
    seq_index_test_sub = []
    for num in seq_index_test:
        if test_ecg_sen[num][7] == 1:
            if isinstance(min_max_norm(test_ecg_sen[num][0]), int):
                continue
            seq_index_test_sub.append(num)
            test_num_Mat[num] = test_ecg_sen[num][3]
            test_data_Mat[num] = min_max_norm(test_ecg_sen[num][0])
            test_label_Mat[num] = test_ecg_sen[num][1]
            test_length_Mat[num] = test_ecg_sen[num][2]
            test_ord_Mat[num] = test_ecg_sen[num][8]
            if test_ecg_sen[num][4] == 0 or test_ecg_sen[num][4] == 1:
                test_seq_lable_Mat[num] = test_ecg_sen[num][4]
            else:
                test_seq_lable_Mat[num] = 2

    true = np.load(data_dir + "/ecg_sen_test_label.npy", allow_pickle=True).item()
    true_val = np.load(data_dir + "/ecg_sen_val_label.npy",allow_pickle=True).item()

    ## about model
    char_num_classes = 2
    seq_num_classes = 3

    ## about data
    # data_dir = args.data_dir
    batch_size = 30
    # steps = 30

    ## about training
    num_epochs = 40
    lr = 0.001

    model = models.MMECG(num_char_classes=char_num_classes, num_seq_classes=seq_num_classes)
    print(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ## data preparation
    train_dataset = data.MMECG(dataMat=train_data_Mat, labelMat=train_label_Mat, LenMat=train_length_Mat,
                                      SeqLabelMat=train_seq_lable_Mat, OrdMat=train_ord_Mat,
                                     numMat=train_num_Mat, index_list=seq_index_train_sub)
    valid_dataset = data.MMECG(dataMat=valid_data_Mat, labelMat=valid_label_Mat, LenMat=valid_length_Mat,
                                      SeqLabelMat=valid_seq_lable_Mat, OrdMat=valid_ord_Mat,
                                     numMat=valid_num_Mat, index_list=seq_index_valid_sub)
    test_dataset = data.MMECG(dataMat=test_data_Mat, labelMat=test_label_Mat, LenMat=test_length_Mat,
                                    SeqLabelMat=test_seq_lable_Mat, OrdMat=test_ord_Mat,
                                   numMat=test_num_Mat, index_list=seq_index_test_sub)

    train_loader = DataLoader(train_dataset, shuffle = True, batch_size=batch_size, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, collate_fn=collate_fn)

    ## optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 40000, gamma=1)

    ## training
    criterion = nn.CrossEntropyLoss(reduction = 'sum')
    seq_criterion =  nn.CrossEntropyLoss(reduction = 'sum') #CostSensitiveLoss()
    train_losses, valid_losses = train_model(model, train_loader, valid_loader, criterion, seq_criterion, optimizer,scheduler, args.save_dir, true_val, num_epochs=num_epochs)
    np.save(os.path.join(args.save_dir,'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(args.save_dir,'valid_losses.npy'), np.array(valid_losses))

    model_dir = 'best_model_MMECG.pt'
    model = torch.load(model_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_acc, all_preds, preds, seq_preds, all_seq_preds = test_model(model,
                                                                      test_loader)  # seq_preds, mid_results

    model_mlp = joblib.load('mlp.model')

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
    test_scores = score_calculate_test(final_seq_preds, final_preds)
    with open('./log/' + log_name, 'a') as file_log:
        print('test scores:{:.4f} {:.4f} {:.4f}'.format(test_scores[0],test_scores[1],test_scores[2]),file = file_log)

    print('---- beat predictions ---- ')
    test_label = np.concatenate(list(test_label_Mat.values()), axis=0)
    all_preds = all_preds.cpu().numpy()
    with open('./log/' + log_name, 'a') as file_log:
        print(metrics.accuracy_score(test_label, all_preds),file = file_log)
        print(cm(test_label, all_preds),file = file_log)
        print(metrics.classification_report(test_label, all_preds),file = file_log)
