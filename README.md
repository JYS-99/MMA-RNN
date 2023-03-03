# MMA-RNN: A Multi-level Multi-task Attention-based Recurrent Neural Network for Discrimination and Localization of Atrial Fibrillation

This repo is the official implementations of MMA-RNN. 
https://arxiv.org/abs/2302.03731

### Data 
1. Download the original datasets from https://drive.google.com/drive/folders/1h7YlZwV714UtoM8RR47Fj3GlEC2SfHQK?usp=sharing (part 1) and https://drive.google.com/drive/folders/1AQRJC4Ib5fAUyMd4i2AYF570dbI5fioe?usp=sharing (part 2).
2. Run data_processing.py file to segment the series and get corresponding input (proc_ecg_sentence_1500.npy).

### Train
To train the MMA-RNN model, please use

python MMA-RNN/main_MMECG.py

### Baseline

Baseline codes are available at http://2021.icbeb.org/CPSC2021. More details can be found in our article.

### Test

To test the model, please use

python MMA-RNN/test.py
