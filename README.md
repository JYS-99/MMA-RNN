# MMA-RNN: A Multi-level Multi-task Attention-based Recurrent Neural Network for Discrimination and Localization of Atrial Fibrillation

This repo is the official implementations of [MMA-RNN](https://www.sciencedirect.com/science/article/abs/pii/S1746809423011801). 

### Data 
1. Download the original datasets from this [link](https://drive.google.com/drive/folders/1Rm7Ba5HAHDxPeKHOR9wK5TSG2gyb01L9?usp=sharing, https://drive.google.com/drive/folders/1hpoijWP5EsKyarubKY00d4rT0EDQJLou?usp=sharing).
2. Run data_processing.py file to segment the series and get corresponding input (proc_ecg_sentence_1500.npy).

### Train
To train the MMA-RNN model, please use

python MMA-RNN/main_MMECG.py

### Baseline

Baseline codes are available at http://2021.icbeb.org/CPSC2021. More details can be found in our article.

### Test

To test the model, please use

python MMA-RNN/test.py
