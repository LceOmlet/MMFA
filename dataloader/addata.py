import os

import numpy as np
import torch
from torch import nn, optim
import random


from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, rand_score, normalized_mutual_info_score

import tsaug

from torch.utils.tensorboard import SummaryWriter

import argparse


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.ensemble import IsolationForest


import numpy as np
import pandas as pd
import joblib
import os
import torch


import torch.distributed as dist
from torch.multiprocessing import Process 
from torch.nn.parallel import DistributedDataParallel as DDP

DATA_PATH = './AD_data'










def evaluate_ad(dataset, window_size=100):
        
    if dataset == 'SMAP' or dataset == 'MSL':
        all_labels = pd.read_csv(os.path.join(DATA_PATH, 'SMAP&MLS', 'labeled_anomalies.csv'))



        if dataset == 'SMAP':
            data_labels = all_labels[all_labels['spacecraft'] == 'SMAP']
        if dataset == 'MSL':
            data_labels = all_labels[all_labels['spacecraft'] == 'MSL']
    
        all_chans = data_labels['chan_id'].values

        print(all_chans)
    
    elif dataset == 'SMD' or dataset == 'ASD':
        if dataset == 'SMD':
            all_chans = ['machine-1-1', 'machine-1-6', 'machine-1-7',
                        'machine-2-1', 'machine-2-2', 'machine-2-7', 'machine-2-8',
                        'machine-3-3', 'machine-3-4', 'machine-3-6', 'machine-3-8', 'machine-3-11']
        if dataset == 'ASD':
            all_chans = ['omi-' + str(i) for i in range(1, 13)]    
    #window_size = args.window_size
    
    progress_bar = tqdm(range(len(all_chans)))
    
    exit()

    #print(all_chans.shape)
    for ch_idx in progress_bar:
        channel = all_chans[ch_idx]
        if dataset == 'SMAP' or dataset == 'MSL':
            train = np.load(os.path.join(DATA_PATH, 'SMAP&MLS', 'train', channel + '.npy'))
            test = np.load(os.path.join(DATA_PATH, 'SMAP&MLS', 'test', channel + '.npy'))
            
            
            label = np.load(os.path.join(DATA_PATH, 'SMAP&MLS', 'labels', channel + '.npy'))[window_size - 1:]
            
        elif dataset == 'SMD' or dataset == 'ASD':
            train = joblib.load(os.path.join(DATA_PATH, 'SMD&ASD/processed', channel + '_train.pkl'))
            test = joblib.load(os.path.join(DATA_PATH, 'SMD&ASD/processed', channel + '_test.pkl'))
            
            raw_label = joblib.load(os.path.join(DATA_PATH, 'SMD&ASD/processed', channel + '_test_label.pkl'))[window_size - 1:]

            label = np.ones_like(raw_label, dtype=np.int32)
            label[raw_label == 1] = -1
        
        train = np.nan_to_num(train, 0)
        test = np.nan_to_num(test, 0)

        scaler = MinMaxScaler((-1, 1)).fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)
        
        train_data = torch.from_numpy(train).unfold(0, window_size, 1).numpy()
        test_data = torch.from_numpy(test).unfold(0, window_size, 1).numpy()
         

evaluate_ad("MSL", 100)