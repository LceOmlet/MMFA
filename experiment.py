#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from ts_url.training_methods import Trainer
import time
import os
# torch.cuda.set_device(4)
import random
import numpy as np
from torch import nn
import json
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
import argparse
torch.autograd.set_detect_anomaly(True)

datasets = [
    # "ArticularyWordRecognition",
    # "AtrialFibrillation",
    # "BasicMotions",
    # "CharacterTrajectories",
    # "Cricket",
    # "DuckDuckGeese",
    # "EigenWorms",
    # "Epilepsy",
    # "EthanolConcentration",
    # "ERing",
    # "FaceDetection",
    # "FingerMovements",
    # "HandMovementDirection",
    "Handwriting",
    # "Heartbeat",
    # "InsectWingbeat",
    # "JapaneseVowels",
    # "Libras",
    # "LSST",
    # "MotorImagery",
    "NATOPS",
    "PenDigits",
    "PEMS-SF",
    "Phoneme",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "SpokenArabicDigits",
    "StandWalkJump",
    "UWaveGestureLibrary"
]

# datasets = ["BasicMotions"]

# Try csl on LSST dataset with SVM as the test module.

# In[2]:

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default="mvts_transformer")
parser.add_argument("--gpu", type=int, default=5)
parser.add_argument("--dataset_name", type=str, default="")
parser.add_argument("--experiment_type", type=str, default="PT")
# parser.add_argument("")
args = parser.parse_args()

experiment = "exp1"

hp_path = "configs/csl_optim.json"
p_path = "configs/csl.json"
optim_config = "configs/csl_optim.json"
task_name = "pretraining"

device = torch.device('cuda')


def get_config(filepath="/home/username/username/aeon/aeon/datasets/data/test/Multivariate_ts", 
               train_ratio=1, test_ratio=1, dsid="HandMovementDirection"):
    filepath += '/' + dsid
    data_configs = [{
        "filepath": filepath,
        "train_ratio": train_ratio,
        "test_ratio": test_ratio,
        "dsid": dsid
    }]
    return data_configs

data_configs = get_config()

data_names = [d['dsid'] for d in data_configs]
task_summary = "_".join(data_names) + "_" + args.model_name
start_time = time.strftime("%m_%d_%H_%M_%S", time.localtime()) 
save_path = os.path.join(experiment, task_summary, start_time)

os.makedirs(save_path, exist_ok=True)


# In[3]:


experiment = "exp1/ucr" + args.experiment_type

# hp_path = "configs/ts_tcc_optim.json"
p_path = f"default_config/{args.model_name}.json"
optim_config = f"default_config/{args.model_name}_optim.json"


if args.experiment_type == "FC":
    with open(optim_config, mode="r") as f:
        optim_config = json.load(f)
        print(f"Current sfa wwm weight loading type: {optim_config['transformations']['sfa']['model']['load_wwm_weights']}")
        optim_config["transformations"]["sfa"]["model"]["load_wwm_weights"] = False
    
task_name = "pretraining"
model_name = args.model_name

device = int(args.gpu)
if args.dataset_name != "":
    datasets = [args.dataset_name]
    
if args.dataset_name == "ucr":
    for i in range(0, 250):
        dsid = "ad_ucr_" + str(i)
        data_configs = get_config(dsid=dsid)
        data_names = [d['dsid'] for d in data_configs]
        task_summary = "_".join(data_names) + "_" + model_name
        start_time = time.strftime("%m_%d_%H_%M_%S", time.localtime()) 
        save_path = os.path.join(experiment, task_summary, start_time)
        os.makedirs(save_path, exist_ok=True)
        trainer = Trainer(data_configs, model_name, p_path, 
                        device, optim_config, task_name, save_path=save_path)

        # trainer.validate(epoch_num=0, key_metric="loss", save_dir=save_path)
        trainer.fit()
else:
    dsid = args.dataset_name
    data_configs = get_config(dsid=dsid)
    data_names = [d['dsid'] for d in data_configs]
    task_summary = "_".join(data_names) + "_" + model_name
    start_time = time.strftime("%m_%d_%H_%M_%S", time.localtime()) 
    save_path = os.path.join(experiment, task_summary, start_time)
    os.makedirs(save_path, exist_ok=True)
    trainer = Trainer(data_configs, model_name, p_path, 
                    device, optim_config, task_name, save_path=save_path)

    # trainer.validate(epoch_num=0, key_metric="loss", save_dir=save_path)
    trainer.fit()


