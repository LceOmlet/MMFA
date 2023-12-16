import torch

import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger, set_requires_grad
from dataloader.dataloader import data_generator, ad_entities
from trainer.trainer import Trainer, model_evaluate
from models.TC import TC
from utils import _calc_metrics, copy_Files
from models.model import base_Model
from models.patchTST import PatchTSTEncoder
from models.csl_pad import LearningShapeletsModelMixDistances, LearningShapeletsModel
from models.rescnn import ResCNN
# Args selections
start_time = datetime.now()


parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--training_mode', default='supervised', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear')
parser.add_argument('--selected_dataset', default='Default', type=str)
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--data_path', default='cuda', type=str)
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
parser.add_argument('--encoder', type=str, default="transformer")
parser.add_argument('--ckpt_epoch', type=str, default="180")
parser.add_argument('--view_list', type=str, default="")
parser.add_argument('--valid_list', type=str, default="")
args = parser.parse_args()



device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'TS-TCC'
training_mode = args.training_mode
run_description = args.run_description

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)


exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0


# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# Load datasets
data_path = args.data_path

# print(train_dl)
# exit()
logger.debug("Data loaded ...")

# Load Model

if 'ad' not in training_mode:
    if args.encoder == "transformer":
        train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode)
        if args.encoder == "transformer":
            model = LearningShapeletsModelMixDistances(num_classes=configs.num_classes).to(device)
            # model = LearningShapeletsModelMixDistances(num_classes=configs.num_classes, in_channels=configs.input_channels, len_ts=configs.input_length).to(device)
            logger.debug("encode: " + str(model))
        elif args.encoder == "cnn":
            model = base_Model(configs).to(device)
        temporal_contr_model = TC(configs, device).to(device)
        # model = LearningShapeletsModelMixDistances(num_classes=configs.num_classes, in_channels=configs.input_channels).to(device)
        model = LearningShapeletsModelMixDistances(num_classes=configs.num_classes, in_channels=configs.input_channels, len_ts=configs.input_length).to(device)
        logger.debug("encode: " + str(model))
    elif args.encoder == "cnn":
        model = base_Model(configs).to(device)
    temporal_contr_model = TC(configs, device).to(device)



def count_parameters(model, loaded_state_dict):
    total_params = 0
    loaded_params = 0
    unloaded_params = 0
    unexpected_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()

        if name in loaded_state_dict:
            loaded_params += param.numel()
        else:
            unloaded_params += param.numel()
            logger.info(f"Unloaded parameter: {name}")
    model_state_dict =  model.state_dict()
    for name in loaded_state_dict.keys():
        if name not in model_state_dict:
            unexpected_params += loaded_state_dict[name].numel()
            logger.info(f"Unexpected parameter: {name}")

    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Loaded parameters: {loaded_params}")
    logger.info(f"Unloaded parameters: {unloaded_params}")
    logger.info(f"Unexpected parameters: {unexpected_params}")

if "fine_tune" in training_mode :
    # load saved model of this experiment
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, f"ckp_{args.ckpt_epoch}.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

if training_mode == "train_linear" or "tl" in training_mode:
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"specreg_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, f"ckp_{args.ckpt_epoch}.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()
    count_parameters(model, pretrained_dict)
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # delete these parameters (Ex: the linear layer at the end)
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    set_requires_grad(model, pretrained_dict, requires_grad=False)  # Freeze everything except last layer.

if training_mode == "random_init":
    model_dict = model.state_dict()

    # delete all the parameters except for logits
    del_list = ['logits']
    pretrained_dict_copy = model_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del model_dict[i]
    set_requires_grad(model, model_dict, requires_grad=False)  # Freeze everything except last layer.



if 'ad' not in training_mode:
    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
    temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

if "self_supervised" in training_mode :  # to do it only once
    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)

if 'ad'not in training_mode:
    train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode)
    train_dls = {data_path: train_dl}
    if args.valid_list:
        valid_list = args.valid_list.split("-")
    else:
        valid_list = []
    
    for v in valid_list:
        train_dl_v, valid_dl_v, test_dl_v = data_generator(v, configs, training_mode)
        valid_dl.update(valid_dl_v)
        train_dls[v] = train_dl_v
    # Trainer
    Trainer(model, temporal_contr_model, model_optimizer, 
        temporal_contr_optimizer, train_dl, train_dls, valid_dl, test_dl,
        device, logger, configs, experiment_log_dir, training_mode, view_list_s=args.view_list)
else:
    for w in [100]:
        for ch in ad_entities[data_path]:
            # print(ch)
            # exit()
            data_path = data_path + "-" + ch
            train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode, w)
            if args.encoder == "transformer":
                # model = LearningShapeletsModelMixDistances(num_classes=configs.num_classes, in_channels=configs.input_channels).to(device)
                model = LearningShapeletsModelMixDistances(num_classes=configs.num_classes, in_channels=configs.input_channels, len_ts=configs.input_length).to(device)
                logger.debug("encode: " + str(model))
            elif args.encoder == "cnn":
                model = base_Model(configs).to(device)
            temporal_contr_model = TC(configs, device).to(device)
            model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
            temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
            # Trainer
            
            Trainer(model, temporal_contr_model, model_optimizer, 
                    temporal_contr_optimizer, train_dl, valid_dl, test_dl,
                    device, logger, configs, experiment_log_dir, training_mode)


if "self_supervised" not in training_mode :
    # Testing
    outs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
    total_loss, total_acc, pred_labels, true_labels = outs
    _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)


logger.debug(f"Training time is : {datetime.now()-start_time}")
