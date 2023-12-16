import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from models.loss import NTXentLoss
from sklearn.svm import SVC
from itertools import tee, chain
from copy import deepcopy
from torchvision import transforms
from models.ResNet12 import ResNet12
from models.bert import SFA_Bert
from models.csl_pad import LearningShapeletsModelMixDistances
from pyts.image import MarkovTransitionField
from pyts.multivariate.image import JointRecurrencePlot
from pyts.image import GramianAngularField
from models.mnas import MnasNet
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pywt
from dataloader.augmentations import DataTransform
from models.patchTST import PatchTSTEncoder
from models.rescnn import ResCNN
from functools import partial
from random import random
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.ensemble import IsolationForest
import pickle as pkl
from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score, rand_score

model_ = None


def orig_model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config, device, training_mode):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()

    for batch_idx, (data, labels, aug1, aug2, _, _, _, _) in enumerate(train_loader):
        # send to device
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        output = model(data)

        # compute loss

        predictions, features = output
        loss = criterion(predictions, labels)
        total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc

def orig_model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels, _, _, _, _, _, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            if training_mode == "self_supervised":
                pass
            else:
                output = model(data)

            # compute loss 
            if training_mode != "self_supervised":
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

            if training_mode != "self_supervised":
                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if training_mode != "self_supervised":
        total_loss = torch.tensor(total_loss).mean()  # average loss
    else:
        total_loss = 0
    if training_mode == "self_supervised":
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_acc = torch.tensor(total_acc).mean()  # average acc
    return total_loss, total_acc, outs, trgs

def slice_ts(data, slice_):
    data_len = data.shape[-1]
    data = data[...,round(slice_[0]*data_len): round(slice_[1]*data_len)]
    data = torch.nn.functional.interpolate(data, (224, ))
    return data

slice_1, slice_2 = None, None
def slice_time_series(data1, data2, content_rate=2/3, temporal_prob=0.5):
    global slice_1, slice_2
    if random() > temporal_prob:
        slice_1 = (0, content_rate)
        slice_2 = (1 - content_rate, 1)
    else:
        slice_1 = (0, 1)
        slice_2 = (0, 1)
    data1 = slice_ts(data1, slice_1)
    data2 = slice_ts(data2, slice_2)
    return data1, data2


def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, 
            train_dl, train_dls, valid_dl, test_dl, 
             device, logger, config, experiment_log_dir, training_mode, save_interval=10, view_list_s=""):
    # Start training
    logger.debug("Training started ....")
    global model_ 
    model_ = model    



    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    if "spec" in training_mode:
        model_list = []
        view_list_ = []

        if view_list_s == "":
            view_list = ["fft"]
        elif '-' not in view_list_s:
            view_list = [view_list_s]
        else:
            view_list = view_list_s.split('-')
            
        view_r = [1] * len(view_list)

        # view_list = ["rp", "fft", "gadf", "sfa", "db1", "coif5", 'dmey']
        # view_r = [1, 1, 1, 1, 1, 1]
        #CUDA_VISIBLE_DEVICES=5 python main.py --experiment_description exp4 --run_description EthanolConcentration --seed 123 --training_mode vicreg --selected_dataset HAR --data_path EthanolConcentration
        model = nn.DataParallel(model)
        model_list.append(model)
        view_list_.append(same)
        for view in view_list:
            view_list_.append(view_constructor[view][0])
            constructor = view_constructor[view][1]
            if constructor == ResNet12 and config.input_length < 224:
                submodel = constructor(wider=True)
            elif constructor == ResNet12:
                submodel = constructor(wider=False)
            else:
                submodel = constructor()
            model_list.append(nn.DataParallel(submodel.to(device)))

        for model_ in model_list[1:]:
            try:
                model_optimizer.add_param_group({'params': model_.parameters()})
            except:
                None

    for epoch in range(1, config.num_epoch + 1):
        
        # Train and validate
        if "spec" in training_mode:
            train_loss, train_acc, view_loss_v, view_loss_i, view_loss_c, loss_c, loss_v = model_train(model_list, temporal_contr_model, model_optimizer, temp_cont_optimizer,
                                             criterion, train_dl, config, device, training_mode,
                                             view_list, view_r, view_list_)
        
            logger.debug(f'\nEpoch : {epoch}\n'
                        f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n')
            logger.debug("variance loss " + "\t | \t".join([f'{view}      : {loss:.4f}' for view, loss in view_loss_v.items()]))
            logger.debug("invariance loss " + "\t | \t".join([f'{view}      : {loss:.4f}' for view, loss in view_loss_i.items()]))
            logger.debug("covariance loss " + "\t | \t".join([f'{view}      : {loss:.4f}' for view, loss in view_loss_c.items()]))
            logger.debug(f"covariance main       : {loss_c:.4f}\t | \tvariance main      : {loss_v:.4f}")

            # print(valid_dl)
            # exit()
            for d_name in valid_dl:
                t_dl = train_dls[d_name]
                v_dl = valid_dl[d_name]
                get_svm_data(model, t_dl, device)
                valid_loss, valid_acc, NMI, RI, _, _ = model_evaluate(model, temporal_contr_model, v_dl, device, training_mode, exp_dir=experiment_log_dir, epoch=epoch)
                logger.debug(d_name)
                logger.debug(f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}\t | \tDataset {d_name}')
                logger.debug(f'Valid Loss     : {valid_loss:.4f}\t | \tValid NMI          : {NMI:2.4f}\t | \tDataset {d_name}')
                logger.debug(f'Valid Loss     : {valid_loss:.4f}\t | \tValid RI           : {RI:2.4f}\t | \tDataset {d_name}')
        else:
            train_loss, train_acc = orig_model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, chain(*train_dl.values()), config, device, training_mode)
            for d_name in valid_dl:
                t_dl = train_dl[d_name]
                v_dl = valid_dl[d_name]
            valid_loss, valid_acc, _, _ = orig_model_evaluate(model, temporal_contr_model, v_dl, device, training_mode)
            if training_mode != 'self_supervised':  # use scheduler in all other modes.
                scheduler.step(valid_loss)

            logger.debug(f'\nEpoch : {epoch}\n'
                        f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                        f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')
        # if training_mode != 'self_supervised':  # use scheduler in all other modes.
        #     scheduler.step(valid_loss)
        
        if epoch % save_interval == 0:
            os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
            chkpoint = {'model_state_dict': model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
            torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_{epoch}.pt'))

    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    if "self_supervised" not in training_mode :  # no need to run the evaluation for self-supervised mode.
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}')

    logger.debug("\n################## Training is Done! #########################")

def ts_tcc(model, aug1, aug2, temporal_contr_model, device, config):

    predictions1, features1 = model(aug1)
    predictions2, features2 = model(aug2)

    # normalize projection feature vectors
    features1 = F.normalize(features1, dim=1)
    features2 = F.normalize(features2, dim=1)

    temp_cont_loss1, temp_cont_lstm_feat1 = temporal_contr_model(features1, features2)
    temp_cont_loss2, temp_cont_lstm_feat2 = temporal_contr_model(features2, features1)

    # normalize projection feature vectors
    zis = temp_cont_lstm_feat1 
    zjs = temp_cont_lstm_feat2 
    lambda1 = 1
    lambda2 = 0.7
    nt_xent_criterion = NTXentLoss(device, features1.shape[0], config.Context_Cont.temperature,
                                    config.Context_Cont.use_cosine_similarity)
    loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 +  nt_xent_criterion(zis, zjs) * lambda2
    return loss

def std_loss(x):
    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_x)) / 2
    return std_loss

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def cov_loss(x):
    batch_size = x.shape[0]
    num_features = x.shape[1]
    cov_x = (x.T @ x) / (batch_size - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            num_features
    )
    return cov_loss


def SameModel():
    global model_
    return deepcopy(model_)

def SameParameter():
    global model_
    return model_

def void(t):
    t = torch.zeros(t.shape, dtype=t.dtype, device=t.device)
    return t

def fft(t):
    # Compute complex-to-complex FFT
    fft_result = torch.fft.fft(t)

    # Split complex tensor into real and imaginary parts
    # real_parts = fft_result.real
    # imaginary_parts = fft_result.imag
    fft_result = torch.abs(fft_result)
    # Concatenate real and imaginary parts along a new dimension
    # fft_input = torch.cat((real_parts, imaginary_parts), dim=-1)
    return fft_result

def same(x):
    return x

#imaging time series as Gramian Angular Difference Field (GADF)
def polar_rep (data):
    #input datatype data : ndarray, 1xn, n-number of samples in each series
    #output datatype phi : ndarray, 1xn 
    #output datatype r : ndarray, 1xn
    
    phi=np.arccos(data)
    r=(np.arange(0,np.shape(data)[1])/np.shape(data)[1])+0.1
    return phi,r

def online_wavelet(data, resize_shape=224, wavelet="db1"):
    batch_size, channel, n = data.shape
    device = data.device
    dtype = data.dtype

    data = data.reshape(batch_size * channel, n).detach().cpu().numpy()
    coeffs = []
    for d in data:
        coef, freqs = pywt.cwt(d, resize_shape, wavelet)
        coef = torch.tensor(coef)
        coef = F.interpolate(torch.abs(coef).unsqueeze(0).unsqueeze(0), size=(resize_shape, resize_shape), 
                    mode='bilinear', align_corners=False).squeeze()
        coeffs.append(coef)
    
    coeffs = torch.stack(coeffs)
    # print(coeffs.shape)
    # exit()
    coeffs = coeffs.reshape((batch_size, channel, resize_shape , resize_shape))
    return coeffs.to(device=device, dtype=dtype)
    
# scaled_image = resize_transform(image)

def r_plot_aug(data_batch, delay=0, resize_shape=224):
    data_batch = strong_augmentation(data_batch)
    return r_plot(data_batch, delay=delay, resize_shape=resize_shape)

def GADF_aug(data_batch, resize_shape=224):
    data_batch = strong_augmentation(data_batch)
    return GADF(data_batch, resize_shape=resize_shape)

def r_plot(data_batch, delay=0, resize_shape=224):
    # Input datatype data_batch: tensor, size (batch_size, n)
    # Input datatype delay: int, delay embedding for RP formation, default value is 1
    # Output datatype rp_batch: tensor, size (batch_size, n - delay, n - delay), unthresholded recurrence plots for each series in the batch
    # print(data_batch.device)
    device = data_batch.device
    batch_size, channel, n = data_batch.shape
    data_batch = data_batch.reshape(batch_size * channel, n)
    data_length = n
    
    transformed = torch.zeros(batch_size * channel, 2, data_length - delay)
    transformed[:, 0, :] = data_batch[:, 0:data_length - delay]
    transformed[:, 1, :] = data_batch[:, delay:data_length]
    
    # Vectorized calculation of recurrence plots
    transformed_expanded = transformed.unsqueeze(3)  # Expand dimensions for broadcasting
    temp = transformed_expanded - transformed_expanded.permute(0, 1, 3, 2)
    temp2 = temp ** 2
    rp_batch = torch.sum(temp2, dim=1)
    rp_batch = rp_batch.reshape(batch_size, channel, n, n)
    if resize_shape > n:
        resize_shape = resize_shape // 2
    rp_batch = F.interpolate(rp_batch, size=(resize_shape, resize_shape), 
                  mode='bilinear', align_corners=False)
    desired_channels = 64
    if channel > desired_channels:
        # print(rp_batch.shape)
        rp_batch = rp_batch.view((batch_size, channel, resize_shape * resize_shape))
        rp_batch = F.adaptive_avg_pool2d(rp_batch, (desired_channels, resize_shape * resize_shape))
        rp_batch = rp_batch.view((batch_size, desired_channels, resize_shape, resize_shape))
    
    # print(rp_batch.shape)
    # exit()
    return rp_batch.detach().contiguous().to(device)
import torch.nn.functional as F
def GADF(data_batch, resize_shape=224):
    # Input datatype data_batch: tensor, size (batch_size, n)
    # Output datatype gadf_batch: tensor, size (batch_size, n, n), GADF for each series in the batch
    batch_size, channel, n = data_batch.shape
    device = data_batch.device
    data_batch = data_batch.cpu()
    data_batch = data_batch.reshape(batch_size * channel, n)
    datacos = data_batch.clone()
    datasin = torch.sqrt(1 - torch.clamp(datacos**2, 0, 1))
    gadf_batch = (datasin.unsqueeze(2) * datacos.unsqueeze(1)) - (datacos.unsqueeze(2) * datasin.unsqueeze(1))
    gadf_batch = gadf_batch.reshape(batch_size, channel, n, n)
    if resize_shape > n:
        resize_shape = resize_shape // 2
    gadf_batch = F.interpolate(gadf_batch, size=(resize_shape, resize_shape), 
                  mode='bilinear', align_corners=False)
    
    desired_channels = 64
    if channel > desired_channels:
        gadf_batch = gadf_batch.view((batch_size, channel, resize_shape * resize_shape))
        gadf_batch = F.adaptive_avg_pool2d(gadf_batch, (desired_channels, resize_shape * resize_shape))
        gadf_batch = gadf_batch.view((batch_size, desired_channels, resize_shape, resize_shape))
    # print(gadf_batch)
    # exit()
    
    return gadf_batch.detach().to(device)


def mtf_transform(data, resize_shape=224):
    
    batch, channel, n = data.shape
    device = data.device
    dtype= data.dtype
    data = data.reshape(batch * channel, n)
    MTF = MarkovTransitionField()
    mtf = MTF.fit_transform(data.detach().cpu().numpy())
    mtf = torch.tensor(mtf).to(device=device, dtype=dtype)
    mtf = mtf.unsqueeze(1)
    mtf = F.interpolate(mtf, size=(resize_shape, resize_shape), 
                  mode='bilinear', align_corners=False)
    mtf = mtf.reshape(batch, channel, resize_shape, resize_shape)
    return mtf

def get_gasf(data, resize_shape=224):
    batch, channel, n = data.shape
    device = data.device
    dtype= data.dtype
    data = data.reshape(batch * channel, n)
    # from pyts.image import GramianAngularField
    GASF = GramianAngularField(method='summation')
    gasf = GASF.fit_transform(data.detach().cpu().numpy())
    gasf = torch.tensor(gasf).to(device=device, dtype=dtype)
    gasf = gasf.unsqueeze(1)
    gasf = F.interpolate(gasf, size=(resize_shape, resize_shape), 
                  mode='bilinear', align_corners=False)
    gasf = gasf.reshape(batch, channel, resize_shape, resize_shape)
    return gasf

loss_func = nn.CrossEntropyLoss()
def multi_scale_loss(multi_scale_shapelet_engergy1, multi_scale_shapelet_engergy2, T=0.1):
    for qi, ki in zip(multi_scale_shapelet_engergy1, multi_scale_shapelet_engergy2):
        # print(qi.shape, ki.shape)
        logits = torch.einsum('nc,ck->nk', [nn.functional.normalize(qi, dim=1), nn.functional.normalize(ki, dim=1).t()])
        logits /= T
        #print(logits)
        labels = torch.arange(qi.shape[0], dtype=torch.long, device=qi.device)
        loss = loss_func(logits, labels)
        return loss

def vicreg(model_list, data, view_list, model_optimizer, view_r, cov_loss_r=25, std_loss_r=25, repr_loss_r=0.1):
    main_repr = None
    total_loss = 0
    base_model = model_list[0]
    base_view = view_list[0]

    loss_list_i = []
    loss_list_v = []
    loss_list_c = []
    loss_list_mv = []
    loss_list_mc = []

    if len(model_list[1:]) != len(view_list[1:]) or len(model_list[1:]) != len(view_r):
        raise RuntimeError(f"model list lenght: {len(model_list[1:])}, view list length: {len(view_list[1:])}, view coeff: {len(view_r)}")
    torch.autograd.set_grad_enabled(True)
    model_optimizer.zero_grad()
    for model, view, v_r in zip(model_list[1:], view_list[1:], view_r):
        loss = 0 
        data_ = view(data)
        if torch.isnan(data_.detach()).any():
            print(view)
            raise RuntimeError("nan in data")
        if isinstance(base_model, LearningShapeletsModelMixDistances) or \
            isinstance(base_model.module, LearningShapeletsModelMixDistances):
            main_repr, base_features, multi_scale_shapelet_engergy1 = base_model(base_view(data), train_mode="train_spec")
        else:
            main_repr, base_features  = base_model(base_view(data), train_mode="train_spec")
        
        if isinstance(model, LearningShapeletsModelMixDistances) or \
            isinstance(model.module, LearningShapeletsModelMixDistances):
            repr_, features, multi_scale_shapelet_engergy2 = model(data_, train_mode="train_spec")
        else:
            repr_, features = model(data_, train_mode="train_spec")

        repr_loss = F.mse_loss(main_repr, repr_)
        loss_list_i.append(repr_loss.detach().cpu().item())
        loss += repr_loss * repr_loss_r * v_r
        features = features - features.mean(dim=0)
        cov_loss_ = cov_loss(features) 
        loss_list_c.append(cov_loss_.detach().cpu().item())
        
        m_cov_loss_ = cov_loss(base_features)
        loss_list_mc.append(m_cov_loss_.detach().cpu().item())
        cov_loss_ += m_cov_loss_

        std_loss_ = std_loss(features) 
        loss_list_v.append(std_loss_.detach().cpu().item())

        m_std_loss_ = std_loss(base_features) 
        loss_list_mv.append(m_std_loss_.detach().cpu().item())
        std_loss_ += m_std_loss_

        loss += cov_loss_ * cov_loss_r
        loss += std_loss_ * std_loss_r

        # if isinstance(model, LearningShapeletsModelMixDistances):
        #     loss += multi_scale_loss(multi_scale_shapelet_engergy1, multi_scale_shapelet_engergy2) * v_r
        # print(loss)
        # exit()
        loss.backward()
        total_loss = total_loss + loss
        # print(total_loss)
    model_optimizer.step()
    torch.autograd.set_grad_enabled(False)
    # print(loss_list)
    
    loss_list_mv = np.average(loss_list_mv)
    loss_list_mc = np.average(loss_list_mc)
    return total_loss, loss_list_v, loss_list_i, loss_list_c, loss_list_mv, loss_list_mc

def get_jrp(data, resize_shape=224):
    device = data.device
    dtype = data.dtype
    data = data.detach().cpu().numpy()
    jrp = JointRecurrencePlot()
    jrp = jrp.fit_transform(data)
    jrp = torch.tensor(jrp).to(device=device, dtype=dtype)
    jrp = jrp.unsqueeze(1)
    jrp = F.interpolate(jrp, size=(resize_shape, resize_shape), 
                  mode='bilinear', align_corners=False)
    return jrp

sfa_ = None
db1_ = None
coif5_ = None
dmey_ = None
strong_augmentation_ = None

def get_sfa(data):
    global sfa_
    return sfa_

def coif5(data):
    global coif5_
    return coif5_

def dmey(data):
    global dmey_
    return dmey_

def db1(data, resize_shape=224):
    global db1_
    if data.shape[-1] < 224:
        db1_ = F.interpolate(db1_, size=(resize_shape // 2, resize_shape // 2), 
                    mode='bilinear', align_corners=False)
    return db1_
def strong_augmentation(data):
    global strong_augmentation_
    return strong_augmentation_

view_constructor = {
    # "same": (same, SameModel)
    "fft": (fft, ResCNN),
    "rp": (r_plot, ResNet12),
    "aug_rp": (r_plot_aug, ResNet12),
    "sfa": (get_sfa, SFA_Bert),
    "gadf": (GADF, ResNet12),
    "aug_gadf": (GADF_aug, ResNet12),
    "gasf": (get_gasf, ResNet12),
    "void": (void, SameModel),
    "db1": (db1, ResNet12),
    "dmey": (dmey, ResNet12),
    "coif5": (coif5, ResNet12),
    "o_db1": (partial(online_wavelet, wavelet="db1"),ResNet12),
    "o_dmey": (partial(online_wavelet, wavelet="dmey"),ResNet12),
    "o_coef5": (partial(online_wavelet, wavelet="coef5"),ResNet12),
    "strong_aug": (strong_augmentation, SameParameter),
    "shapelet": (same, LearningShapeletsModelMixDistances)
}

def model_train(model_list, temporal_contr_model, model_optimizer, temp_cont_optimizer, 
                criterion, train_loader, config, device, training_mode, view_list=None, view_r=None, view_list_=None):
    global sfa_, db1_, strong_augmentation_, coif5_, dmey_
    total_loss = []
    total_acc = []
    # model_list = []
    # view_list_ = []

    view_loss_v = dict()
    view_loss_i = dict()
    view_loss_c = dict()
    for view in view_list:
        view_loss_v[view] = []
        view_loss_i[view] = []
        view_loss_c[view] = []

    (model.train() for model in model_list)
    
    temporal_contr_model.train()
    total_loss_c = []
    total_loss_v = []

    for batch_idx, (data, labels, aug1, aug2, sfa, db1, coif5, dmey) in tqdm(enumerate(train_loader), total=len(train_loader.dataset) // config.batch_size):
        # send to device
        # aug1, aug2 = slice_time_series(aug1, aug2)
        sfa_ = sfa.to(device)
        db1_ = db1.float().to(device)
        coif5_ = coif5.float().to(device)
        dmey_ = dmey.float().to(device)

        strong_augmentation_ = aug2.float().to(device)
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        if  "self_supervised" in training_mode:
            loss = ts_tcc(model_list[0], temporal_contr_model, aug1, aug2, device, config)
        
        if "vicreg" in training_mode:
            loss, view_rep_loss_v, view_rep_loss_i, view_rep_loss_c, loss_list_mv, loss_list_mc  = vicreg(model_list, aug1, view_list_, model_optimizer, view_r)
            for view, v_loss, i_loss, c_loss in zip(view_loss_i, view_rep_loss_v, view_rep_loss_i, view_rep_loss_c):
                view_loss_v[view].append(v_loss)
                view_loss_i[view].append(i_loss)
                view_loss_c[view].append(c_loss)
            total_loss_c.append(loss_list_mc)
            total_loss_v.append(loss_list_mv)

        else: # supervised training or fine tuining
            output = model(data)
            predictions, features = output
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

            loss.backward()
            model_optimizer.step()
            temp_cont_optimizer.step()
        
        total_loss.append(loss.item())

    total_loss = torch.tensor(total_loss).mean()

    if "self_supervised" in training_mode :
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    
    for view in view_loss_i:
        view_loss_i[view] = np.average(view_loss_i[view])
        view_loss_v[view] = np.average(view_loss_v[view])
        view_loss_c[view] = np.average(view_loss_c[view])
    loss_c = np.average(total_loss_c)
    loss_v = np.average(total_loss_v)
    return total_loss, total_acc, view_loss_v, view_loss_i, view_loss_c, loss_c, loss_v

train_features = []
train_labels = []
def get_svm_data(model, train_loader, device):
    global train_labels, train_features
    train_labels = []
    train_features = []
    for batch_idx, (data, labels, aug1, aug2, sfa ,db1, coif5, dmey) in enumerate(train_loader):
        data, labels = data.float().to(device), labels.long().to(device)
        output = model(data)
        vec = F.max_pool1d(output[1], kernel_size=output[1].shape[2]).squeeze(-1)
        train_features.append(vec.detach().cpu().numpy())
        train_labels.append(labels.detach().cpu().numpy())
    train_features = np.concatenate(train_features,axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode, exp_dir, epoch):
    global train_features, train_labels
    # svm = SVC( kernel="rbf", gamma='scale')
    if 'ad' not in training_mode: 
        svm = LogisticRegression()
        svm.fit(train_features, train_labels)
    else:
        clf = IsolationForest(random_state=1000).fit(train_features)

    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    valid_features = []
    valid_labels = []

    # if "self_supervised" in training_mode:
    #     features_ = []
    #     labels_ = []

    with torch.no_grad():
        for data, labels, _, _, _, _, _, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            if "self_supervised" in training_mode or \
            "vicreg" in training_mode:
                # pass
                output = model(data)
                vec = F.max_pool1d(output[1], kernel_size=output[1].shape[2]).squeeze(-1)
                valid_features.append(vec.detach().cpu().numpy())
                valid_labels.append(labels.detach().cpu().numpy())
            else:
                output = model(data)
                # compute loss
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if "self_supervised" not in training_mode and "vicreg" not in training_mode:
        total_loss = torch.tensor(total_loss).mean()  # average loss
        total_acc = torch.tensor(total_acc).mean()  # average acc
        return total_loss, total_acc, outs, trgs
    else:
        total_loss = 0
        valid_features = np.concatenate(valid_features, axis=0)
        valid_labels = np.concatenate(valid_labels, axis=0)
        if 'ad' not in training_mode:
            total_acc = svm.score(valid_features, valid_labels)
        else:
            preds = clf.predict(valid_features)
            total_acc = f1_score(valid_labels, preds, pos_label=-1)

            save_dir = os.path.join(exp_dir, test_dl.dataset.d_name)
            os.makedirs(save_dir,exist_ok=True)
            with open(os.path.join(save_dir, "train_ep" + str(epoch) + ".pkl"), mode="wb") as t:
                pkl.dump(train_features, t)
            with open(os.path.join(save_dir, "test_ep" + str(epoch) + ".pkl"), mode="wb") as t:   
                pkl.dump({"features":valid_features, "labels": valid_labels}, t)             
        label_num = np.max(valid_labels) + 1
        kmeans = KMeans(label_num * 2 + 2)
        pred = kmeans.fit_predict(valid_features)
        NMI_score = normalized_mutual_info_score(valid_labels, pred)
        RI_score = rand_score(valid_labels, pred)
        ans = {"NMI":NMI_score, "RI":RI_score}

        return total_loss, total_acc, NMI_score, RI_score, [], []
        
    
