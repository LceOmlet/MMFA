from ..registry.registry import TRAIN_FN, TRAIN_STEP, PRETRAIN_STEP, EVALUATOR, TRAIN_LOOP_INIT, TRAINER_INIT, TEST_METHODS, MODELS
from collections import OrderedDict
import torch
from ..utils.loss import hierarchical_contrastive_loss
import numpy as np
from sklearn.linear_model import Ridge
import torch.nn.functional as F
from torch import nn 
from ..utils.utils import list2array
from collections.abc import Collection

def have_None(inp):
    # dict is not allowed here
    if isinstance(inp, dict):
        return True
    if inp is None:
        return True
    elif isinstance(inp, Collection) \
        and not isinstance(inp, torch.Tensor) \
            and not isinstance(inp, np.ndarray):
        return have_None(inp[0])
    else:
        return False

def fit_imputation(reprs, targets, masks, valid_ratio, loss_module):
    reprs = list2array(reprs)
    targets = list2array(targets)
    masks = list2array(masks)
    alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    valid_split = int(len(reprs) * valid_ratio)
    valid_repr, train_repr = reprs[:valid_split], reprs[valid_split:]
    valid_targets, train_targets = targets[: valid_split], targets[valid_split:]
    valid_masks, train_masks = masks[:valid_split], masks[valid_split :] 
    valid_results = []
    for alpha in alphas:
        target_shape = train_targets.shape[1:]
        lr = Ridge(alpha=alpha).fit(
            train_repr.reshape(train_repr.shape[0], -1), 
            train_targets.reshape(train_repr.shape[0], -1)
        )
        valid_pred = lr.predict(valid_repr.reshape((valid_repr.shape[0], -1)))
        valid_pred = valid_pred.reshape((valid_split, target_shape[0], target_shape[1]))
        score = loss_module(torch.tensor(valid_targets), torch.tensor(valid_pred), torch.tensor(valid_masks)).detach().cpu().numpy()
        score = np.mean(score)
        valid_results.append(score)
    best_alpha = alphas[np.argmin(valid_results)]
    lr = Ridge(alpha=best_alpha)
    lr.fit(reprs.reshape((reprs.shape[0], -1)), targets.reshape((reprs.shape[0], -1)))
    return lr

@TRAIN_FN.register("default")
def train_epoch(model, dataloader, task, device, 
                model_name, print_interval, print_callback, val_loss_module,
                logger,
                epoch_num=None, **kwargs):
    epoch_metrics = OrderedDict()
    model.train()
    epoch_loss = 0  # total loss of epoch
    total_active_elements = 0  # total unmasked elements in epoch
    init_loop = TRAIN_LOOP_INIT.get(model_name)
    kwargs_init = {}
    if init_loop is not None:
        init_kwargs = dict(model=model, device=device)
        kwargs_init.update(init_loop(**init_kwargs))

    train_step = TRAIN_STEP.get(task)
    if train_step is not None:
        for i, batch in enumerate(dataloader):
            train_step_kwargs = dict(batch=batch, model=model, device=device, model_name=model_name)
            train_step_kwargs.update(kwargs)
            train_step_kwargs.update(kwargs_init)
            model.train()
            loss = train_step(**train_step_kwargs)

            if isinstance(loss, dict):
                metrics = dict()
                for ll in loss:
                    metrics[ll] = loss[ll].item()
                batch_loss = loss["loss"]
                loss = loss["loss"]
            
            elif isinstance(loss, torch.Tensor):
                if len(loss.shape):
                    loss = loss.reshape(loss.shape[0], -1)
                    loss = torch.mean(loss, dim=-1)
                if not loss.shape:
                    loss = loss.unsqueeze(0)
                
                batch_loss = torch.sum(loss)
                mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization
                metrics = {"loss": mean_loss.item()}

            if i % print_interval == 0:
                ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                print_callback(i, metrics, prefix='Training ' + ending, total_batches=len(dataloader))

            with torch.no_grad():
                if loss.dim() == 0:
                    total_active_elements += 1
                else:
                    total_active_elements += len(loss)
                epoch_loss += batch_loss.item()  # add total loss of batch
    else:
        epoch_loss = 0.
        total_active_elements = 1


    epoch_loss = epoch_loss / max(total_active_elements, 1)  # average loss per element for whole epoch
    epoch_metrics['epoch'] = epoch_num
    epoch_metrics['loss'] = float(epoch_loss)
    

    return epoch_metrics



@EVALUATOR.register("default")
class PretrainAgg:
    def __init__(self, evaluator) -> None:
        self.per_batch_train = dict()
        self.per_batch_valid = dict()
        # print(evaluator)
        # exit()
        self.evaluator = TEST_METHODS.get(evaluator)
    
    def train_module(self, val_loss_module, logger, valid_ratio=0.125, **kwargs):
        # if self.per_batch_train.get("repr") is None:
        #     logger.info("The batches are not collected during training.")
        #     return None
        
        test_module_kwargs = dict()
        for k in self.per_batch_train:
            try:
                test_module_kwargs[k] = list2array(self.per_batch_train[k])
            except Exception as e:
                test_module_kwargs[k] = None
                logger.info(f"Caught an exception: {e}")
                logger.info(f"collated data: {k} can not be converted into array.")

        
        test_module_kwargs.update(dict(valid_ratio=valid_ratio, loss_module=val_loss_module))

        if self.evaluator is not None:
            self.model = self.evaluator(**test_module_kwargs)
        # raise RuntimeError()
    
    def append_train(self, model, **kwargs):
        with torch.no_grad():
            model.eval()
            kwargs.update(self.evaluator.collate(model, **kwargs))
            model.train()
        for k in kwargs:
            if have_None(kwargs[k]):
                self.per_batch_train[k] = None
                continue
            if k in self.per_batch_train:
                self.per_batch_train[k].append(kwargs[k])
            else:
                self.per_batch_train[k] = [kwargs[k]]

    def append_valid(self, model, **kwargs):
        if self.evaluator is not None:
            with torch.no_grad():
                model.eval()
                kwargs.update(self.evaluator.collate(model, **kwargs))
                model.train()
        for k in kwargs:
            if kwargs[k][0] is None:
                self.per_batch_train[k] = None
                continue
            if k in self.per_batch_valid:
                self.per_batch_valid[k].append(kwargs[k])
            else:
                self.per_batch_valid[k] = [kwargs[k]]
    
    def infer(self, dset="valid", **kwargs):
        kwargs_eval = dict()
        if dset == "valid":
            per_batch = self.per_batch_valid
        elif dset == "train":
            per_batch = self.per_batch_train
        for k in per_batch:
            # print(k)
            # for it in self.per_batch_valid[k]:
            #     print(it.shape)
            kwargs_eval[k] = list2array(per_batch[k])
        kwargs_eval.update(kwargs)
        results = self.model.evaluate(per_batch=per_batch, **kwargs_eval)
        self.per_batch_valid.update(results)
        # results = self.model.evaluate(per_batch=self.per_batch_train, **kwargs_eval)
        return results
    
    def clear(self):
        self.per_batch_train = dict()
        self.per_batch_valid = dict()

    def get(self, record, train=False):
        if train:
            return list2array(self.per_batch_train[record])
        else:
            return list2array(self.per_batch_valid[record])


@TRAIN_STEP.register("classification")
def step_classification(batch, model, device, loss_module, optimizer, **kwargs):
    X, targets, padding_masks, IDs = tuple(batch.values())
    targets = targets.to(device)
    padding_masks = padding_masks.to(device)  # 0s: ignore
    # regression: (batch_size, num_labels); classification: (batch_size, num_classes) of logits
    predictions = model(X.to(device), padding_masks)

    loss = loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
    batch_loss = torch.mean(loss)
    optimizer.zero_grad()
    batch_loss.backward()
    # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
    optimizer.step()
    return loss




@TRAIN_STEP.register("anomaly_detection")
def step_anomaly_detection(batch, model, device, loss_module, optimizer, **kwargs):
    X, targets, padding_masks, IDs = tuple(batch.values())
    padding_masks = padding_masks.to(device)
    predictions = model(X.to(device), padding_masks)
    loss = loss_module(predictions, X)
    batch_loss = torch.mean(loss)
    optimizer.zero_grad()
    batch_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
    optimizer.step()
    return loss

@TRAIN_STEP.register("regression")
def step_regression(batch, model, device, loss_module, optimizer, **kwargs):
    X, preds, padding_masks, IDs = tuple(batch.values())
    predictions = model(X.to(device), padding_masks)
    # print(predictions.shape, preds.shape)
    loss = loss_module(predictions, preds)
    batch_loss = torch.mean(loss)
    optimizer.zero_grad()
    batch_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
    optimizer.step()
    return loss

@TRAIN_STEP.register("imputation")
def step_imputation(batch, model, device, loss_module, optimizer, **kwargs):
    X, target, target_masks, padding_masks, label, IDs = tuple(batch.values())
    target_masks = target_masks.to(device)  # 1s: mask and predict, 0s: unaffected input (ignore)
    padding_masks = padding_masks.to(device)  # 0s: ignore
    target = target.to(device)
    predictions = model(X.to(device), padding_masks)
    loss = loss_module(target, predictions, target_masks)
    batch_loss = torch.mean(loss)
    optimizer.zero_grad()
    batch_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
    optimizer.step()
    return loss

@TRAIN_STEP.register("pretraining")
def step_pretraining(model_name, **kwargs):
    return PRETRAIN_STEP.get(model_name)(**kwargs)

@PRETRAIN_STEP.register("mvts_transformer")
def step_mvts_transformer(batch, model, device, loss_module, optimizer, evaluator, **kwargs):
    X, target, target_masks, padding_masks, label, IDs = tuple(batch.values())
    # print(torch.mean(torch.abs(X)))
    target = target.to(device)
    X = X.to(device)
    X[target_masks] = 0
    target_masks = target_masks.to(device)  # 1s: mask and predict, 0s: unaffected input (ignore)
    padding_masks = torch.ones(X.shape[0], X.shape[2]).to(dtype=bool)
    padding_masks = padding_masks.to(device)  # 0s: ignore
    predictions = model(X.to(device), padding_masks)  # (batch_size, padded_length, feat_dim)
    # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
    target_masks = target_masks * padding_masks.unsqueeze(-1)
    loss = loss_module(predictions, target, target_masks)  # (num_active,) individual loss (square error per element) for each active value in batch
    optimizer.zero_grad()
    loss = torch.mean(loss)
    loss.backward()
    # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
    optimizer.step()
    # evaluator.append_train(model, X=X, target=target, label=label, mask=target_masks)
    return loss

@PRETRAIN_STEP.register("ts2vec")
def step_ts2vec(batch, model, device, loss_module, optimizer, evaluator, **kwargs):
    gt1, gt2, crop_l, m1, m2, X, mask, label, IDs = tuple(batch.values())
    gt1 = gt1.to(device)
    gt2 = gt2.to(device)

    out1 = model._net(gt1, m1)[:, -crop_l:]
    out2 = model._net(gt2, m2)[:, :crop_l]
    loss = loss_module(
        out1,
        out2
    )
    # evaluator.append_train(model, X=X, label=label,mask=mask)
    optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
    optimizer.step()
    model.net.update_parameters(model._net)
    return loss

@PRETRAIN_STEP.register("csl")
def step_csl(batch, model, device, loss_module, optimizer, optim_config, evaluator,
             C_accu_q, c_normalising_factor_q, C_accu_k, c_normalising_factor_k, **kwargs):
    X, x_k, x_q, mask, label, IDs = tuple(batch.values())
    X, x_k, x_q = X.to(device), x_k.to(device), x_q.to(device)
    # evaluator.append_train(model, X=X, label=label, mask=mask)
    num_shapelet_lengths = len(model.shapelets_size_and_len)
    num_shapelet_per_length = model.num_shapelets // num_shapelet_lengths
    with torch.autograd.set_detect_anomaly(True):
        q = model(x_q, optimize=None, masking=False)
        k = model(x_k, optimize=None, masking=False)
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        logits = torch.einsum('nc,ck->nk', [q, k.t()])
        logits /= optim_config['T']
        labels = torch.arange(q.shape[0], dtype=torch.long).to(device)
        loss = loss_module(logits, labels)
        q_sum = None
        q_square_sum = None
        
        
        k_sum = None
        k_square_sum = None
        
        loss_sdl = 0
        c_normalising_factor_q = optim_config['alpha'] * c_normalising_factor_q + 1
        
        c_normalising_factor_k = optim_config['alpha'] * c_normalising_factor_k + 1
        #print(q.shape)
        for length_i in range(num_shapelet_lengths):
            qi = q[:, length_i * num_shapelet_per_length: (length_i + 1) * num_shapelet_per_length]
            ki = k[:, length_i * num_shapelet_per_length: (length_i + 1) * num_shapelet_per_length]
            
            logits = torch.einsum('nc,ck->nk', [nn.functional.normalize(qi, dim=1), nn.functional.normalize(ki, dim=1).t()])
            logits /= optim_config['T']
            #print(logits)
            loss += loss_module(logits, labels)
            
            
            if q_sum == None:
                q_sum = qi
                q_square_sum = qi * qi
            else:
                q_sum = q_sum + qi
                q_square_sum = q_square_sum + qi * qi
                
            C_mini_q = torch.matmul(qi.t(), qi) / (qi.shape[0] - 1)
            C_accu_t_q = optim_config['alpha'] * C_accu_q[length_i] + C_mini_q
            C_appx_q = C_accu_t_q / c_normalising_factor_q
            loss_sdl += torch.norm(C_appx_q.flatten()[:-1].view(C_appx_q.shape[0] - 1, C_appx_q.shape[0] + 1)[:, 1:], 1).sum()
            #print(length_i)
            C_accu_q[length_i] = C_accu_t_q.detach()
            
            if k_sum == None:
                k_sum = ki
                k_square_sum = ki * ki
            else:
                k_sum = k_sum + ki
                k_square_sum = k_square_sum + ki * ki
                
            C_mini_k = torch.matmul(ki.t(), ki) / (ki.shape[0] - 1)
            C_accu_t_k = optim_config['alpha'] * C_accu_k[length_i] + C_mini_k
            C_appx_k = C_accu_t_k / c_normalising_factor_k
            loss_sdl += torch.norm(C_appx_k.flatten()[:-1].view(C_appx_k.shape[0] - 1, C_appx_k.shape[0] + 1)[:, 1:], 1).sum()
            #print(length_i)
            C_accu_k[length_i] = C_accu_t_k.detach()
        
        loss_cca = 0.5 * torch.sum(q_square_sum - q_sum * q_sum / num_shapelet_lengths) + 0.5 * torch.sum(k_square_sum - k_sum * k_sum / num_shapelet_lengths)
        loss += optim_config['l3'] * (loss_cca + optim_config['l4'] * loss_sdl)       
        optimizer.zero_grad()
        # print(list(model.state_dict().items())[10])
        loss.backward()
        optimizer.step()

    return loss


@PRETRAIN_STEP.register("ts_tcc")
def step_ts_tcc(batch, model, device, loss_module, optimizer, temporal_contr_optimizer, evaluator, **kwargs):
    X, aug1, aug2, mask, label, IDs = tuple(batch.values())
    data = X.float().to(device)
    X = data
    aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

    # optimizer
    optimizer.zero_grad()
    temporal_contr_optimizer.zero_grad()
    predictions1, features1 = model.model(aug1)
    predictions2, features2 = model.model(aug2)

    # normalize projection feature vectors
    features1 = F.normalize(features1, dim=1)
    features2 = F.normalize(features2, dim=1)

    temp_cont_loss1, temp_cont_lstm_feat1 = model.tenporal_contr_model(features1, features2)
    temp_cont_loss2, temp_cont_lstm_feat2 = model.tenporal_contr_model(features2, features1)

    # normalize projection feature vectors
    zis = temp_cont_lstm_feat1 
    zjs = temp_cont_lstm_feat2 

    # compute loss
    lambda1 = 1
    lambda2 = 0.7
    nt_xent_criterion = loss_module
    loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 +  nt_xent_criterion(zis, zjs) * lambda2
    # if l2_reg:
    #     loss = mean_loss + l2_reg * l2_reg_loss(model)
    # else:
    #     loss = mean_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
    optimizer.step()
    temporal_contr_optimizer.step()
    target = X.detach().clone()
    # evaluator.append_train(model, X=X, label=label,mask=mask)
    return loss

@PRETRAIN_STEP.register("t_loss")
def step_t_loss(batch, model, device, loss_module, optimizer, t_loss_train, evaluator, **kwargs):
    X, target, target_masks, padding_masks, label, IDs = tuple(batch.values())
    X = X.to(device)
    target =target.to(device)
    target_masks = target_masks.to(device)  # 1s: mask and predict, 0s: unaffected input (ignore)
    padding_masks = padding_masks.to(device)  # 0s: ignore
    X_ = X
    X_[target_masks] = 0
    loss = loss_module(target, model, 
                            t_loss_train, save_memory=False)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
    optimizer.step()
    # evaluator.append_train(model, X=X, label=label, mask=target_masks)
    return loss

@PRETRAIN_STEP.register("mmfa_rec")
@PRETRAIN_STEP.register("mmfa")
def step_mmfa(batch, model, device, loss_module, optimizer, models, evaluator, **kwargs):
    X = batch["X"].to(device)
    for t in batch["features"]:
        batch["features"][t] = batch["features"][t].to(device)
    
    loss = loss_module(model, X, models, batch["features"], optimizer)
    return loss

@TRAIN_LOOP_INIT.register("csl")
def init_train_variables(model, device, **kwargs):
    c_normalising_factor_q = torch.tensor([0], dtype=torch.float).to(device)
    C_accu_q = [torch.tensor([0], dtype=torch.float).to(device) for _ in range(len(model.shapelets_size_and_len))]
    c_normalising_factor_k = torch.tensor([0], dtype=torch.float).to(device)
    C_accu_k = [torch.tensor([0], dtype=torch.float).to(device) for _ in range(len(model.shapelets_size_and_len))]
    return {
        "c_normalising_factor_k": c_normalising_factor_k,
        "c_normalising_factor_q": c_normalising_factor_q,
        "C_accu_k": C_accu_k,
        "C_accu_q": C_accu_q
    }

@TRAINER_INIT.register("pretraining")
def train_init_pretraining(model_name, **kwargs):
    initer = TRAINER_INIT.get(model_name)
    if initer is not None:
        return initer(**kwargs)
    else:
        return {}

@TRAINER_INIT.register("t_loss")
def train_init_t_loss(device, train_ds, **kwargs):
    t_loss_train = torch.cat([torch.tensor(X[0]).to(device).unsqueeze(0) for X in train_ds], dim=0)
    return {
        "t_loss_train": t_loss_train
    }

@TRAINER_INIT.register("ts_tcc")
def train_init_ts_tcc(model, optim_config, **kwargs):
    temporal_contr_optimizer = torch.optim.Adam(model.parameters_tc(), lr=optim_config["lr"], 
                        betas=(optim_config["beta1"], optim_config["beta2"]), weight_decay=3e-4)
    return {
        "temporal_contr_optimizer": temporal_contr_optimizer
    }

@TRAINER_INIT.register("mmfa_rec")
@TRAINER_INIT.register("mmfa")
def train_init_ts_tcc(optimizer, optim_config, device, **kwargs):
    transformations = optim_config["transformations"]
    models = dict()
    for t in transformations:
        transformations[t]["model"]["device"] = device
        models[t] = MODELS.get(transformations[t]["model"]["model_name"]) \
            (**transformations[t]["model"])
        if transformations[t]["model"]["model_name"] in ["Gemma", "TinyLlama"]:
            pass
        elif isinstance(device, list):
            # exit()
            models[t] = models[t].to(device[0])
            models[t] = torch.nn.DataParallel(models[t], device_ids=device)
        else:
            models[t] = models[t].to(device)
        params = {'params': models[t].parameters(), }
        if "lr" in transformations[t]["model"]:
            print( f"{transformations[t]['model']} learning rate: {transformations[t]['model']['lr']}")
            params["lr"] = transformations[t]["model"]["lr"]
        optimizer.add_param_group(params)
        
            
    return {
        "models": models
    } 
