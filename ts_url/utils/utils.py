import json
import os
import sys
import builtins
import functools
import time
import ipdb
from copy import deepcopy

import numpy as np
import torch
import xlrd
import xlwt
from xlutils.copy import copy
from torch import nn

import logging
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def list2array(cvt_list):
    if cvt_list is None:
        return None
    if isinstance(cvt_list, list) or isinstance(cvt_list, tuple):
        if isinstance(cvt_list[0], list) or isinstance(cvt_list[0], tuple):
            cvt_list_ = []
            for cvt in cvt_list:
                cvt_list_.append(torch.tensor(cvt))
            cvt_list = cvt_list_
        if len(cvt_list) == 0:
            return np.array([])
        if isinstance(cvt_list[0], torch.Tensor):
            cvt_list = torch.cat(cvt_list, dim=0)
        elif isinstance(cvt_list[0], np.ndarray):
            cvt_list = np.concatenate(cvt_list, axis=0)
        else:
            print(cvt_list[0])
            raise NotImplementedError
    if isinstance(cvt_list, torch.Tensor):
        cvt_list = cvt_list.detach().cpu().numpy()
    return cvt_list

def reduce(reprs, method):
    if len(reprs.shape) == 2:
        return reprs
    if method == "mean":
        reprs = torch.mean(reprs, dim=-1)
    elif method == "max":
        reprs = torch.mean(reprs, dim=-1).data
    elif method == "last":
        reprs = reprs[..., -1]
    return reprs

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time} secs")
        return value
    return wrapper_timer


def save_model(path, save_name, epoch, model, optimizer=None, optim_config=None, model_config=None, model_name=""):
    if model_name is None:
        model_name = "joint"
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict,
            'optim_config': optim_config,
            'model_config': model_config
            }
    with open(os.path.join(path,model_name + "_model.json"), "w") as model_, \
    open(os.path.join(path,model_name + "_optim.json"), "w") as optim_:
        print(model_config)
        json.dump(model_config, model_, indent="    ")
        json.dump(optim_config, optim_, indent="    ")
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, os.path.join(path, save_name))


def load_model(model, model_path, optimizer=None, resume=False, change_output=False,
               lr=None, lr_step=None, lr_factor=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = deepcopy(checkpoint['state_dict'])
    if change_output:
        for key, val in checkpoint['state_dict'].items():
            if key.startswith('output_layer'):
                state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    print('Loaded model from {}. Epoch: {}'.format(model_path, checkpoint['epoch']))

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for i in range(len(lr_step)):
                if start_epoch >= lr_step[i]:
                    start_lr *= lr_factor[i]
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def load_config(config_filepath):
    """
    Using a json file with the master configuration (config file for each part of the pipeline),
    return a dictionary containing the entire configuration settings in a hierarchical fashion.
    """

    with open(config_filepath) as cnfg:
        config = json.load(cnfg)

    return config


def create_dirs(dirs):
    """
    Input:
        dirs: a list of directories to create, in case these directories are not found
    Returns:
        exit_code: 0 if success, -1 if failure
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def export_performance_metrics(filepath, metrics_table, header, book=None, sheet_name="metrics"):
    """Exports performance metrics on the validation set for all epochs to an excel file"""

    if book is None:
        book = xlwt.Workbook()  # new excel work book

    book = write_table_to_sheet([header] + metrics_table, book, sheet_name=sheet_name)

    book.save(filepath)
    logger.info("Exported per epoch performance metrics in '{}'".format(filepath))

    return book


def write_row(sheet, row_ind, data_list):
    """Write a list to row_ind row of an excel sheet"""

    row = sheet.row(row_ind)
    for col_ind, col_value in enumerate(data_list):
        row.write(col_ind, col_value)
    return


def write_table_to_sheet(table, work_book, sheet_name=None):
    """Writes a table implemented as a list of lists to an excel sheet in the given work book object"""

    sheet = work_book.add_sheet(sheet_name)

    for row_ind, row_list in enumerate(table):
        write_row(sheet, row_ind, row_list)

    return work_book


def export_record(filepath, values):
    """Adds a list of values as a bottom row of a table in a given excel file"""

    read_book = xlrd.open_workbook(filepath, formatting_info=True)
    read_sheet = read_book.sheet_by_index(0)
    last_row = read_sheet.nrows

    work_book = copy(read_book)
    sheet = work_book.get_sheet(0)
    write_row(sheet, last_row, values)
    work_book.save(filepath)


def register_record(filepath, timestamp, experiment_name, best_metrics, final_metrics=None, comment=''):
    """
    Adds the best and final metrics of a given experiment as a record in an excel sheet with other experiment records.
    Creates excel sheet if it doesn't exist.
    Args:
        filepath: path of excel file keeping records
        timestamp: string
        experiment_name: string
        best_metrics: dict of metrics at best epoch {metric_name: metric_value}. Includes "epoch" as first key
        final_metrics: dict of metrics at final epoch {metric_name: metric_value}. Includes "epoch" as first key
        comment: optional description
    """
    metrics_names, metrics_values = zip(*best_metrics.items())
    row_values = [timestamp, experiment_name, comment] + list(metrics_values)
    if final_metrics is not None:
        final_metrics_names, final_metrics_values = zip(*final_metrics.items())
        row_values += list(final_metrics_values)

    if not os.path.exists(filepath):  # Create a records file for the first time
        logger.warning("Records file '{}' does not exist! Creating new file ...".format(filepath))
        directory = os.path.dirname(filepath)
        if len(directory) and not os.path.exists(directory):
            os.makedirs(directory)
        header = ["Timestamp", "Name", "Comment"] + ["Best " + m for m in metrics_names]
        if final_metrics is not None:
            header += ["Final " + m for m in final_metrics_names]
        book = xlwt.Workbook()  # excel work book
        book = write_table_to_sheet([header, row_values], book, sheet_name="records")
        book.save(filepath)
    else:
        try:
            export_record(filepath, row_values)
        except Exception as x:
            alt_path = os.path.join(os.path.dirname(filepath), "record_" + experiment_name)
            logger.error("Failed saving in: '{}'! Will save here instead: {}".format(filepath, alt_path))
            export_record(alt_path, row_values)
            filepath = alt_path

    logger.info("Exported performance record to '{}'".format(filepath))


class Printer(object):
    """Class for printing output by refreshing the same line in the console, e.g. for indicating progress of a process"""

    def __init__(self, console=True):

        if console:
            self.print = self.dyn_print
        else:
            self.print = builtins.print

    @staticmethod
    def dyn_print(data):
        """Print things to stdout on one line, refreshing it dynamically"""
        sys.stdout.write("\r\x1b[K" + data.__str__())
        sys.stdout.flush()


def readable_time(time_difference):
    """Convert a float measuring time difference in seconds into a tuple of (hours, minutes, seconds)"""

    hours = time_difference // 3600
    minutes = (time_difference // 60) % 60
    seconds = time_difference % 60

    return hours, minutes, seconds


# def check_model1(model, verbose=False, stop_on_error=False):
#     status_ok = True
#     for name, param in model.named_parameters():
#         nan_grads = torch.isnan(param.grad)
#         nan_params = torch.isnan(param)
#         if nan_grads.any() or nan_params.any():
#             status_ok = False
#             print("Param {}: {}/{} nan".format(name, torch.sum(nan_params), param.numel()))
#             if verbose:
#                 print(param)
#             print("Grad {}: {}/{} nan".format(name, torch.sum(nan_grads), param.grad.numel()))
#             if verbose:
#                 print(param.grad)
#             if stop_on_error:
#                 ipdb.set_trace()
#     if status_ok:
#         print("Model Check: OK")
#     else:
#         print("Model Check: PROBLEM")


def check_model(model, verbose=False, zero_thresh=1e-8, inf_thresh=1e6, stop_on_error=False):
    status_ok = True
    for name, param in model.named_parameters():
        param_ok = check_tensor(param, verbose=verbose, zero_thresh=zero_thresh, inf_thresh=inf_thresh)
        if not param_ok:
            status_ok = False
            print("Parameter '{}' PROBLEM".format(name))
        grad_ok = True
        if param.grad is not None:
            grad_ok = check_tensor(param.grad, verbose=verbose, zero_thresh=zero_thresh, inf_thresh=inf_thresh)
        if not grad_ok:
            status_ok = False
            print("Gradient of parameter '{}' PROBLEM".format(name))
        if stop_on_error and not (param_ok and grad_ok):
            ipdb.set_trace()

    if status_ok:
        print("Model Check: OK")
    else:
        print("Model Check: PROBLEM")


def check_tensor(X, verbose=True, zero_thresh=1e-8, inf_thresh=1e6):

    is_nan = torch.isnan(X)
    if is_nan.any():
        print("{}/{} nan".format(torch.sum(is_nan), X.numel()))
        return False

    num_small = torch.sum(torch.abs(X) < zero_thresh)
    num_large = torch.sum(torch.abs(X) > inf_thresh)

    if verbose:
        print("Shape: {}, {} elements".format(X.shape, X.numel()))
        print("No 'nan' values")
        print("Min: {}".format(torch.min(X)))
        print("Median: {}".format(torch.median(X)))
        print("Max: {}".format(torch.max(X)))

        print("Histogram of values:")
        values = X.view(-1).detach().numpy()
        hist, binedges = np.histogram(values, bins=20)
        for b in range(len(binedges) - 1):
            print("[{}, {}): {}".format(binedges[b], binedges[b + 1], hist[b]))

        print("{}/{} abs. values < {}".format(num_small, X.numel(), zero_thresh))
        print("{}/{} abs. values > {}".format(num_large, X.numel(), inf_thresh))

    if num_large:
        print("{}/{} abs. values > {}".format(num_large, X.numel(), inf_thresh))
        return False

    return True


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def recursively_hook(model, hook_fn):
    for name, module in model.named_children(): #model._modules.items():
        if len(list(module.children())) > 0:  # if not leaf node
            for submodule in module.children():
                recursively_hook(submodule, hook_fn)
        else:
            module.register_forward_hook(hook_fn)


def compute_loss(net: torch.nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 loss_function: torch.nn.Module,
                 device: torch.device = 'cpu') -> torch.Tensor:
    """Compute the loss of a network on a given dataset.

    Does not compute gradient.

    Parameters
    ----------
    net:
        Network to evaluate.
    dataloader:
        Iterator on the dataset.
    loss_function:
        Loss function to compute.
    device:
        Torch device, or :py:class:`str`.

    Returns
    -------
    Loss as a tensor with no grad.
    """
    running_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            netout = net(x.to(device)).cpu()
            running_loss += loss_function(y, netout)

    return running_loss / len(dataloader)

def Projector(mlp, embedding):
    mlp_spec = f"{embedding}-{mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        # layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)

def normalize(memmap, norm_type):
	"""
	Args:
		memmap: input dataframe
	Returns:
		dmemmapf: normalized dataframe
	"""
	if norm_type == "standardization":
		mean = memmap.mean()
		std = memmap.std()
		return (memmap - mean) / (std + np.finfo(float).eps)
	
	if norm_type == "z_normalization":
		return (memmap - np.mean(memmap, axis=-1, keepdims=True)) / (np.finfo(float).eps + np.std(memmap, axis=-1, keepdims=True))

	elif norm_type == "minmax":
		max_val = np.max(memmap)
		min_val = np.min(memmap)
		return (memmap - min_val) / (max_val - min_val + np.finfo(float).eps)

	elif norm_type == "per_sample_std":
		return (memmap - np.mean(memmap, axis=0)) / np.std(memmap, axis=0)

	elif norm_type == "per_sample_minmax":
		min_vals = np.min(memmap, axis=0)
		max_vals = np.max(memmap, axis=0)
		return (memmap - min_vals) / (max_vals- min_vals + np.finfo(float).eps)

	else:
		raise (NameError(f'Normalize method "{norm_type}" not implemented'))