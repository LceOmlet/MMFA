from ..registry import TRANSFORMATION
import os
import torch
import numpy as np
from tqdm import tqdm
import pywt
from torch.nn import functional as F
from .ts_transformation import VoidTransfromation
from functools import partial

@TRANSFORMATION.register("wavelet")
class WaveLet(VoidTransfromation):
    def __init__(self, X, d_name, resize_shape=224, dset_type="train", wavelet="db3", logger=None, **kwargs):
        os.makedirs("augmentation", exist_ok=True)
        save_path = f"augmentation/wave_{d_name}_{resize_shape}_{wavelet}_{dset_type}.pt"
        coeffs = []
        if os.path.exists(save_path):
            self.coeffs = torch.load(save_path)
            return None
        train_size, channel, n = X.shape
        scale = resize_shape // 2
        scales = np.arange(1, scale + 1, 1)
        # X = X.reshape((train_size * channel, n))
        if logger is None:
            print("wavelet trainsform to: " + save_path)
        else:
            logger.info("wavelet trainsform to: " + save_path)
        for d in tqdm(X):
            d_ = []
            for dd in d:
                coef, freqs = pywt.cwt(dd, scales, wavelet)
                coef = F.interpolate(torch.abs(torch.tensor(coef)).unsqueeze(0).unsqueeze(0), size=(resize_shape, resize_shape), 
                            mode='bilinear', align_corners=False).squeeze()
                d_.append(coef)
            coef = torch.stack(d_)
            desired_channels = 64
            # print(coef.shape)
            if channel > desired_channels:
                coef = coef.view(1, coef.shape[0], resize_shape * resize_shape)
                coef = F.adaptive_avg_pool2d(coef, (desired_channels, resize_shape * resize_shape))
                coef = coef.view(desired_channels, resize_shape, resize_shape)
            coeffs.append(coef)
        print("stacking")
        coeffs = torch.stack(coeffs)
        print("finally stacked")
        # coeffs = coeffs.view((train_size, channel, resize_shape , resize_shape))
        # coeffs = torch.tensor(coeffs)
        print("saving")
        torch.save(coeffs, save_path)
        print("final saved")
        self.coeffs = coeffs
    
    def transform(self, x, index, **kwargs):
        return self.coeffs[index].unsqueeze(0).float()



# for i in range(10):
#     wavelet_name = "db" + str(i)
#     TRANSFORMATION.register(wavelet_name)(partial(WaveLet, kwargs={"wavelet": wavelet_name}))

for i in pywt.wavelist(kind='continuous'):
    TRANSFORMATION.register(i)(partial(WaveLet, wavelet=i))