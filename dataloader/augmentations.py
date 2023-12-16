import numpy as np
import torch
import tsaug 


def DataTransform(sample, config):
    augmentation_list = ['AddNoise(seed=np.random.randint(2 ** 32 - 1))',
                             'Crop(int(0.9 * ts_l), seed=np.random.randint(2 ** 32 - 1))',
                             'Pool(seed=np.random.randint(2 ** 32 - 1))',
                             'Quantize(seed=np.random.randint(2 ** 32 - 1))',
                             'TimeWarp(seed=np.random.randint(2 ** 32 - 1))'
                             ]
    x_q = sample
    aug1 = np.random.choice(augmentation_list, 1, replace=False)
    for aug in aug1:
        x_q = eval('tsaug.' + aug + '.augment(x_q)')
    weak_aug = x_q

    aug2 = np.random.choice(augmentation_list, 1, replace=False)
    while (aug2 == aug1).all():
        aug2 = np.random.choice(augmentation_list, 1, replace=False)
    
    x_k = sample
    for aug in aug2:
        x_k = eval('tsaug.' + aug + '.augment(x_k)')
    strong_aug = x_k

    return weak_aug, strong_aug


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

