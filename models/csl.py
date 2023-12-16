import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from collections import OrderedDict
import sys
sys.path.append('/home/user1/liangchen/lora/TS-TCC_l/')
from dataloader.dataloader import get_data
import torch.nn.functional as F

from .utils import generate_binomial_mask
try:
    from .patchTST import Projector
except:
    from patchTST import Projector

from .layers.basics import get_activation_fn

from .layers.attention import *

from .layers.tcn import TemporalConvNet
from typing import Tuple

from .layers.rescnn import ResCNN

class Lambda(nn.Module):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
    
class DotProductAttention(nn.Module):
    """
    Compute the dot products of the query with all values and apply a softmax function to obtain the weights on the values
    """
    def __init__(self, output_dim, hidden_dim, act=None):
        super(DotProductAttention, self).__init__()
        self.q_linear = nn.Linear(output_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        if act is None:
            self.act = lambda x: x
        else:
            self.act = get_activation_fn(act)
        self.out_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, query: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        query = self.q_linear(query)
        query = self.act(query)

        value = self.v_linear(value)
        value = self.act(value)

        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(query, value.transpose(1, 2))
        attn = F.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(attn, value)

        context = self.act(context)
        context = self.out_linear(context)

        return context, attn
    
class MinEuclideanDistBlock(nn.Module):
  
    def __init__(self, shapelets_size, num_shapelets, in_channels=3, to_cuda=False, n_head=4):
        super(MinEuclideanDistBlock, self).__init__()
        self.to_cuda = to_cuda
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.in_channels = in_channels

        self.hidden_dim = 128

        # if not registered as parameter, the optimizer will not be able to see the parameters
        shapelets = torch.randn(self.in_channels, self.num_shapelets, self.shapelets_size, requires_grad=True,
                               dtype=torch.float)
        if self.to_cuda:
            shapelets = shapelets.cuda()
        self.shapelets = nn.Parameter(shapelets).contiguous()
        # otherwise gradients will not be backpropagated
        self.shapelets.retain_grad()

        self.shaplet_agg = nn.Sequential(
            Lambda(lambda x: x.reshape((-1, ) + tuple(x.shape[-2:]))),
            Lambda(lambda x: x.permute(0, 2, 1)),
            # nn.MaxPool1d(kernel_size=4, stride=4),
            get_activation_fn("gelu"),
            # TemporalConvNet(self.num_shapelets, [self.hidden_dim] * 4 + [self.num_shapelets] + [self.num_shapelets], kernel_size=3),
            ResCNN(self.num_shapelets, self.num_shapelets),
            # Lambda(lambda x: x.reshape())
            # Lambda(lambda x: x.permute(0, 2, 1))
        )

        
        self.channel_attn = nn.Sequential(
            get_activation_fn("gelu"),
            MultiheadAttention(self.num_shapelets, 2, self.num_shapelets // 2, self.num_shapelets // 2, sequencial=True)
        )

        self.temporal_attn = DotProductAttention(self.num_shapelets, self.hidden_dim, "gelu")

    def forward(self, x, masking=False):
       
        
        
        
        # unfold time series to emulate sliding window
        # print(x.shape)
        # exit()
        pad_size = (self.shapelets_size -1)// 2
        x = F.pad(x, (pad_size, pad_size + (self.shapelets_size -1) % 2))
        x = x.unfold(2, self.shapelets_size, 1).contiguous()
        # exit()
        
        # calculate euclidean distance
        x = torch.cdist(x, self.shapelets, p=2, compute_mode='donot_use_mm_for_euclid_dist')
        
        #x = torch.cdist(x, self.shapelets, p=2)
        
        # add up the distances of the channels in case of
        # multivariate time series
        # Corresponds to the approach 1 and 3 here: https://stats.stackexchange.com/questions/184977/multivariate-time-series-euclidean-distance
        x_shape = list(x.shape)
        x_shape[-2] = -1
        # print(x.shape)
        x_ = x
        x = self.shaplet_agg(x)
        x = x.reshape(x_shape)
        
        x_, _ = torch.min(x_, 2)

        # query = x_.reshape(x.shape[0], -1, self.num_shapelets)
        x = torch.mean(x, dim=-2)
        # print(x.shape)
        # exit()
        x = x.reshape(x_.shape)

        x = x + x_
        x_ = x
        x_shape = x.shape

        # print(x_shape)
        # exit()
        
        x = torch.mean(x, dim=1, keepdim=True)
        
        x_ = self.channel_attn(x_)
        x_ = x_.reshape(x_shape)
        x_ = torch.mean(x_, dim = 1, keepdim=True)

        x = x_
        
        """
        n_dims = x.shape[1]
        out = torch.zeros((x.shape[0],
                           1,
                           x.shape[2] - self.shapelets_size + 1,
                           self.num_shapelets),
                        dtype=torch.float)
        if self.to_cuda:
            out = out.cuda()
        for i_dim in range(n_dims):
            x_dim = x[:, i_dim : i_dim + 1, :]
            x_dim = x_dim.unfold(2, self.shapelets_size, 1).contiguous()
            out += torch.cdist(x_dim, self.shapelets[i_dim : i_dim + 1, :, :], p=2, compute_mode='donot_use_mm_for_euclid_dist')
        x = out
        x = x.transpose(2, 3)
        """
        
        # hard min compared to soft-min from the paper
        
        return x

  
   

        
class MaxCosineSimilarityBlock(nn.Module):
   
    def __init__(self, shapelets_size, num_shapelets, in_channels=3, to_cuda=False, n_head=4):
        super(MaxCosineSimilarityBlock, self).__init__()
        self.to_cuda = to_cuda
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.in_channels = in_channels

        self.hidden_dim = 128

        self.relu = nn.ReLU()

        # if not registered as parameter, the optimizer will not be able to see the parameters
        shapelets = torch.randn(self.in_channels, self.num_shapelets, self.shapelets_size, requires_grad=True,
                                dtype=torch.float)
        if self.to_cuda:
            shapelets = shapelets.cuda()
        self.shapelets = nn.Parameter(shapelets).contiguous()
        # otherwise gradients will not be backpropagated
        self.shapelets.retain_grad()

        self.shaplet_agg = nn.Sequential(
            Lambda(lambda x: x.reshape((-1, ) + tuple(x.shape[-2:]))),
            Lambda(lambda x: x.permute(0, 2, 1)),
            # nn.MaxPool1d(kernel_size=4, stride=4),
            get_activation_fn("gelu"),
            ResCNN(self.num_shapelets, self.num_shapelets),
            # Lambda(lambda x: x.permute(0, 2, 1))
        )

        self.channel_attn = nn.Sequential(
            get_activation_fn("gelu"),
            MultiheadAttention(self.num_shapelets, 2, self.num_shapelets // 2, self.num_shapelets // 2, sequencial=True)
        )

        self.temporal_attn = DotProductAttention(self.num_shapelets, self.hidden_dim, "gelu")

    def forward(self, x, masking=False):
     
        """
        n_dims = x.shape[1]
        shapelets_norm = self.shapelets / self.shapelets.norm(p=2, dim=2, keepdim=True).clamp(min=1e-8)
        shapelets_norm = shapelets_norm.transpose(1, 2).half()
        out = torch.zeros((x.shape[0],
                           1,
                           x.shape[2] - self.shapelets_size + 1,
                           self.num_shapelets),
                        dtype=torch.float)
        if self.to_cuda:
            out = out.cuda()
        for i_dim in range(n_dims):
            x_dim = x[:, i_dim : i_dim + 1, :].half()
            x_dim = x_dim.unfold(2, self.shapelets_size, 1).contiguous()
            x_dim = x_dim / x_dim.norm(p=2, dim=3, keepdim=True).clamp(min=1e-8)
            out += torch.matmul(x_dim, shapelets_norm[i_dim : i_dim + 1, :, :]).float()
        
        x = out.transpose(2, 3) / n_dims
        """
        
        pad_size = (self.shapelets_size -1)// 2
        x = F.pad(x, (pad_size, pad_size + (self.shapelets_size -1) % 2))
        # unfold time series to emulate sliding window
        x = x.unfold(2, self.shapelets_size, 1).contiguous()
        
       
        # normalize with l2 norm
        x = x / x.norm(p=2, dim=3, keepdim=True).clamp(min=1e-8)
        
        shapelets_norm = (self.shapelets / self.shapelets.norm(p=2, dim=2, keepdim=True).clamp(min=1e-8))
        # calculate cosine similarity via dot product on already normalized ts and shapelets
        x = torch.matmul(x, shapelets_norm.transpose(1, 2))
       
        
        # add up the distances of the channels in case of
        # multivariate time series
        # Corresponds to the approach 1 and 3 here: https://stats.stackexchange.com/questions/184977/multivariate-time-series-euclidean-distance
        n_dims = x.shape[1]
        x = self.relu(x)
        x_ = x
        x_shape = list(x.shape)
        x_shape[-2] = -1
        x = self.shaplet_agg(x)
        x = x.reshape(x_shape)
        x_, _ = torch.min(x_, 2)
        
        # query = x_.reshape(x.shape[0], -1, self.num_shapelets)
        # x, _ = self.temporal_attn(query, x)

        x = torch.mean(x,-2)

        x = x.reshape(x_.shape)

        x = x + x_

        x_ = x
        x_shape = x.shape
        
        x = torch.mean(x, dim=1, keepdim=True)
        
        x_ = self.channel_attn(x_)
        x_ = x_.reshape(x_shape)
        x_ = torch.mean(x_, dim = 1, keepdim=True)

        x = x_
        
        
        # ignore negative distances
        return x

   
        

class MaxCrossCorrelationBlock(nn.Module):
   
    def __init__(self, shapelets_size, num_shapelets, in_channels=1, to_cuda=True, n_head=4):
        super(MaxCrossCorrelationBlock, self).__init__()
        self.shapelets = nn.Conv1d(in_channels, num_shapelets, kernel_size=shapelets_size)
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.to_cuda = to_cuda
        self.in_channels = in_channels
        self.hidden_dim = 128

        self.shaplet_agg = nn.Sequential(
            Lambda(lambda x: x.reshape((-1, ) + tuple(x.shape[-2:]))),
            Lambda(lambda x: x.permute(0, 2, 1)),
            # nn.MaxPool1d(kernel_size=4, stride=4),
            get_activation_fn("gelu"),
            # TemporalConvNet(self.num_shapelets, [self.hidden_dim] * 4 + [self.num_shapelets], kernel_size=3),
            ResCNN(self.num_shapelets, self.num_shapelets),
            # Lambda(lambda x: x.permute(0, 2, 1))
        )

        self.channel_attn = nn.Sequential(
            get_activation_fn("gelu"),
            MultiheadAttention(self.num_shapelets, 2, self.num_shapelets // 2, self.num_shapelets // 2, sequencial=True)
        )

        self.temporal_attn = DotProductAttention(self.num_shapelets, self.hidden_dim, "gelu")

        if self.to_cuda:
            self.cuda()
        
        
        
    def forward(self, x, masking=False):
        if self.in_channels == 1:
            new_x = [] 
            x = x.permute(1,0,2)     
            pad_size = (self.shapelets_size -1)// 2
            # print(x.shape)
            x = F.pad(x, (pad_size, pad_size + (self.shapelets_size -1) % 2))  
            for x_ in x:
                x_ = x_[:, None, :]
                x_ = self.shapelets(x_)
                new_x.append(x_[None])
            x = torch.cat(new_x, dim=0)
            x = x.permute(1, 0, 3, 2)
            # print(x.shape)
            # exit()
            x_ = x
            x_shape = list(x.shape)
            x_shape[-2] = -1
            x = self.shaplet_agg(x)
            x = x.reshape(x_shape)
            x_, _ = torch.max(x_, 2)

            x = torch.mean(x, dim=-2)

            x = x.reshape(x_.shape)
            x = x + x_
            x_ = x
            x_shape = x.shape
            
            x = torch.mean(x, dim=1, keepdim=True)
            
            x_ = self.channel_attn(x_)
            x_ = x_.reshape(x_shape)
            x_ = torch.mean(x_, dim = 1, keepdim=True)

            x = x_
        else:
            x = self.shapelets(x)
            x, _ = torch.max(x, 2, keepdim=True)
        if masking:
            mask = generate_binomial_mask(x.shape)
            x *= mask
        
        return x

    



class ShapeletsDistBlocks(nn.Module):
   
    def __init__(self, shapelets_size_and_len, in_channels=1, dist_measure='euclidean', to_cuda=True, checkpoint=False):
        super(ShapeletsDistBlocks, self).__init__()
        self.checkpoint = checkpoint
        self.to_cuda = to_cuda
        self.shapelets_size_and_len = OrderedDict(sorted(shapelets_size_and_len.items(), key=lambda x: x[0]))
        self.in_channels = in_channels
        self.dist_measure = dist_measure
        if dist_measure == 'euclidean':
            self.blocks = nn.ModuleList(
                [MinEuclideanDistBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                       in_channels=in_channels, to_cuda=self.to_cuda)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        elif dist_measure == 'cross-correlation':
            self.blocks = nn.ModuleList(
                [MaxCrossCorrelationBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                          in_channels=in_channels, to_cuda=self.to_cuda)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        elif dist_measure == 'cosine':
            self.blocks = nn.ModuleList(
                [MaxCosineSimilarityBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                          in_channels=in_channels, to_cuda=self.to_cuda)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        elif dist_measure == 'mix':
            module_list = []
            for shapelets_size, num_shapelets in self.shapelets_size_and_len.items():
                module_list.append(MinEuclideanDistBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets//3,
                                                         in_channels=in_channels, to_cuda=self.to_cuda))
                module_list.append(MaxCosineSimilarityBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets//3,
                                                         in_channels=in_channels, to_cuda=self.to_cuda))
                module_list.append(MaxCrossCorrelationBlock(shapelets_size=shapelets_size,
                                                            num_shapelets=num_shapelets - 2 * num_shapelets//3,
                                                            in_channels=in_channels, to_cuda=self.to_cuda))
            self.blocks = nn.ModuleList(module_list)
        
        else:
            raise ValueError("dist_measure must be either of 'euclidean', 'cross-correlation', 'cosine'")

    def forward(self, x, masking=False):
       
        out = torch.tensor([], dtype=torch.float).cuda() if self.to_cuda else torch.tensor([], dtype=torch.float)
        for block in self.blocks:
            if self.checkpoint and self.dist_measure != 'cross-correlation':
                out = torch.cat((out, checkpoint(block, x, masking)), dim=2)
            
            else:
                out = torch.cat((out, block(x, masking)), dim=2)
            
       

        return out



  

class LearningShapeletsModel(nn.Module):
   
    def __init__(self, in_channels=1, num_classes=2, output_size=320, dist_measure='euclidean',
                 to_cuda=True, checkpoint=False):
        super(LearningShapeletsModel, self).__init__()
        len_ts = 224
        shapelets_size_and_len = {int(i): 200 for i in np.linspace(min(128, max(3, int(0.1 * len_ts))), int(0.8 * len_ts), 8, dtype=int)}

        self.to_cuda = to_cuda
        self.checkpoint = checkpoint
        self.shapelets_size_and_len = shapelets_size_and_len
        self.num_shapelets = sum(shapelets_size_and_len.values())
        self.shapelets_blocks = ShapeletsDistBlocks(in_channels=in_channels,
                                                    shapelets_size_and_len=shapelets_size_and_len,
                                                    dist_measure=dist_measure, to_cuda=to_cuda, checkpoint=checkpoint)
        self.linear = nn.Linear(self.num_shapelets, num_classes)
        
        self.projection = nn.Sequential(nn.BatchNorm1d(num_features=self.num_shapelets),
                                              #nn.Linear(self.model.num_shapelets, 256),
                                              #nn.ReLU(),
                                              #nn.Linear(self.num_shapelets, 128)
                                        )
        
        self.projection2 = nn.Sequential(nn.Linear(self.num_shapelets, 256),
                                              nn.ReLU(),
                                              nn.Linear(256, 128))
        
        self.act = get_activation_fn('gelu')
        
        self.outpt = nn.Linear(1600, output_size)
        
        self.projector = Projector("4096-8192", output_size)

        if num_classes is not None:
            self.logits = nn.Sequential(
                # nn.Linear(self.embed_patch_aggr, num_classes),
                # get_activation_fn(act),
                nn.Linear(output_size, num_classes)
            )
        
        if self.to_cuda:
            self.cuda()

    def forward(self, x, train_mode='acc', masking=False):
       
        x = self.shapelets_blocks(x, masking)
        
        x = torch.squeeze(x, 1)
        
        # test torch.cat
        #x = torch.cat((x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:]), dim=1)
        
        x = self.projection(x)

        # print(x.shape)
        # exit()
        
        feature = self.outpt(x)
        project = self.projector(feature)
        
        if train_mode == "train_vic":
            return feature, project        

        z = feature
        if hasattr(self, "logits"):
            if len(z.shape) ==2:
                z = z.unsqueeze(-1)
            z_pool = F.max_pool1d(z, kernel_size=z.size(2))
            z_pool = z_pool.squeeze(-1)
            # print(z_pool.shape)
            logits = self.logits(z_pool)
            return logits, z
        return z

   
  


class LearningShapeletsModelMixDistances(nn.Module):
   
    def __init__(self, in_channels=1, num_classes=None, dist_measure='mix',
                 to_cuda=True, checkpoint=False, output_size=320):
        super(LearningShapeletsModelMixDistances, self).__init__()
        len_ts = 224
        shapelets_size_and_len = {int(i): 200 for i in np.linspace(min(128, max(3, int(0.1 * len_ts))), int(0.8 * len_ts), 8, dtype=int)}

        self.checkpoint = checkpoint
        self.to_cuda = to_cuda
        self.shapelets_size_and_len = shapelets_size_and_len
        self.num_shapelets = sum(shapelets_size_and_len.values())
        
        self.shapelets_euclidean = ShapeletsDistBlocks(in_channels=in_channels,
                                                    shapelets_size_and_len={item[0]: item[1] // 3 for item in shapelets_size_and_len.items()},
                                                    dist_measure='euclidean', to_cuda=to_cuda, checkpoint=checkpoint)
        
        
        self.shapelets_cosine = ShapeletsDistBlocks(in_channels=in_channels,
                                                    shapelets_size_and_len={item[0]: item[1] // 3 for item in shapelets_size_and_len.items()},
                                                    dist_measure='cosine', to_cuda=to_cuda, checkpoint=checkpoint)
        
        self.shapelets_cross_correlation = ShapeletsDistBlocks(in_channels=in_channels,
                                                    shapelets_size_and_len={item[0]: item[1] - 2 * (item[1] // 3) for item in shapelets_size_and_len.items()},
                                                    dist_measure='cross-correlation', to_cuda=to_cuda, checkpoint=checkpoint)
        
        
        # self.linear = nn.Linear(self.num_shapelets, num_classes)
        
        self.projection = nn.Sequential(nn.BatchNorm1d(num_features=self.num_shapelets),
                                              #nn.Linear(self.model.num_shapelets, 256),
                                              #nn.ReLU(),
                                              #nn.Linear(self.num_shapelets, 128)
                                        )
        
        self.bn1 = nn.BatchNorm1d(num_features=sum(num // 3 for num in self.shapelets_size_and_len.values()))
        self.bn2 = nn.BatchNorm1d(num_features=sum(num // 3 for num in self.shapelets_size_and_len.values()))
        self.bn3 = nn.BatchNorm1d(num_features=sum(num - 2 * (num // 3) for num in self.shapelets_size_and_len.values()))
        
        self.projection2 = nn.Sequential(nn.Linear(self.num_shapelets, 256),
                                              nn.ReLU(),
                                              nn.Linear(256, 128))
        
        self.act = get_activation_fn('gelu')
        
        self.outpt = nn.Linear(1600, output_size)
        
        self.projector = Projector("4096-8192", output_size)

        if num_classes is not None:
            self.logits = nn.Sequential(
                # nn.Linear(self.embed_patch_aggr, num_classes),
                # get_activation_fn(act),
                nn.Linear(output_size, num_classes)
            )
        
        if self.to_cuda:
            self.cuda()

    def forward(self, x, train_mode="", masking=False):
       

        
        n_samples = x.shape[0]
        num_lengths = len(self.shapelets_size_and_len)
        
        out = torch.tensor([], dtype=torch.float).cuda() if self.to_cuda else torch.tensor([], dtype=torch.float)
        
        x_out = self.shapelets_euclidean(x, masking)
        x_out = torch.squeeze(x_out, 1)
        #x_out = torch.nn.functional.normalize(x_out, dim=1)
        x_out = self.bn1(x_out)
        x_out = x_out.reshape(n_samples, num_lengths, -1)
        #print(x_out.shape)
        out = torch.cat((out, x_out), dim=2)
        
        x_out = self.shapelets_cosine(x, masking)
        x_out = torch.squeeze(x_out, 1)
        #x_out = torch.nn.functional.normalize(x_out, dim=1)
        x_out = self.bn2(x_out)
        x_out = x_out.reshape(n_samples, num_lengths, -1)
        #print(x_out.shape)
        out = torch.cat((out, x_out), dim=2)
        
        x_out = self.shapelets_cross_correlation(x, masking)
        x_out = torch.squeeze(x_out, 1)
        #x_out = torch.nn.functional.normalize(x_out, dim=1)
        x_out = self.bn3(x_out)
        x_out = x_out.reshape(n_samples, num_lengths, -1)
        #print(x_out.shape)
        out = torch.cat((out, x_out), dim=2)
        
        
        out = out.reshape(n_samples, -1)

        out = self.act(out)

        feature = self.outpt(out)
        project = self.projector(feature)
        
        
        #print(out.shape)
        #out = self.projection(out)
        
        if train_mode == "train_vic":
            return feature, project        

        z = feature
        if hasattr(self, "logits"):
            if len(z.shape) ==2:
                z = z.unsqueeze(-1)
            z_pool = F.max_pool1d(z, kernel_size=z.size(2))
            z_pool = z_pool.squeeze(-1)
            # print(z_pool.shape)
            logits = self.logits(z_pool)
            return logits, z
        return z

if __name__ == "__main__":
    dataset = get_data("Handwriting", "train")
    len_ts = 224
    samples = dataset["samples"]
    sample = samples[:10].to(torch.float)
    sample = torch.nn.functional.interpolate(sample, size=(1000, ))
    print(sample.shape)
    SMMD = LearningShapeletsModelMixDistances()
    out = SMMD(sample)
    print(out[0].shape)