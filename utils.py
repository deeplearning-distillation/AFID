import os
import shutil
import numpy as np

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
import numpy as np



def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path,'model_best.pth.tar'))

def load_checkpoint(model, checkpoint):
    m_keys = list(model.state_dict().keys())

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        c_keys = list(checkpoint['state_dict'].keys())
        not_m_keys = [i for i in c_keys if i not in m_keys]
        not_c_keys = [i for i in m_keys if i not in c_keys]
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    else:
        c_keys = list(checkpoint.keys())
        not_m_keys = [i for i in c_keys if i not in m_keys]
        not_c_keys = [i for i in m_keys if i not in c_keys]
        model.load_state_dict(checkpoint, strict=False)

    print("--------------------------------------\n LOADING PRETRAINING \n")
    print("Not in Model: ")
    print(not_m_keys)
    print("Not in Checkpoint")
    print(not_c_keys)
    print('\n\n')


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=3

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = Variable(target_data.data.cuda(),requires_grad=False)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
