import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.autograd import Function

from core.config import cfg
import nn as mynn
import utils.net as net_utils
import numpy as np

DEBUG = False



class GradReverse(Function):
    '''
        Gradient reversal layer

        Based off:
            https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/4
    '''
    def __init__(self, scaler = -0.1):
        self.scaler = scaler

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * self.scaler)

def grad_reverse(x):
    return GradReverse()(x)


class domain_discriminator_im(nn.Module):
    """ Image-level adversarial domain classifier """
    def __init__(self, n_in=1024, grl_scaler=-0.1):
        super(domain_discriminator_im, self).__init__()
        # self.conv1 = nn.Conv2d(n_in, 512, kernel_size=4, stride=2, padding=1)
        # self.conv2 = nn.Conv2d(512, 512, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(n_in, 512, kernel_size=1, stride=1)
        self.classifier = nn.Conv2d(512, 1, kernel_size=1, stride=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.grad_reverse = GradReverse(scaler=grl_scaler)

    def forward(self, x):
        x = grad_reverse(x)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        if DEBUG:
            print(x.size())
        # x = self.conv2(x)
        # x = self.leaky_relu(x)
        # if DEBUG:
        #     print(x.size())
        x = self.classifier(x)
        if DEBUG:
            print(x.size())
        return x

    def detectron_weight_mapping(self):
        # do not load from (or save to) checkpoint
        detectron_weight_mapping = {
            'conv1.weight': None,
            'conv1.bias': None,
            'classifier.weight': None,
            'classifier.bias': None,
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron


def domain_loss_im(pred, domain_label):
    """
    Image-level domain adversarial loss
    
    """
    if DEBUG:
        print('\tDA-image loss')
    device_id = pred.get_device()
    target_label = Variable(
                    torch.FloatTensor(pred.data.size()).fill_(float(domain_label))
                    ).cuda(device_id)
    loss_da_im = F.binary_cross_entropy_with_logits(pred, target_label)

    if net_utils.is_nan_loss(loss_da_im):
        loss_da_im *= 0
    return loss_da_im


class domain_discriminator_roi(nn.Module):
    """ ROI-level adversarial domain classifier """
    def __init__(self, n_in=2048, grl_scaler=-0.1):
        super(domain_discriminator_roi, self).__init__()
        # self.conv1 = nn.Conv2d(n_in, 512, kernel_size=4, stride=2, padding=1)
        # self.conv2 = nn.Conv2d(512, 512, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(n_in, 1024, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.classifier = nn.Conv2d(1024, 1, kernel_size=1, stride=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.grad_reverse = GradReverse(scaler=grl_scaler)

    def forward(self, x):
        x = grad_reverse(x)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        if DEBUG:
            print(x.size())
        x = self.conv2(x)
        x = self.leaky_relu(x)
        if DEBUG:
            print(x.size())
        x = self.classifier(x)
        if DEBUG:
            print(x.size())
        return x

    def detectron_weight_mapping(self):
        # do not load from (or save to) checkpoint
        detectron_weight_mapping = {
            'conv1.weight': None,
            'conv1.bias': None,
            'conv2.weight': None,
            'conv2.bias': None,
            'classifier.weight': None,
            'classifier.bias': None,
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron


def domain_loss_roi(pred, domain_label):
    """
    ROI-level domain adversarial loss
    
    """
    if DEBUG:
        print('\tDA-ROI loss')
    device_id = pred.get_device()
    target_label = Variable(
                    torch.FloatTensor(pred.data.size()).fill_(float(domain_label))
                    ).cuda(device_id)
    loss_da_roi = F.binary_cross_entropy_with_logits(pred, target_label)
    if net_utils.is_nan_loss(loss_da_roi):
        loss_da_roi *= 0
    return loss_da_roi


def domain_loss_cst(im_pred, roi_pred):
    """
    Consistency regularization between image and ROI predictions
    
    """
    if DEBUG:
        print('\tDA-CST loss')
    assert im_pred.get_device() == roi_pred.get_device()
    device_id = im_pred.get_device()
    loss_cst = torch.mean((im_pred.mean() - roi_pred)**2)    
    return loss_cst
