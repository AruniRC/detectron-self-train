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

# def grad_reverse(x):
#     return GradReverse()(x)


class domain_discriminator_im(nn.Module):
    """ Image-level adversarial domain classifier """
    def __init__(self, n_in=1024, grl_scaler=-0.1):
        super(domain_discriminator_im, self).__init__()
        # self.conv1 = nn.Conv2d(n_in, 512, kernel_size=4, stride=2, padding=1)
        # self.conv2 = nn.Conv2d(512, 512, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(n_in, 512, kernel_size=1, stride=1)
        self.classifier = nn.Conv2d(512, 1, kernel_size=1, stride=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.grad_reverse = GradReverse(scaler=grl_scaler)

    def forward(self, x):
        x = self.grad_reverse(x)
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
        self.grad_reverse = GradReverse(scaler=grl_scaler)

    def forward(self, x):
        x = self.grad_reverse(x)
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
    assert not net_utils.is_nan_loss(loss_da_im)
    return loss_da_im


def domain_loss_roi(pred, domain_label):
    """
    ROI-level domain adversarial loss
    
    """
    if DEBUG:
        print('\tDA-ROI loss')
    device_id = pred.get_device()
    import pdb; pdb.set_trace()  # breakpoint ef686727 //

    target_label = Variable(
                    torch.FloatTensor(pred.data.size()).fill_(float(domain_label))
                    ).cuda(device_id)
    loss_da_roi = F.binary_cross_entropy_with_logits(pred, target_label)
    assert not net_utils.is_nan_loss(loss_da_roi)
    return loss_da_roi