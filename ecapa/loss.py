'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
'''

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from preprocessing.ecapa.tools import *

class AAMsoftmax(nn.Module):
    def __init__(self, n_class, m, s):
        
        super(AAMsoftmax, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 192), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        
        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return loss, prec1
    
class CosineMarginLoss(nn.Module):
    def __init__(self, embed_dim, num_classes, m=0.35, s=64):
        super(CosineMarginLoss, self).__init__()
        self.w = nn.Parameter(torch.randn(embed_dim, num_classes), requires_grad=True)
        self.num_classes = num_classes
        self.m = m
        self.s = s

    def forward(self, x, labels):
        x_norm = x / torch.norm(x, dim=1, keepdim=True)
        w_norm = self.w / torch.norm(self.w, dim=0, keepdim=True)
        xw_norm = torch.matmul(x_norm, w_norm)

        label_one_hot = F.one_hot(labels.view(-1), self.num_classes).float() * self.m
        value = self.s * (xw_norm - label_one_hot)

        loss = F.cross_entropy(input=value, target=labels.view(-1))
        prec1 = accuracy(value.detach(), labels.view(-1).detach(), topk=(1,))[0]
        return loss, prec1