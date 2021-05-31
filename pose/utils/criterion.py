#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pdb
import torch.nn as nn
# import encoding.nn as nn
import math
import os
import sys
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable

class Cos_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, activation):
        super(Cos_Attn, self).__init__()
        # self.chanel_in = in_dim
        self.activation = activation
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        m_batchsize, C, width, height = x.size()
        proj_query = x.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = x.view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        q_norm = proj_query.norm(2, dim=2)
        nm = torch.bmm(q_norm.view(m_batchsize, width * height, 1), q_norm.view(m_batchsize, 1, width * height))
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        norm_energy = energy / (nm+1e-9)
        attention = self.softmax(norm_energy)  # BX (N) X (N)
        return attention

class Cos_Attn_sig(nn.Module):
    """ Self attention Layer"""

    def __init__(self, activation):
        super(Cos_Attn_sig, self).__init__()
        # self.chanel_in = in_dim
        self.activation = activation
        self.softmax = nn.Sigmoid()  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        m_batchsize, C, width, height = x.size()
        proj_query = x.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = x.view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        q_norm = proj_query.norm(2, dim=2)
        nm = torch.bmm(q_norm.view(m_batchsize, width * height, 1), q_norm.view(m_batchsize, 1, width * height))
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        norm_energy = energy / (nm+1e-9)
        attention = self.softmax(norm_energy)  # BX (N) X (N)
        return attention

class Cos_Attn_no(nn.Module):
    """ Self attention Layer"""

    def __init__(self, activation):
        super(Cos_Attn_no, self).__init__()
        # self.chanel_in = in_dim
        self.activation = activation
        self.softmax = nn.Sigmoid()  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        m_batchsize, C, width, height = x.size()
        proj_query = x.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = x.view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        q_norm = proj_query.norm(2, dim=2)
        nm = torch.bmm(q_norm.view(m_batchsize, width * height, 1), q_norm.view(m_batchsize, 1, width * height))
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        norm_energy = energy / (nm+1e-9)
        # attention = self.softmax(norm_energy)  # BX (N) X (N)
        return norm_energy

class CriterionSDcos(nn.Module):
    '''
    structure distillation loss based on graph
    '''

    def __init__(self, ignore_index=255, use_weight=True, pp=1, sp=1):
        super(CriterionSDcos, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        self.soft_p = sp
        self.pred_p = pp
        self.attn = Cos_Attn('relu')
        # self.attn = Cos_Attn_no('relu')
        # self.attn2 = Cos_Attn(320, 'relu')
        # self.criterion = torch.nn.NLLLoss(ignore_index=ignore_index)
        # # self.criterion_cls = torch.nn.BCEWithLogitsLoss()
        self.criterion_sd = torch.nn.MSELoss()

    def forward(self, preds, soft):
        # h, w = labels.size(1), labels.size(2)
        graph_s = self.attn(preds)
        graph_t = self.attn(soft)
        loss_graph = self.criterion_sd(graph_s, graph_t)

        return loss_graph

class CriterionSDcos_no(nn.Module):
    '''
    structure distillation loss based on graph
    '''

    def __init__(self, ignore_index=255, use_weight=True):
        super(CriterionSDcos_no, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight

        self.attn = Cos_Attn_no('relu')
        # self.attn2 = Cos_Attn(320, 'relu')
        # self.criterion = torch.nn.NLLLoss(ignore_index=ignore_index)
        # # self.criterion_cls = torch.nn.BCEWithLogitsLoss()
        self.criterion_sd = torch.nn.MSELoss()

    def forward(self, preds, soft):
        # h, w = labels.size(1), labels.size(2)
        graph_s = self.attn(preds)
        graph_t = self.attn(soft)
        loss_graph = self.criterion_sd(graph_s, graph_t)

        return loss_graph

class CriterionSDcos_sig(nn.Module):
    '''
    structure distillation loss based on graph
    '''

    def __init__(self, ignore_index=255, use_weight=True):
        super(CriterionSDcos_sig, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight

        self.attn = Cos_Attn_sig('relu')
        # self.attn2 = Cos_Attn(320, 'relu')
        # self.criterion = torch.nn.NLLLoss(ignore_index=ignore_index)
        # # self.criterion_cls = torch.nn.BCEWithLogitsLoss()
        self.criterion_sd = torch.nn.MSELoss()

    def forward(self, preds, soft):
        # h, w = labels.size(1), labels.size(2)
        graph_s = self.attn(preds)
        # print(preds.max())
        graph_t = self.attn(soft)
        loss_graph = self.criterion_sd(graph_s, graph_t)

        return loss_graph