# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import network.blocks as blocks
import torch.nn.functional as F
from warping.shiftandadd import featureAdd, featureWeight


class WeightedSum(nn.Module):
    """ Performs adaptive weighted-sum fusion to merge the input embeddings of the burst images """
    def __init__(self, input_dim, project_dim, offset_feat_dim,local_rank, offset_modulo=None, softmax=True,
                 use_bn=False, activation='relu',):
        super().__init__()
        self.local_rank = local_rank
        self.offset_modulo = offset_modulo
        self.input_dim=input_dim
        self.softmax = softmax

        offset_feat_extractor = []
        offset_feat_extractor.append(blocks.conv_block(2, offset_feat_dim, 3, stride=1, padding=1, batch_norm=use_bn,
                                                       activation=activation))
        self.offset_feat_extractor = nn.Sequential(*offset_feat_extractor)

        weight_predictor = []
        weight_predictor.append(blocks.conv_block(input_dim + offset_feat_dim, project_dim, 3,
                                                  stride=1, padding=1, batch_norm=use_bn, activation=activation))
        weight_predictor.append(blocks.conv_block(project_dim, input_dim, 3, stride=1, padding=1,
                                                  batch_norm=use_bn,
                                                  activation='none'))

        self.weight_predictor = nn.Sequential(*weight_predictor)

    def forward(self, ref_feat, oth_feat, offsets):
        shape = ref_feat.shape
        # Select the base embeddings which is either the embeddings of the reference (first) image, or the mean
        # embedding over all images
        # Compute the residual between the base embeddings and other embeddings
        res_feat = oth_feat - ref_feat
        res_feat = res_feat.view(-1, *ref_feat.shape[-3:])
        offsets = offsets.view(-1, *offsets.shape[-3:])

        # Since we are only interested in sub-pixel sampling location, compute a modulo of the offsets
        if getattr(self, 'offset_modulo', None) is not None:
            offsets = offsets % self.offset_modulo

        offsets_feat = self.offset_feat_extractor(offsets)
        res_feat = torch.cat((res_feat,offsets_feat), dim=1)
        # Compute attention weights
        weights = self.weight_predictor(res_feat)
        weights = weights.view(shape[0], -1, *weights.shape[-3:])

        # Normalize the weights
        if self.softmax:
            weights = F.softmax(weights, dim=1)
        else:
            weights = F.relu(weights)
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

        # Perform fusion
        oth_feat = (oth_feat * weights).sum(dim=1)

        #out = {'fused_enc': fused_feat, 'fusion_weights': weights_norm}
        return oth_feat

class Weighted(nn.Module):
    """ Performs adaptive weighted-sum fusion to merge the input embeddings of the burst images """
    def __init__(self, input_dim, project_dim, local_rank, softmax=True,
                 use_bn=False, activation='relu',):
        super().__init__()
        self.local_rank = local_rank
        self.input_dim=input_dim
        self.softmax = softmax

        self.feat_project_layer = blocks.conv_block(input_dim, project_dim, 1, stride=1, padding=0,
                                                    batch_norm=use_bn,
                                                    activation=activation)
        weight_predictor = []
        weight_predictor.append(blocks.conv_block(project_dim * 2, 2*project_dim, 3,
                                                  stride=1, padding=1, batch_norm=use_bn, activation=activation))

        for _ in range(1):
            weight_predictor.append(blocks.ResBlock(2 * project_dim, 2 * project_dim, stride=1,
                                                    batch_norm=use_bn, activation=activation))

        weight_predictor.append(blocks.conv_block(2 * project_dim, input_dim, 3, stride=1, padding=1,
                                                  batch_norm=use_bn,
                                                  activation='none'))

        self.weight_predictor = nn.Sequential(*weight_predictor)

    def forward(self, ref_feat, oth_feat):
        shape = oth_feat.shape
        # Select the base embeddings which is either the embeddings of the reference (first) image, or the mean
        # embedding over all images
        # Compute the residual between the base embeddings and other embeddings
        #res_feat = oth_feat - ref_feat
        #res_feat = res_feat.view(-1, *ref_feat.shape[-3:])
        oth_feat = oth_feat.view(-1, *shape[-3:])
        ref_feat = ref_feat.view(-1, *shape[-3:])
        oth_feat = self.feat_project_layer(oth_feat)
        ref_feat=self.feat_project_layer(ref_feat)
        ref_feat = (ref_feat.unsqueeze(1).repeat(1, shape[1], 1, 1, 1).view(-1, *oth_feat.shape[-3:]))
        res_feat = oth_feat-ref_feat
        res_feat = torch.cat((ref_feat, res_feat), dim=1)
        # Compute attention weights
        weights = self.weight_predictor(res_feat)
        weights = weights.view(shape[0], -1, *weights.shape[-3:])

        # Normalize the weights
        if self.softmax:
            weights = F.softmax(weights, dim=1)
        else:
            weights = F.relu(weights)
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

        # Perform fusion

        #out = {'fused_enc': fused_feat, 'fusion_weights': weights_norm}
        return weights

class Weighted_conf(nn.Module):
    """ Performs adaptive weighted-sum fusion to merge the input embeddings of the burst images """
    def __init__(self, input_dim, project_dim, local_rank, softmax=True,
                 use_bn=False, activation='relu',):
        super().__init__()
        self.local_rank = local_rank
        self.input_dim=input_dim
        self.softmax = softmax

        conf_feat_extractor = []
        conf_feat_extractor.append(blocks.conv_block(1, project_dim, 3, stride=1, padding=1, batch_norm=use_bn,
                                                       activation=activation))
        conf_feat_extractor.append(blocks.ResBlock(project_dim, project_dim, stride=1,
                                                     batch_norm=use_bn, activation=activation))
        self.offset_feat_extractor = nn.Sequential(*conf_feat_extractor)

        weight_predictor = []
        weight_predictor.append(blocks.conv_block(input_dim+project_dim, project_dim, 3,
                                                  stride=1, padding=1, batch_norm=use_bn, activation=activation))
        weight_predictor.append(blocks.conv_block(project_dim, input_dim, 3, stride=1, padding=1,
                                                  batch_norm=use_bn,
                                                  activation='none'))

        self.weight_predictor = nn.Sequential(*weight_predictor)

    def forward(self, ref_feat, oth_feat,conf):
        shape = ref_feat.shape
        # Select the base embeddings which is either the embeddings of the reference (first) image, or the mean
        # embedding over all images
        # Compute the residual between the base embeddings and other embeddings
        res_feat = oth_feat - ref_feat
        res_feat = res_feat.view(-1, *ref_feat.shape[-3:])
        conf = self.offset_feat_extractor(conf.view(-1, 1, *ref_feat.shape[-2:]))
        res_feat = torch.cat((res_feat, conf), dim=1)

        # Compute attention weights
        weights = self.weight_predictor(res_feat)
        weights = weights.view(shape[0], -1, *weights.shape[-3:])

        # Normalize the weights
        if self.softmax:
            weights = F.softmax(weights, dim=1)
        else:
            weights = F.relu(weights)
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

        # Perform fusion

        #out = {'fused_enc': fused_feat, 'fusion_weights': weights_norm}
        return weights
