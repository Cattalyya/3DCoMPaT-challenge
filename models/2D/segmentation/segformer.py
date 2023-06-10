"""
Segformer finetuning on 2D segmentation.
CUDA_VISIBLE_DEVICES=0 python test.py  --n_comp=10 --data_type=coarse --split='test' --batch_size=128
"""
import argparse
import os
import time
import sys

import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../loaders/2D')))
from compat2D import GCRLoader, EvalLoader, wds_identity
from custom_metrics import compute_overall_iou, compute_overall_precision
from tqdm import tqdm
from transformers import SegformerConfig, SegformerForSemanticSegmentation


class SegFormer2D:
    def __init__(self, pretrain_part_path, pretrain_mat_path, args, num_part_labels, num_mat_labels):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create an instance of SegformerForSemanticSegmentation
        ## 1. Load pretrained part model
        self.config_part = SegformerConfig()
        self.config_part.num_labels = num_part_labels
        self.partmodel = SegformerForSemanticSegmentation(self.config_part)
        self.partmodel = nn.DataParallel(self.partmodel)
        self.partmodel.load_state_dict(torch.load(pretrain_part_path))
        self.partmodel.to(self.device)

        ## 2. Load pretrained mat model
        self.config_mat = SegformerConfig()
        self.config_mat.num_labels = num_mat_labels
        self.matmodel = SegformerForSemanticSegmentation(self.config_mat)
        self.matmodel = nn.DataParallel(self.matmodel)
        self.matmodel.load_state_dict(torch.load(pretrain_mat_path))
        self.matmodel.to(self.device)
        

    def infer_part(self, images):
        outputs = self.partmodel(images.to(self.device))["logits"]
        # Upsample the logits
        # print(images.shape[-2:])
        upsampled_logits = nn.functional.interpolate(
            outputs, size=images.shape[-2:], mode="bicubic", align_corners=False
        )
        predicted = (
            torch.softmax(upsampled_logits, dim=1).argmax(dim=1).cpu().numpy()
        )
        return outputs, upsampled_logits, predicted

    def infer_mat(self, images):
        outputs = self.matmodel(images.to(self.device))["logits"]
        # Upsample the logits
        # print(images.shape[-2:])
        upsampled_logits = nn.functional.interpolate(
            outputs, size=images.shape[-2:], mode="bicubic", align_corners=False
        )
        predicted = (
            torch.softmax(upsampled_logits, dim=1).argmax(dim=1).cpu().numpy()
        )
        return outputs, upsampled_logits, predicted


        