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
    def __init__(self, pretrain_path, args, num_labels):
        self.args = args

        # Create an instance of SegformerForSemanticSegmentation
        self.config = SegformerConfig()
        self.config.num_labels = num_labels
        self.model = SegformerForSemanticSegmentation(self.config)

        self.model = nn.DataParallel(self.model)

        self.model.load_state_dict(torch.load(pretrain_path))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def infer(self, images):
        outputs = self.model(images.to(self.device))["logits"]
        # Upsample the logits
        # print(images.shape[-2:])
        upsampled_logits = nn.functional.interpolate(
            outputs, size=images.shape[-2:], mode="bicubic", align_corners=False
        )
        predicted = (
            torch.softmax(upsampled_logits, dim=1).argmax(dim=1).cpu().numpy()
        )
        predicted -= np.ones(predicted.shape, dtype=int)
        return outputs, upsampled_logits, predicted


        