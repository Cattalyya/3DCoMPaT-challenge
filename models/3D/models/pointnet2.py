import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import numpy as np
import pdb
import torch.nn.functional as F
import json
import yaml
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))
from compat_utils import compute_overall_iou, to_categorical

class PointNet2:
    def __init__(self, cls_log_dir, part_log_dir, mat_log_dir, shape_prior, args):
        ##== 1. Setup
        self.shape_prior = shape_prior
        if args.data_type == "coarse":
            metadata = json.load(open("/home/ubuntu/3dcompat/workspace/3DCoMPaT-v2/models/3D/metadata/coarse_seg_classes.json"))
        else:
            metadata = json.load(open("/home/ubuntu/3dcompat/workspace/3DCoMPaT-v2/models/3D/metadata/fine_seg_classes.json"))

        metadata_mat = json.load(open("/home/ubuntu/3dcompat/workspace/3DCoMPaT-v2/models/3D/metadata/coarse_mat_seg_classes.json"))
        self.num_classes = metadata["num_classes"]
        self.num_part = metadata["num_part"]
        self.num_mat = metadata_mat["num_mat"]
        self.seg_classes = metadata["seg_classes"]
        print("nclasses=", self.num_classes)

        ##======= Load Models =======
        experiment_cls_dir = "/home/ubuntu/3dcompat/workspace/3DCoMPaT-v2/models/3D/log/classification_mn40/" + cls_log_dir
        experiment_parts_dir = "/home/ubuntu/3dcompat/workspace/3DCoMPaT-v2/models/3D/log/part_seg/" + part_log_dir
        experiment_mat_dir = "/home/ubuntu/3dcompat/workspace/3DCoMPaT-v2/models/3D/log/mat_seg/" + mat_log_dir
        
        ##== 2. Load cls model 
        if cls_log_dir != "":
            model_name = os.listdir(experiment_cls_dir + "/logs")[0].split(".")[0]
            MODEL = importlib.import_module(model_name)
            self.clsmodel = MODEL.get_model(
                self.num_classes, normal_channel=args.normal
            ).cuda()
            cls_ckpt = torch.load(str(experiment_cls_dir) + "/checkpoints/best_model.pth")
            self.clsmodel.load_state_dict(cls_ckpt["model_state_dict"])

        ##== 3. Load parts seg model
        if part_log_dir != "":
            model_name = os.listdir(experiment_parts_dir + "/logs")[0].split(".")[0]
            MODEL = importlib.import_module(model_name)
            self.partmodel = MODEL.get_model(
                self.num_part, shape_prior=shape_prior, normal_channel=args.normal, seg_mode="part"
            ).cuda()
            part_ckpt = torch.load(str(experiment_parts_dir) + "/checkpoints/best_model.pth")
            self.partmodel.load_state_dict(part_ckpt["model_state_dict"])

        ##== 3. Load mat seg model
        if mat_log_dir != "":
            model_name = os.listdir(experiment_mat_dir + "/logs")[0].split(".")[0]
            MODEL = importlib.import_module(model_name)
            self.matmodel = MODEL.get_model(
                self.num_mat, shape_prior=shape_prior, normal_channel=args.normal, seg_mode="mat"
            ).cuda()
            mat_ckpt = torch.load(str(experiment_mat_dir) + "/checkpoints/best_model.pth")
            self.matmodel.load_state_dict(mat_ckpt["model_state_dict"])

    def infer_part(self, points, label):
        points, label = (
            points.float().cuda(),
            label.cuda()
        )
        points = points.transpose(2, 1)
        # print(label, label.shape, to_categorical(label, self.num_classes))
        if self.shape_prior:
            seg_pred, _ = self.partmodel(points, to_categorical(label, self.num_classes))
        else:
            seg_pred, _ = self.partmodel(points)
        cur_pred_val = seg_pred.cpu().data.numpy()
        cur_pred_val = np.argmax(cur_pred_val, -1)
        return seg_pred, cur_pred_val


    def infer_mat(self, points, label):
        points, label = (
            points.float().cuda(),
            label.cuda()
        )
        points = points.transpose(2, 1)
        # print(label, label.shape, to_categorical(label, self.num_classes))
        if self.shape_prior:
            seg_pred, _ = self.matmodel(points, to_categorical(label, self.num_classes))
        else:
            seg_pred, _ = self.matmodel(points)
        cur_pred_val = seg_pred.cpu().data.numpy()
        cur_pred_val = np.argmax(cur_pred_val, -1)
        return seg_pred, cur_pred_val

    def infer_cls(self, points):
        points = points.cuda()
        points = points.transpose(2, 1)
        pred, _ = self.clsmodel(points)
        pred_choice = pred.data.max(1)[1]
        return pred_choice

