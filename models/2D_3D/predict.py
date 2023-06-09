'''
This script predicts results of test dataset (no ground truth). 

TODO(cattalyya): DRY: Merge eval.py and predict.py or extract more common functions.
python predict.py  --data_type=coarse --n_comp=10 --batch_size=128
'''
import sys
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../3D/')))
from submission_utils import write_result
import json

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../loaders/2D_3D/')))
from compat2D_3D import EvalLoader2D_3D

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../2D/segmentation/')))
from segformer import SegFormer2D

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../3D/models/')))
from pointnet2 import PointNet2

from helper_2d3d import *

def parse_args(argv):
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(
        description="Training the Baseline Models on 2D Part and Material Segmentation"
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=60,
        help="Number of epochs to be used for training",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="fine",
        help="Semantic level to be used for training",
    )
    parser.add_argument(
        "--task", type=str, default="part", help="Type of task to be used for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs to be used for training",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="Name of the model file",
    )
    parser.add_argument(
        "--shape_prior", action="store_true", default=True, help="use shape prior"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="debug"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="segformer",
        help="Name of the model file",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size to be used for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate to be used for training",
    )
    parser.add_argument(
        "--n_comp",
        type=int,
        default=1,
        help="Number of components to be used for training",
    )

    parser.add_argument(
        "--npoint",
        type=int,
        default=2048,
        help="Number of point clouds",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="Optimizer to be used for training",
    )
    parser.add_argument(
        "--normal", action="store_true", default=False, help="use normals"
    )
    args = parser.parse_args(argv)

    # Printing input arguments
    print("Input arguments:")
    print(args)

    return args

def main(argv=None):
    # Setup
    args = parse_args(argv)
    root_url_2D = "/home/ubuntu/3dcompat/workspace/3DCoMPaT-v2/download/3DCoMPaT_2D/"
    root_dir_3D = os.path.join(os.getcwd(),"../../models/3D/" "data/" + args.data_type + "_grained/")
    num_workers = 6
    PART_CLASSES = 276 if args.data_type == "fine" else 44
    MAT_CLASSES = 14
    num_labels = PART_CLASSES if args.task == "part" else MAT_CLASSES
    shape_prior = args.shape_prior

    # Load dataset
    test_loader_2D_3D = (
        EvalLoader2D_3D(root_url_2D=root_url_2D,
                        root_dir_3D=root_dir_3D,
                        num_points=2048,
                        semantic_level=args.data_type,
                        n_compositions=args.n_comp)
        .make_loader(batch_size=args.batch_size, num_workers=num_workers)
    )

    # ======== Load 2D model ========
    segformer_pretrain_path = '/home/ubuntu/3dcompat/workspace/3DCoMPaT-v2/models/2D/segmentation/pretrain/coarse_{}_best_model.pth'.format(args.task)
    segformer = SegFormer2D(segformer_pretrain_path, args, num_labels)
    # ======== Load 3D model ========
    ##== 1. Setup model
    part_log_dir = "2023-06-04_05-57"
    cls_log_dir = "2023-06-08_14-31"
    mat_log_dir = ""
    pointnet2 = PointNet2(part_log_dir, cls_log_dir, mat_log_dir, shape_prior, args)

    saved_results = dict()

    saved_part_predictions = dict()
    saved_cls_predictions = dict()
    # ======== Inference ========
    total_seen2d, total_seen3d, total_correct2d, total_correct3d, total_correct2d3d = 0.0, 0.0, 0, 0, 0
    for i, data_tuple in tqdm(enumerate(test_loader_2D_3D)):
        shape_id, image, depth, \
        style_id, view_id, view_type, \
        cam_parameters,\
        points = data_tuple

        # Start compute
        if args.debug:
            print("------>>> ", shape_id, image.shape)
        with torch.no_grad():
            ## Predict shape cls
            predicted_cls = pointnet2.infer_cls(points).cpu()
            # print(predicted_cls, shape_label)
            ## Predict 2D
            outputs, logits, predicted = segformer.infer(image)
            # outputs, logits, predicted = outputs.cpu(), logits.cpu(), predicted.cpu()
            ## Predict 3D
            pointnet2.partmodel = pointnet2.partmodel.eval()
            logits3d, predicted3d = pointnet2.infer_part(points.cpu(), predicted_cls)
            logits3d, predicted3d =  logits3d.cpu(), predicted3d
            
            # if args.debug:
            #     print(logits.shape, predicted.shape, logits3d.shape, predicted3d.shape)
            # THIS
            logits3d_from2d = get_logits_from2d(points.cpu(), logits.cpu(), cam_parameters.cpu())
            zeros_tensor = torch.zeros((logits3d.shape[0], logits3d.shape[1], 1)).cpu()
            # # Concatenate the original tensor with the zeros tensor along the third dimension
            logits3d_extended = torch.cat((zeros_tensor, logits3d), dim=2).cpu()
            fused_logits = logits3d_from2d + logits3d_extended

            saved_results = update_part_logits(saved_results, shape_id, style_id, fused_logits)
            fused_prediction_np = get_fused_prediction(fused_logits, predicted3d)
            # correct_fuse_multi = np.sum(points_part_labels[0] == fused_sample_prediction_np[0])
            saved_cls_predictions, saved_part_predictions = update_predictions(shape_id, style_id, predicted_cls, fused_prediction_np, saved_cls_predictions, saved_part_predictions)

            if i % 50 == 0:
                print(shape_id[0], predicted_cls[0], predicted3d[0], fused_prediction_np[0])
            # break
    np.savez("out_np/cls_test_predictions.npz", saved_cls_predictions)
    np.savez("out_np/parts_test_predictions.npz", saved_part_predictions)
    np.savez("out_np/parts_fused_test_logits.npz", saved_results)
        # break
        # if i > 3:
        #     break
    # print("2D_acc: {}\t3D_acc: {}\t2D_3D_acc: {}".format(total_correct2d/total_seen2d, total_correct3d/total_seen3d, -1))

if __name__ == "__main__":
    main()
