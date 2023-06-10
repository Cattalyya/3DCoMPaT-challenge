'''
This script evaluates how well our predictions for cls, part seg and mat seg does for validation ds (with ground truth).

TODO(cattalyya): DRY: Merge eval.py and predict.py or extract more common functions.
python eval.py  --data_type=coarse --n_comp=10 --batch_size=128
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
from submission_utils import write_result, Submission
import json

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../loaders/2D_3D/')))
from compat2D_3D import FullLoader2D_3D

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
    parser.add_argument(
        "--nofuse", action="store_true", default=False, help="use normals"
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
    valid_loader_2D_3D = (
        FullLoader2D_3D(root_url_2D=root_url_2D,
                        root_dir_3D=root_dir_3D,
                        num_points=2048,
                        split="valid",
                        semantic_level=args.data_type,
                        n_compositions=args.n_comp)
        .make_loader(batch_size=args.batch_size, num_workers=num_workers)
    )

    # ======== Load 2D model ========
    segformer_pretrain_path = '/home/ubuntu/3dcompat/workspace/3DCoMPaT-v2/models/2D/segmentation/pretrain/coarse_part_best_model.pth'
    segformer = SegFormer2D(segformer_pretrain_path, args, PART_CLASSES)

    segformer_mat_pretrain_path = '/home/ubuntu/3dcompat/workspace/3DCoMPaT-v2/models/2D/segmentation/pretrain/coarse_mat_best_model.pth'
    segformer_mat = SegFormer2D(segformer_mat_pretrain_path, args, MAT_CLASSES)
    # ======== Load 3D model ========
    ##== 1. Setup model
    part_log_dir = "2023-06-04_05-57"
    mat_log_dir = "" #TODO
    cls_log_dir = "2023-06-08_14-31"
    pointnet2 = PointNet2(part_log_dir, cls_log_dir, mat_log_dir, shape_prior, args)

    ##== 2. Setup Output
    # init_predicted_files()

    saved_results = dict()
    saved_part_predictions = dict()
    saved_cls_predictions = dict()

    # ======== Inference ========
    total_seen2d, total_seen3d, total_correct2d, total_correct3d, total_correct2d3d = 0.0, 0.0, 0, 0, 0
    total_correct2d_mat, total_correct3d_mat = 0, 0
    for i, data_tuple in tqdm(enumerate(valid_loader_2D_3D)):
        shape_id, image, shape_label, \
        part_mask, mat_mask, depth, \
        style_id, view_id, view_type, \
        cam_parameters,\
        points, points_part_labels, points_mat_labels = data_tuple

        # Setup data:
        points_part_labels = points_part_labels.cpu().data.numpy()
        part_mask = part_mask.cpu().data.numpy()
        # 2D data shifted labels by 1 to make 0 represent null pixel
        part_mask -= np.ones(part_mask.shape, dtype=int)
        imsize = image.shape[-1]

        # Start compute
        if args.debug:
            print("------>>> ", shape_id, shape_label, image.shape)
        with torch.no_grad():
            ## Predict shape cls
            predicted_cls = pointnet2.infer_cls(points).cpu().data.numpy()
            # print(predicted_cls, shape_label)
            ## Predict 2D
            outputs, logits, predicted = segformer.infer(image)
            outputs_mat, logits_mat, predicted_mat = segformer_mat.infer(image)
            # outputs, logits, predicted = outputs.cpu(), logits.cpu(), predicted.cpu()
            ## Predict 3D
            pointnet2.partmodel = pointnet2.partmodel.eval()
            logits3d, predicted3d = pointnet2.infer_part(points, shape_label)
            logits3d, predicted3d =  logits3d.cpu(), predicted3d
            ## Predict Mat 3D
            if mat_log_dir != "":
                pointnet2.matmodel = pointnet2.matmodel.eval()
                logits3d_mat, predicted3d_mat = pointnet2.infer_mat(points, shape_label)
                logits3d_mat, predicted3d_mat =  logits3d_mat.cpu(), predicted3d_mat
            
            if not args.nofuse:
                logits3d_from2d = get_logits_from2d(points.cpu(), logits.cpu(), cam_parameters.cpu())
                zeros_tensor = torch.zeros((logits3d.shape[0], logits3d.shape[1], 1)).cpu()
                # # Concatenate the original tensor with the zeros tensor along the third dimension
                logits3d_extended = torch.cat((zeros_tensor, logits3d), dim=2).cpu()
                fused_logits = logits3d_from2d + logits3d_extended

                saved_results = update_part_logits(saved_results, shape_id, style_id, fused_logits)
                fused_prediction_np = get_fused_prediction(fused_logits, predicted3d)
                correct_fuse_multi = np.sum(points_part_labels[0] == fused_prediction_np[0])
                saved_cls_predictions, saved_part_predictions = update_predictions(shape_id, style_id, predicted_cls, fused_prediction_np, saved_cls_predictions, saved_part_predictions)

            ## Count scores
            correct3d = np.sum(points_part_labels == predicted3d)
            correct2d = np.sum(part_mask == predicted)

            correct3d_mat = np.sum(points_mat_labels == predicted3d_mat) if mat_log_dir != "" else 0
            correct2d_mat = np.sum(mat_mask == predicted_mat) if mat_log_dir != "" else 0

            if args.debug:
                print(points_part_labels[0], fused_prediction_np[0])
                print("3D: {}\tFuse_single: {}\tFuse_multi:{}".format(np.sum(points_part_labels[0] == predicted3d[0]), -1, correct_fuse_multi))
            
            if args.debug:
                print("Part(2D, 3D): {} {}\t Mat(2D, 3D): {} {}".format(correct2d, correct3d, correct2d_mat, correct3d_mat))
            # print(points_part_labels[0], predicted_oncloud, correct2d3d)
        total_correct3d += correct3d
        total_correct2d += correct2d
        total_correct3d_mat += correct3d_mat
        total_correct2d_mat += correct2d_mat
        # total_correct2d3d += correct2d3d
        total_seen3d += args.batch_size * args.npoint
        total_seen2d += args.batch_size * imsize * imsize
        if i % 50 == 0:
            save_predictions_checkpoint(saved_cls_predictions, saved_part_predictions, saved_results, test_order)
            saved_cls_predictions, saved_part_predictions, saved_results = dict(), dict(), dict()
    # np.savez("cls_predictions.npz", saved_cls_predictions)
    # np.savez("parts_predictions.npz", saved_part_predictions)
    # np.savez("parts_fused_logits.npz", saved_results)
        # break
        # if i > 3:
        #     break
    print("2D_acc: {}, 3D_acc: {}\t2D_mat: {}, 3D_mat: {}".format(
        total_correct2d/total_seen2d, total_correct3d/total_seen3d,
        total_correct2d_mat/total_seen2d, total_correct3d_mat/total_seen3d))

if __name__ == "__main__":
    main()
