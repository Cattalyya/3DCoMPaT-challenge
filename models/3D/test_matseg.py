"""
Script for testing mat segmentation.
python test_matseg.py --data_type=coarse --batch_size=128 --npoint=2048
"""
import argparse
import importlib
import json
import logging
import os
import sys

import numpy as np
import torch
from compat_loader import CompatLoader3D as CompatSeg
from compat_utils import compute_overall_iou, to_categorical
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))


sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../loaders/3D/')))
from compat3D_PC import EvalLoader_PC

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), './models/')))
from pointnet2 import PointNet2

def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser("Model")
    parser.add_argument(
        "--batch_size", type=int, default=24, help="batch size in testing"
    )
    parser.add_argument("--gpu", type=str, default="0", help="specify gpu device")
    parser.add_argument(
        "--normal", action="store_true", default=False, help="use normals"
    )
    parser.add_argument(
        "--num_votes",
        type=int,
        default=3,
        help="aggregate segmentation scores with voting",
    )
    parser.add_argument(
        "--model", type=str, default="pointnet2_part_seg_ssg", help="model name"
    )
    parser.add_argument(
        "--npoint", type=int, required=True, default=1024, help="point Number"
    )
    parser.add_argument(
        "--shape_prior", action="store_true", default=True, help="use shape prior"
    )
    parser.add_argument("--data_type", type=str, default="coarse", help="data_type")
    parser.add_argument("--split", type=str, default="test", help="split")

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    cls_log_dir = "2023-06-08_14-31"
    mat_log_dir = "2023-06-10_09-20"
    # log/mat_seg/2023-06-10_10-29/checkpoints/best_model.pth
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = "log/mat_seg/" + mat_log_dir

    # Logging
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler("%s/eval.txt" % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string("PARAMETER ...")
    log_string(args)
    root_dir_3D = os.path.join(os.getcwd(), "data/" + args.data_type + "_grained/")

    dataset3d = EvalLoader_PC(
            split=split
            root_dir=root_dir_3D,
            semantic_level=args.data_type,
            num_points=args.npoint,
            is_rgb=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset3d,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=10,
        drop_last=False,
    )
    log_string("The number of test data is: %d" % len(dataset3d))

    metadata = json.load(open("/home/ubuntu/3dcompat/workspace/3DCoMPaT-v2/models/3D/metadata/coarse_mat_seg_classes.json"))
    
    num_classes = metadata["num_classes"]
    num_mat = metadata["num_mat"]
    # seg_classes = metadata["seg_classes"]

    # Loading the models
    shape_prior = args.shape_prior
    
    pointnet2 = PointNet2(cls_log_dir, "", mat_log_dir, shape_prior, args)

    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_mat)]
        total_correct_class = [0 for _ in range(num_mat)]
        # shape_ious = {cat: [] for cat in seg_classes.keys()}
        general_miou = []

        pointnet2.matmodel = pointnet2.matmodel.eval()

        for _, (shape_id, style_id, points) in tqdm(
            enumerate(test_data_loader), total=len(test_data_loader), smoothing=0.9
        ):
            print(points.shape)

            predicted_cls = pointnet2.infer_cls(points[:,:,:3]).cpu()
            logits3d_mat, predicted3d_mat = pointnet2.infer_mat(points, predicted_cls)
            print(predicted3d_mat[0], predicted3d_mat.shape)
            break
            target = target.cpu().data.numpy()

            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += points.size

            for part_k in range(num_mat):
                total_seen_class[part_k] += np.sum(target == part_k)
                total_correct_class[part_k] += np.sum(
                    (cur_pred_val == part_k) & (target == part_k)
                )

        #     # calculate the mIoU given shape prior knowledge and without it
        #     miou = compute_overall_iou(cur_pred_val, target, num_part)
        #     general_miou = general_miou + miou
        #     for i in range(cur_batch_size):
        #         segp = cur_pred_val[i, :]
        #         segl = target[i, :]
        #         shape = str(label[i].item())
        #         part_ious = {}
        #         for class_k in seg_classes[shape]:
        #             if (np.sum(segl == class_k) == 0) and (
        #                 np.sum(segp == class_k) == 0
        #             ):  # part is not present, no prediction as well
        #                 part_ious[class_k] = 1.0
        #             else:
        #                 part_ious[class_k] = np.sum(
        #                     (segl == class_k) & (segp == class_k)
        #                 ) / float(np.sum((segl == class_k) | (segp == class_k)))
        #         # Convert the dictionary to a list
        #         part_ious = list(part_ious.values())
        #         shape_ious[shape].append(np.mean(part_ious))

        # all_shape_ious = []
        # for cat in shape_ious.keys():
        #     for iou in shape_ious[cat]:
        #         all_shape_ious.append(iou)
        #     shape_ious[cat] = np.mean(shape_ious[cat])
        # mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics["accuracy"] = total_correct / float(total_seen)
        # test_metrics["class_avg_accuracy"] = np.mean(
        #     np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)
        # )

        # for cat in sorted(shape_ious.keys()):
        #     log_string(
        #         "eval mIoU of %s %f" % (cat + " " * (14 - len(cat)), shape_ious[cat])
        #     )
        # test_metrics["class_avg_iou"] = mean_shape_ious
        # test_metrics["inctance_avg_iou"] = np.mean(all_shape_ious)
        test_metrics["avg_iou_wihtout_shape"] = np.nanmean(general_miou)

    log_string("Best accuracy is: %.5f" % test_metrics["accuracy"])
    # log_string("Best class avg mIOU is: %.5f" % test_metrics["class_avg_iou"])
    # log_string("Best inctance avg mIOU is: %.5f" % test_metrics["inctance_avg_iou"])
    log_string("Best general avg mIOU is: %.5f" % test_metrics["avg_iou_wihtout_shape"])


if __name__ == "__main__":
    args = parse_args()
    main(args)
