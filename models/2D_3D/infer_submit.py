"""
Evaluate 3D shape classification.
python infer_submit.py  --log_dir="2023-06-08_14-31" --batch_size=128 --num_point=2048 --data_type="coarse" --n_comp=10
"""
import argparse
import importlib
import logging
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../loaders/3D/')))
from compat3D_PC import EvalLoader_PC

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../3D/')))
from submission_utils import write_result, Submission

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../3D/models/')))
from pointnet2 import PointNet2


def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser("Testing")
    parser.add_argument(
        "--use_cpu", action="store_true", default=False, help="use cpu mode"
    )
    parser.add_argument("--gpu", type=str, default="0", help="specify gpu device")
    parser.add_argument(
        "--batch_size", type=int, default=24, help="batch size in training"
    )
    parser.add_argument(
        "--num_category",
        default=42,
        type=int,
        choices=[10, 40],
        help="training on ModelNet10/40",
    )
    parser.add_argument("--num_point", type=int, default=1024, help="Point Number")
    parser.add_argument("--log_dir", type=str, required=True, help="Experiment root")
    parser.add_argument(
        "--normal", action="store_true", default=False, help="use normals"
    )
    parser.add_argument(
        "--use_uniform_sample",
        action="store_true",
        default=False,
        help="use uniform sampiling",
    )
    parser.add_argument(
        "--num_votes",
        type=int,
        default=1,
        help="Aggregate classification scores with voting",
    )
    parser.add_argument(
        "--shape_prior", action="store_true", default=True, help="use shape prior"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="fine",
        help="Semantic level to be used for training",
    )
    parser.add_argument(
        "--n_comp",
        type=int,
        default=1,
        help="Number of components to be used for training",
    )
    return parser.parse_args()


def read_predictions(model, loader, num_class=40, vote_num=1, batch_size=24):
    mean_correct = []
    if not args.use_cpu:
        predictions = torch.zeros(len(loader) * batch_size).cuda()

    cls_path = "/home/ubuntu/3dcompat/workspace/3DCoMPaT-v2/models/2D_3D/out_np/cls_test_predictions.npz"
    parts_path = "/home/ubuntu/3dcompat/workspace/3DCoMPaT-v2/models/2D_3D/out_np/parts_test_predictions.npz"
    loaded_cls_predictions = read_npz(cls_path)
    loaded_parts_predictions = read_npz(parts_path)
    assert len(loaded_cls_predictions) == len(loaded_parts_predictions)
    n_shapes = len(loaded_cls_predictions)
    loaded_shape_ids = loaded_cls_predictions.keys()
    print("Loaded {} shapes from predictions: {}".format(n_shapes, loaded_shape_ids))

    submission = Submission(n_shapes)
    
    idx = 0
    checked_sids = set()
    visited = set()
    for i, (shape_id, style_id, points) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points = points.cuda()
        for j, sid in enumerate(shape_id):
            stid =  int(style_id[j])
            key = (sid, stid)
            print(idx, key)
            # assert key in loaded_shape_ids
            assert key not in visited
            visited.add(key)
            # cl, parts = loaded_cls_predictions[sid], loaded_parts_predictions[sid]
            # submission.update_cl(idx, cl, sid)
            # submission.update_parts(idx, parts, sid)
            # idx += 1
            # checked_sids.add(sid)
            # break
    submission.write_file(checked_sids)

def read_npz(filepath):
    loaded_data = np.load(filepath, allow_pickle=True)
    # print([key for key in loaded_data.keys()])
    results = loaded_data['arr_0'].item()
    return dict(results)

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    shape_prior = args.shape_prior

    # Logging
    args = parse_args()
    root_url_2D = "/home/ubuntu/3dcompat/workspace/3DCoMPaT-v2/download/3DCoMPaT_2D/"
    root_dir_3D = os.path.join(os.getcwd(),"../../models/3D/" "data/" + args.data_type + "_grained/")
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_string("PARAMETER ...")
    log_string(args)

    # Dataloader
    log_string("Load dataset ...")
    DATASET = EvalLoader_PC(
        root_dir=root_dir_3D, 
        semantic_level=args.data_type,
        num_points=args.num_point, 
        # transform=pc_transform, 
        # half_precision=pc_half_precision,
        # normalize_points=normalize_points,
        #seg_mode="part", #, random=False
        is_rgb=True,
        #n_compositions=args.n_comp,
    )
    dataLoader = torch.utils.data.DataLoader(
        DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10
    )
    # Loading the models
    part_log_dir = "2023-06-04_05-57"
    cls_log_dir = "2023-06-08_14-31"

    pointnet2 = PointNet2(part_log_dir, cls_log_dir, shape_prior, args)

    with torch.no_grad():
        instance_acc_avg, class_acc_avg = 0.0, 0.0
        # for _ in range(5):
        predictions = read_predictions(
            pointnet2.clsmodel.eval(),
            dataLoader,
            vote_num=args.num_votes,
            num_class=43,
            batch_size=args.batch_size
        )
        log_string(
            "Test Instance Accuracy: %f, Class Accuracy: %f"
            % (instance_acc, class_acc)
        )
        instance_acc_avg += instance_acc
        class_acc_avg += class_acc

        # print("Running 5 times average: ", instance_acc_avg / 5.0, class_acc_avg / 5.0)
        # write_result(predictions)

if __name__ == "__main__":
    args = parse_args()
    main(args)
