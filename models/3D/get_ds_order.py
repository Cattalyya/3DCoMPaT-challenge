'''
get sorted keys {shape_id}_{style_id} from 3D ds.
python get_ds_order.py --batch_size=128 --npoint=2024 --data_type=coarse --split=test
'''
import argparse
import importlib
import torch
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../loaders/3D/')))
from compat3D_PC import EvalLoader_PC


def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser("Model")
    parser.add_argument(
        "--batch_size", type=int, default=24, help="batch size in testing"
    )
    parser.add_argument(
        "--npoint", type=int, required=True, default=1024, help="point Number"
    )
    parser.add_argument("--data_type", type=str, default="coarse", help="data_type")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split for inference",
    )
    return parser.parse_args()


def main(argv=None):

    args = parse_args()

    root_dir_3D = os.path.join(os.getcwd(), "data/" + args.data_type + "_grained/")

    dataset3d = EvalLoader_PC(
            root_dir=root_dir_3D,
            split=args.split,
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
    ordered_keys = []
    with torch.no_grad():
        for _, (shape_id, style_id, points) in tqdm(
            enumerate(test_data_loader), total=len(test_data_loader), smoothing=0.9
        ):
            for sid, stid in zip(shape_id, style_id):
                ordered_keys.append("{}_{}".format(sid, stid))

    print(ordered_keys, len(ordered_keys))


if __name__ == "__main__":
    main()