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
from segformer import SegFormer2D

def parse_args(argv):
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(
        description="Training the Baseline Models on 2D Part and Material Segmentation"
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
        "--model_name",
        type=str,
        required=False,
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="Name of the model file",
    )
    parser.add_argument(
        "--root_url",
        type=str,
        required=False,
        default="/home/ubuntu/3dcompat/workspace/3DCoMPaT-v2/download/3DCoMPaT_2D/",
        help="Root URL of the 2D data",
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
        "--n_comp",
        type=int,
        default=1,
        help="Number of components to be used for training",
    )
    parser.add_argument(
        "--split",
        default="validgt",
        type=str,
        required=True,
        choices=["valid", "test", "validgt"],
        help="Type of data to make inference with (default=%(default)s)",
    )
    args = parser.parse_args(argv)

    # Printing input arguments
    print("Input arguments:")
    print(args)

    return args


def load_data(args):
    """
    Instantiate the data loaders.
    """
    if args.split == "validgt":
        return GCRLoader(
            root_url=args.root_url,
            split="valid",
            semantic_level=args.data_type,
            n_compositions=args.n_comp,
        ).make_loader(batch_size=args.batch_size, num_workers=6)

    loader = EvalLoader(
        root_url=args.root_url,
        depth_transform=wds_identity,
        semantic_level=args.data_type,
        n_compositions=args.n_comp,
    ).make_loader(batch_size=args.batch_size, num_workers=6)

    return loader




def inference(model, gcr_loader, device, args):

    IMSIZE = 256
    predicted_all = np.empty((args.batch_size, IMSIZE, IMSIZE))
    for  _model, images, _depth, _style, _view, _view_type, _cam  in tqdm(gcr_loader):
        assert images.shape[-2:][0] == IMSIZE
        with torch.no_grad():
            predicted = model.infer(images)
            predicted_all = np.concatenate((predicted_all, predicted), axis=0)
    return predicted_all

def evaluation(model, gcr_loader, args, num_labels):

    # Define the evaluation metric
    if not os.path.exists("./logs"):
        os.mkdir("./logs")

    if not os.path.exists("./logs/" + str(args.model)):
        os.mkdir("./logs/" + str(args.model))

    checkpoint_name = (
        "./logs/"
        + str(args.model)
        + "/"
        + str(args.data_type)
        + "_"
        + str(args.task)
        + "_best_model.pth"
    )
    start_time = time.time()

    if args.split == "validgt":
        # Torch array of predictions
        general_miou = []
        precisions = []
        for images, _, y_parts, y_mats   in tqdm(gcr_loader):
            y = y_parts if args.task == "part" else y_mats
            with torch.no_grad():
                outputs, predicted = model.infer(images.to(model.device))
                y_truth = y.cpu().numpy()
                miou = compute_overall_iou(predicted, y_truth, num_labels)
                prec = compute_overall_precision(predicted, y_truth)
                general_miou = general_miou + miou
                precisions = precisions + prec
        # Compute the metrics: mean IOU and mean precision
        mIOU = np.nanmean(general_miou)
        mPrecision = np.nanmean(precisions)
        print(
            f"mIOU: {mIOU:.6f}",
            f"mPrecision: {mPrecision:.6f}",
        )
    else: 
        outputs, sementic_images = inference(model, gcr_loader, model.device, args)

def main(argv=None):
    """
    Train and evaluate the model.
    """
    args = parse_args(argv)

    PART_CLASSES = 276
    MAT_CLASSES = 14

    if args.data_type == "coarse":
        PART_CLASSES = 44

    num_labels = PART_CLASSES if args.task == "part" else MAT_CLASSES


    # Path to the pre-trained checkpoint
    pretrain_path = '/home/ubuntu/3dcompat/workspace/3DCoMPaT-v2/models/2D/segmentation/pretrain/coarse_{}_best_model.pth'.format(args.task)
    segformer = SegFormer2D(pretrain_path, args, num_labels)

    loader = load_data(args)
    evaluation(segformer, loader, args, num_labels)


if __name__ == "__main__":
    main()
