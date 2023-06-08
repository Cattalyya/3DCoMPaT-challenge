"""
Training a basic ResNet-18 on the 3DCompat dataset for shape classification.
python test.py --resnet-type="resnet50" --num-classes=42 --n-comp=10  --data_type=coarse  --split="test" --batch-size=128
"""
import argparse
import os
import time
from datetime import timedelta
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from tqdm import tqdm
import sys
sys.path.append('../../../loaders/2D/')

from compat2D import ShapeLoader

from torch.nn.parallel import DataParallel
from torchvision import models
from training_utils import calculate_metrics, seed_everything


def make_transform(mode):
    if mode == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.Normalize([0.8726, 0.8628, 0.8577], [0.2198, 0.2365, 0.2451]),
            ]
        )
    elif mode == "valid" or mode == "test":
        return T.Compose(
            [T.Normalize([0.8710, 0.8611, 0.8561], [0.2217, 0.2384, 0.2468])]
        )

def parse_args(argv):
    """
    Parse input arguments.
    """
    ORG_PRETRAINED_DIR = "/home/ubuntu/3dcompat/workspace/3DCoMPaT-v2/models/2D/shape_classifier/pretrain/"
    PRETRAINED_DIR = "/home/ubuntu/3dcompat/workspace/3DCoMPaT-v2/models/2D/shape_classifier/ckpts/"
    # Arguments
    parser = argparse.ArgumentParser(
        description="Training a basic ResNet-18 on the 3DCompat dataset."
    )

    # miscellaneous args
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (default=%(default)s)"
    )

    # dataset args
    parser.add_argument(
        "--num-workers",
        default=6,
        type=int,
        required=False,
        help="Number of subprocesses to use for the dataloader (default=%(default)s)",
    )
    parser.add_argument(
        "--use-tmp",
        action="store_true",
        help="Use local temporary cache when loading the dataset (default=%(default)s)",
    )

    # data args
    parser.add_argument(
        "--model-path",
        type=str,
        default=ORG_PRETRAINED_DIR + "resnet50_batch_151000.ckpt",
        # default=PRETRAINED_DIR + "resnet18_batch_7000.ckpt",
        help="Pretrained model path (default=%(default)s)",
    )
    # data args
    parser.add_argument(
        "--root-url",
        type=str,
        default="/home/ubuntu/3dcompat/workspace/3DCoMPaT-v2/download/3DCoMPaT_2D/",
        #default="/lustre/scratch/project/k1546/3DCoMPaT-v2/shards/",
        help="Root URL for WebDataset shards (default=%(default)s)",
    )
    parser.add_argument(
        "--n-comp",
        type=int,
        required=True,
        help="Number of compositions per model to train with",
    )
    parser.add_argument(
        "--view-type",
        type=str,
        default="all",
        choices=["canonical", "random", "all"],
        help="Train on a specific view type (default=%(default)s)",
    )

    # training args
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        required=False,
        help="Batch size to use (default=%(default)s)",
    )
    parser.add_argument(
        "--weight-decay",
        default=0.0005,
        type=float,
        required=False,
        help="Weight decay (default=%(default)s)",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        required=False,
        help="Momentum (default=%(default)s)",
    )
    parser.add_argument(
        "--num-classes",
        default=43,
        type=int,
        required=True,
        help="Number of classes to train with",
    )
    parser.add_argument(
        "--data_type",
        default="coarse",
        type=str,
        required=True,
        choices=["coarse", "fine"],
        help="Type of data to train with (default=%(default)s)",
    )
    parser.add_argument(
        "--split",
        default="valid",
        type=str,
        required=True,
        choices=["valid", "test"],
        help="Type of data to make inference with (default=%(default)s)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        required=False,
        help="Use patience while training (default=%(default)s)",
    )

    parser.add_argument(
        "--resnet-type",
        default="resnet18",
        type=str,
        required=True,
        choices=["resnet18", "resnet50"],
        help="ResNet variant to be used for training (default=%(default)s)",
    )
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="Use a model pre-trained on ImageNet (default=%(default)s)",
    )

    args = parser.parse_args(argv)
    args.view_type = (
        -1 if args.view_type == "all" else ["canonical", "random"].index(args.view_type)
    )

    # Printing input arguments
    print("Input arguments:")
    print(args)

    return args


def evaluate(net, test_loader, device):
    """
    Evaluate the resulting classifier using a given test set loader.
    """
    total_top1_hits, total_top5_hits, N = 0, 0, 0
    top1_avg, top5_avg = 0, 0

    # net.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device).squeeze()

            outputs = net(images)
            predicted_labels = np.argmax(outputs.cpu(), axis=1)
            # print(outputs[0], predicted_labels[0])
            # print(outputs.shape, predicted_labels.shape)

            top5_hits, top1_hits = calculate_metrics(outputs, labels)
            total_top1_hits += top1_hits
            total_top5_hits += top5_hits
            N += labels.shape[0]

            # print(predicted_labels, labels)
            # break

    print("Res top1: ", total_top1_hits, N)
    print("Res top5: ", total_top5_hits, N)
    top1_avg = 100 * (float(total_top1_hits) / N)
    top5_avg = 100 * (float(total_top5_hits) / N)

    return top1_avg, top5_avg

def inference(net, test_loader, device):
    """
    Evaluate the resulting classifier using a given test set loader.
    """
    # net.eval()

    all_outputs = torch.tensor([])
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device).squeeze()

            outputs = net(images)
            print(outputs)
            all_outputs += outputs
            print("ALL", all_outputs)
            # all_outputs = torch.cat((all_outputs.unsqueeze(0), outputs.unsqueeze(0)), dim=0)
            N += labels.shape[0]
            if N > 280:
                break

    print("[INFO] Inference on {} shapes".format(N))
    return outputs


def inference_evaluation(args):
    """
    Run the main training routine.
    """

    # Fixing random seed
    seed_everything(args.seed)

    # Setting up dataset
    print("Loading data from: [%s]" % args.root_url)

    num_classes = args.num_classes
    res_model = {"resnet18": models.resnet18, "resnet50": models.resnet50}
    fv_size = {"resnet18": 512, "resnet50": 2048}
    model = res_model[args.resnet_type](pretrained=args.use_pretrained)
    model.fc = nn.Linear(fv_size[args.resnet_type], num_classes)

    device = torch.device("cuda:0")

    # Optionally resume training
    print("Loading model from: [%s]" % args.model_path)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    # model.fc = nn.Linear(fv_size[args.resnet_type], num_classes)

    model.to(device)
    model = DataParallel(model)

    # Defining Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer_ft, base_lr=0.01, max_lr=0.1
    )

    # Training
    starting_time = time.time()

    # Main loop
    tstart = time.time()

    print("Starting inference...")

    if args.split == "valid":
        # Defining data loaders
        valid_loader = ShapeLoader(
            args.root_url,
            split="valid",
            semantic_level=args.data_type,
            n_compositions=args.n_comp,
            transform=make_transform("valid"),
        ).make_loader(args.batch_size, args.num_workers)

        total_top1_hits, total_top5_hits, N = 0, 0, 0

        # Measuring model validation-accuracy
        top1_acc, top5_acc = evaluate(model, valid_loader, device)

        # Logging results
        current_elapsed_time = time.time() - starting_time
        print(
            "{} | Validation : top-1 acc = {:.3f} | top-5 acc = {:.3f}".format(
                timedelta(seconds=round(current_elapsed_time)),
                top1_acc,
                top5_acc,
            )
        )
    elif args.split == "test": 
        test_loader = ShapeLoader(
            args.root_url,
            split="test",
            semantic_level=args.data_type,
            n_compositions=args.n_comp,
            transform=make_transform("test"),
        ).make_loader(args.batch_size, args.num_workers)
        outputs = inference(model, valid_loader, device)



    # Final output
    print("[Elapsed time = {:.1f} mn]".format((time.time() - tstart) / 60))
    print("Done!")

    print("-" * 108)


def main(argv=None):
    args = parse_args(argv)
    inference_evaluation(args)


if __name__ == "__main__":
    main()
