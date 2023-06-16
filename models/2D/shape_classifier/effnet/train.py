from transformers import AutoImageProcessor, EfficientNetForImageClassification
import torch
import time
from datetime import timedelta
# from datasets import load_dataset

import os
import argparse
import sys
sys.path.append('../../../../loaders/2D/')

from compat2D import ShapeLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import timm
sys.path.append('../')
from training_utils import calculate_metrics, seed_everything

# def make_transform(mode):
#     if mode == "train":
#         return T.Compose(
#             [
#                 T.RandomHorizontalFlip(),
#                 T.Normalize([0.8726, 0.8628, 0.8577], [0.2198, 0.2365, 0.2451]),
#             ]
#         )
#     elif mode == "valid":
#         return T.Compose(
#             [T.Normalize([0.8710, 0.8611, 0.8561], [0.2217, 0.2384, 0.2468])]
#         )

# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.resnetmodel = EfficientNet.from_pretrained('efficientnet-b3')
        
#         self.fc = nn.Sequential(nn.Linear(1000, 512), nn.ReLU(),
#                                   nn.Linear(512, 1))
# #                                 , nn.Sigmoid())
        
#     def forward(self, x):
#         x = self.resnetmodel(x)
#         return self.fc(x)
class EfficientNet_V2(nn.Module):
    def __init__(self, n_out):
        super(EfficientNet_V2, self).__init__()
        # Define model
        self.effnet = timm.create_model('efficientnetv2_s', pretrained=True, num_classes=n_out)

    def forward(self, x):
        return self.effnet(x)


def parse_args(argv):
    """
    Parse input arguments.
    """

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
        "--nepochs",
        default=50,
        type=int,
        required=True,
        help="Number of epochs to train for (default=%(default)s)",
    )
    parser.add_argument(
        "--num-classes",
        default=42,
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
        "--patience",
        type=int,
        default=3,
        required=False,
        help="Use patience while training (default=%(default)s)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training with the last saved model (default=%(default)s)",
    )

    # parser.add_argument(
    #     "--resnet-type",
    #     default="resnet18",
    #     type=str,
    #     required=True,
    #     choices=["resnet18", "resnet50"],
    #     help="ResNet variant to be used for training (default=%(default)s)",
    # )
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="Use a model pre-trained on ImageNet (default=%(default)s)",
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        required=False,
        default="/home/ubuntu/3dcompat/workspace/3DCoMPaT-v2/models/2D/shape_classifier/effnet/ckpt/",
        help="Model directory to use to save models (default=%(default)s)",
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

            top5_hits, top1_hits = calculate_metrics(outputs, labels)
            total_top1_hits += top1_hits
            total_top5_hits += top5_hits
            N += labels.shape[0]

    top1_avg = 100 * (float(total_top1_hits) / N)
    top5_avg = 100 * (float(total_top5_hits) / N)

    return top1_avg, top5_avg

def train(args):

    # dataset = 
    # image = dataset["test"]["image"][0]

    # image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b3")
    device = torch.device("cuda:0")
    # model = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b3")
    model = EfficientNet_V2(args.num_classes)
    if args.resume:
        all_models = [f for f in os.listdir(args.models_dir) if ".ckpt" in f]
        all_models.sort()

        last_model = os.path.join(args.models_dir, all_models[-1])

        checkpoint = torch.load(last_model)
        print(
            "Loaded last checkpoint from [%s], with %0.2f train top-1 accuracy."
            % (last_model, checkpoint["top1_train_acc"])
        )
        print(checkpoint["state_dict"])
        model.load_state_dict(checkpoint["state_dict"], strict=False)

    model = model.to(device)

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

    # Defining data loaders
    train_loader = ShapeLoader(
        args.root_url,
        split="train",
        semantic_level=args.data_type,
        n_compositions=args.n_comp,
        # transform=make_transform("train"),
    ).make_loader(args.batch_size, args.num_workers)

    test_loader = ShapeLoader(
        args.root_url,
        split="valid",
        semantic_level=args.data_type,
        n_compositions=args.n_comp,
        # transform=make_transform("valid"),
    ).make_loader(args.batch_size, args.num_workers)

    starting_time = time.time()

    n_batch = 0

    for n_epoch in tqdm(range(args.nepochs)):
        # Training the model
        # optimizer_ft.zero_grad()
        model.train()

        total_top1_hits, total_top5_hits, N = 0, 0, 0
        top1_avg, top5_avg = 0, 0

        for images, targets in tqdm(train_loader):
            images, targets = images.to(device), targets.to(device)
            # inputs = image_processor(images, return_tensors="pt")

            # logits = model(images)
            # print(logits, logits.shape)
            # # # model predicts one of the 1000 ImageNet classes
            # predicted_label = logits.argmax(-1)
            # print(model.config.id2label[predicted_label])

            # print(logits[0], targets[0], logits.shape, targets.shape)

            # np.argmax(logits)
            targets = targets.squeeze()

            ## Forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_ft.step()
            optimizer_ft.zero_grad()

            ## Logging results
            top5_hits, top1_hits = calculate_metrics(outputs, targets)
            total_top1_hits += top1_hits
            total_top5_hits += top5_hits
            N += images.shape[0]
            top1_avg = 100 * (float(total_top1_hits) / N)
            top5_avg = 100 * (float(total_top5_hits) / N)

            ## Making a checkpoint
            if n_batch % 1000 == 0:
                predicted_label = outputs.argmax(-1)
                print(predicted_label, targets)
                saved_model = os.path.join(
                    args.models_dir, "batch_%d.ckpt" % (n_batch)
                )

                # Measuring model test-accuracy
                top1_test_acc, top5_test_acc = evaluate(model, test_loader, device)

                print("Saved model at: [" + saved_model + "]")
                state = {
                    "top1_train_acc": top1_avg,
                    "top5_train_acc": top5_avg,
                    "top1_test_acc": top1_test_acc,
                    "top5_test_acc": top5_test_acc,
                    "state_dict": model.state_dict(),
                }
                torch.save(state, saved_model)
            
            n_batch += 1
            scheduler.step()


        # Logging results
        current_elapsed_time = time.time() - starting_time
        print(
            "{:03}/{:03} | {} | Train : loss = {:.4f} | top-1 acc = {:.3f} | top-5 acc = {:.3f}".format(
                n_epoch + 1,
                args.nepochs,
                timedelta(seconds=round(current_elapsed_time)),
                loss,
                top1_avg,
                top5_avg,
            )
        )

def main(argv=None):
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()

