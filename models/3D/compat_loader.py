""""
Dataloaders for the preprocessed point clouds from 3DCoMPaT dataset.
"""
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

N_CH = 6 # 3= xyz, 6=xyzrgb

def pc_normalize(pc):
    """
    Center and scale the point cloud.
    """
    pmin = np.min(pc, axis=0)
    pmax = np.max(pc, axis=0)
    pc -= (pmin + pmax) / 2
    scale = np.max(np.linalg.norm(pc, axis=1))
    pc *= 1.0 / scale
    return pc


def load_data(data_dir, partition, seg_mode):
    """
    Pre-load and process the pointcloud data into memory.
    """
    semantic_level = data_dir.split("_")[-2].split("/")[-1]
    h5_name = os.path.join(data_dir, "{}_{}.hdf5".format(partition, semantic_level))
    with h5py.File(h5_name, "r") as f:
        print(f.keys())
        # points = np.array(f["points"][:]).astype("float32")
        points = np.array(f["points"][:, :,:N_CH]).astype("float32")
        # points_labels = np.array(f["points_labels"][:]).astype("uint16")
        points_labels = None
        if partition != "test":
            point_label_key = "points_labels" if seg_mode == "" else "points_{}_labels".format(seg_mode)
            points_labels = np.array(f[point_label_key][:]).astype("uint16")
        shape_ids = f["shape_id"][:].astype("str")
        NP =1
        # print(len(f["points"][0]), f["shape_label"][0], len(f[point_label_key][0]), len(f["points_mat_labels"][0]))
        shape_labels = None
        if partition != "test":
            shape_labels = np.array(f["shape_label"][:]).astype("uint8")

        normalized_points = np.zeros(points.shape)
        for i in range(points.shape[0]):
            normalized_points[i] = pc_normalize(points[i])

    return normalized_points, points_labels, shape_ids, shape_labels


class CompatLoader3D(Dataset):
    """
    Base class for loading preprocessed 3D point clouds.

    Args:
    ----
        data_root:   Base dataset URL containing data split shards
        split:       One of {train, valid}.
        num_points:  Number of sampled points
        transform:   data transformations
    """

    def __init__(
        self,
        data_root,
        split="train",
        num_points=4096,
        transform=None,
        seg_mode="",
        random=True,
    ):
        # train, test, valid
        self.partition = split.lower()
        self.seg_mode = seg_mode
        self.data, self.seg, self.shape_ids, self.label = load_data(
            data_root, self.partition, self.seg_mode
        )

        self.num_points = num_points
        self.transform = transform
        self.random = random

    def __getitem__(self, item):
        MAX_RAND = self.num_points
        idx = range(self.num_points)
        if self.random:
            idx = np.random.choice(MAX_RAND, self.num_points, False)
        pointcloud = self.data[item][idx]
        label = self.label[item]
        seg = self.seg[item][idx].astype(np.int32)
        shape_id = self.shape_ids[item]
        pointcloud = torch.from_numpy(pointcloud)
        seg = torch.from_numpy(seg)
        return pointcloud, label, seg, shape_id

    def __len__(self):
        return self.data.shape[0]

    def num_classes(self):
        return np.max(self.label) + 1

    def num_segments(self):
        return np.max(self.seg) + 1

    def get_shape_label(self):
        return self.label


class CompatLoader3DCls(CompatLoader3D):
    """
    Classification data loader using preprocessed 3D point clouds.

    Args:
    ----
        data_root:   Base dataset URL containing data split shards
        split:       One of {train, valid}.
        num_points:  Number of sampled points
        transform:   data transformations
    """

    def __init__(
        self,
        data_root="data/compat",
        split="train",
        num_points=4096,
        transform=None,
        seg_mode="",
        random=True
    ):
        super().__init__(data_root, split, num_points, transform, seg_mode, random)

    def __getitem__(self, item):
        return super().__getitem__(item)
        # # print(item)
        # idx = range(len(self.data[item]))
        # if self.random:
        #     idx = np.random.choice(self.num_points, self.num_points, False)
        # pointcloud = self.data[item][idx].astype(np.float32)
        # label = None

        # pointcloud = torch.from_numpy(pointcloud)
        # if self.partition == "test":
        #     return pointcloud

        # label = self.label[item]
        # # seg = self.seg[item][idx].astype(np.int32)
        # # seg = torch.from_numpy(seg)
        # return pointcloud, label
