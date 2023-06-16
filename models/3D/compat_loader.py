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


def load_data(data_dir, partition, seg_mode, use_features=False):
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
        points_part_labels = None
        if partition != "test":
            points_part_labels = np.array(f["points_part_labels"][:]).astype("int16")
            points_mat_labels = np.array(f["points_mat_labels"][:]).astype("uint8")
        shape_ids = f["shape_id"][:].astype("str")
        NP =1
        # print(len(f["points"][0]), f["shape_label"][0], len(f[point_label_key][0]), len(f["points_mat_labels"][0]))
        shape_labels = None
        if partition != "test":
            shape_labels = np.array(f["shape_label"][:]).astype("uint8")

        normalized_points = np.zeros(points.shape)
        for i in range(points.shape[0]):
            normalized_points[i] = pc_normalize(points[i])

    partseg_2dlogits, matseg_2dlogits = None, None

    if use_features:
        features_file_name = os.path.join("/home/ubuntu/3dcompat/features/", "seg2d_logits_{}.hdf5".format(partition))
        with h5py.File(features_file_name, "r") as ff:
            partseg_2dlogits = ff["partseg_2dlogits"][:].astype("float32")
            matseg_2dlogits = ff["matseg_2dlogits"][:].astype("float")

        ## ======= Use logits
        # normalized_partlogits = np.zeros(partseg_2dlogits.shape)
        # normalized_matlogits = np.zeros(matseg_2dlogits.shape)
        # for i in range(matseg_2dlogits.shape[0]):
        #     # part_norm = np.linalg.norm(partseg_2dlogits[i], axis=1).reshape((partseg_2dlogits[i].shape[0], 1))
        #     # part_norm_reshaped = np.tile(part_norm, (1, partseg_2dlogits[i].shape[1]))
        #     # normalized_partlogits[i] = partseg_2dlogits[i] / part_norm_reshaped

        #     mat_norm = np.linalg.norm(matseg_2dlogits[i], axis=1).reshape((matseg_2dlogits[i].shape[0], 1))
        #     mat_norm_reshaped = np.tile(mat_norm, (1, matseg_2dlogits[i].shape[1]))
        #     normalized_matlogits[i] = matseg_2dlogits[i] / mat_norm_reshaped

        #     assert not np.isnan(mat_norm_reshaped).any()
        #     assert not np.isinf(mat_norm_reshaped).any()
        #     # print("\nPart:", partseg_2dlogits[i], normalized_partlogits[i], partseg_2dlogits[i].shape)
        #     # print("\nMat:", matseg_2dlogits[i], normalized_matlogits[i], matseg_2dlogits[i].shape)
        # normalized_points = np.concatenate((normalized_points, normalized_matlogits), axis=2)
        
        ## ===== Use top 3 ======
        print(matseg_2dlogits.shape)
        seglogits = partseg_2dlogits if seg_mode == "part" else matseg_2dlogits if seg_mode == "mat" else None
        normalized_top3 = np.zeros(seglogits.shape)[:,:,:3]
        for i in range(seglogits.shape[0]):
            sorted_matlogits = np.argsort(-seglogits[i], axis=1)
            top_indices = sorted_matlogits[:, :3]
            half_nmats = seglogits.shape[2] // 2
            normalized_top3[i] = (top_indices - half_nmats) / half_nmats
        normalized_part_points = np.concatenate((normalized_points, normalized_top3), axis=2)

    return normalized_points, normalized_part_points, points_part_labels, points_mat_labels, shape_ids, shape_labels


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
        use_features=False,
        random=True,
    ):
        # train, test, valid
        self.partition = split.lower()
        self.seg_mode = seg_mode
        self.use_features = use_features
        self.data, self.partseg, self.matseg, self.shape_ids, self.label = load_data(
            data_root, self.partition, self.seg_mode, use_features=use_features
        )

        self.num_points = num_points
        self.transform = transform
        self.random = random
        self.pm = set()

    def __getitem__(self, item):
        MAX_RAND = self.num_points
        idx = range(self.num_points)
        if self.random:
            idx = np.random.choice(MAX_RAND, self.num_points, False)
        pointcloud = self.data[item][idx]
        label = self.label[item]
        partseg = self.partseg[item][idx].astype(np.int32)
        matseg = self.matseg[item][idx].astype(np.int32)
        shape_id = self.shape_ids[item]
        pointcloud = torch.from_numpy(pointcloud)
        seg = partseg if self.seg_mode == "part" else matseg if self.seg_mode == "mat" else None
        
        # print(len(self.pm))
        if self.seg_mode == "partmat":
            # seg = np.concatenate((partseg, matseg), axis=0).astype(np.int32)
            seg = partseg * self.num_mats() + matseg
        seg = torch.from_numpy(seg)
        return pointcloud, label, seg, shape_id

    def __len__(self):
        return self.data.shape[0]

    def num_classes(self):
        return np.max(self.label) + 1

    def num_parts(self):
        return np.max(self.partseg) + 1

    def num_mats(self):
        return np.max(self.matseg) + 1

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
        idx = range(len(self.data[item]))
        if self.random:
            idx = np.random.choice(self.num_points, self.num_points, False)
        pointcloud = self.data[item][idx].astype(np.float32)
        label = None

        pointcloud = torch.from_numpy(pointcloud)
        if self.partition == "test":
            return pointcloud

        label = self.label[item]
        return pointcloud, label
