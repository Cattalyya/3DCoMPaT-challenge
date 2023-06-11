"""
Dataloaders for the mixed 2D-3D 3DCoMPaT tasks.
"""
from functools import partial
import sys
import os
import torch
import numpy as np

import webdataset as wds
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../2D/')))
from compat2D import EvalLoader, FullLoader
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../3D/')))
from compat3D_PC import EvalLoader_PC, StylizedShapeLoader_PC

N_CH = 6 

def fetch_3D_PC(loader_3D, style_id_idx, test_mode, loader_2D):
    """
    Fetch the 3D pointcloud data from the 3D loader,
    given shape ids and style ids from the input stream.
    """
    for tuple_2D in loader_2D:
        shape_id = tuple_2D[0][0]
        style_id = tuple_2D[style_id_idx][0]
        # Fetch pointcloud
        if not test_mode:
            (
                points,
                rgb_points,
                points_part_labels,
                points_mat_labels,
            ) = loader_3D.get_stylized_shape(shape_id, style_id)[-3:]
            tuple_3D = [points, rgb_points, points_part_labels, points_mat_labels]
        else:
            points, rgb_points = loader_3D.get_stylized_shape(shape_id, style_id)[-2]
            tuple_3D = [points, rgb_points]
        yield list(tuple_2D) + tuple_3D

# `fetch_3D_PC()` loads 2D in batch but load only a single 3D object, so
# this function solves that issue by loading 3D ds in batch as well.
def fetch_3D_PC_batch(loader_3D, style_id_idx, test_mode, loader_2D):
    """
    Fetch the 3D pointcloud data from the 3D loader,
    given shape ids and style ids from the input stream.
    """
    for tuple_2D in loader_2D:
        shape_ids = tuple_2D[0]
        style_ids = tuple_2D[style_id_idx]
        batch_size = len(tuple_2D[0])
        # Fetch pointcloud
        tuple_3D = None
        if not test_mode:
            for i in range(batch_size):
                (
                    # shape_label, # int because one object only has 1 shape
                    points,
                    rgb_points,
                    points_part_labels,
                    points_mat_labels,
                ) = loader_3D.get_stylized_shape(shape_ids[i], style_ids[i])[3:]

                if tuple_3D == None:
                    # TODO(cattalyya): Train 3D dataset with RGB
                    tuple_3D = [points[:,:3].unsqueeze(0), rgb_points[:,:6].unsqueeze(0), points_part_labels.unsqueeze(0), points_mat_labels.unsqueeze(0)]#, torch.tensor([shape_label])]
                else:
                    tuple_3D[0] = torch.cat((tuple_3D[0], points[:,:3].unsqueeze(0)), dim=0)
                    tuple_3D[1] = torch.cat((tuple_3D[1], rgb_points[:,:6].unsqueeze(0)), dim=0)
                    tuple_3D[2] = torch.cat((tuple_3D[2], points_part_labels.unsqueeze(0)), dim=0)
                    tuple_3D[3] = torch.cat((tuple_3D[3], points_mat_labels.unsqueeze(0)), dim=0)
        else:
            for i in range(batch_size):
                points, rgb_points = loader_3D.get_stylized_shape(shape_ids[i], style_ids[i])[2:]
                if tuple_3D == None:
                    tuple_3D = [points[:,:3].unsqueeze(0), rgb_points[:,:6].unsqueeze(0)]
                else:
                    tuple_3D[0] = torch.cat((tuple_3D[0], points[:,:3].unsqueeze(0)), dim=0)
                    tuple_3D[1] = torch.cat((tuple_3D[1], rgb_points[:,:6].unsqueeze(0)), dim=0)
        yield list(tuple_2D) + tuple_3D

class FullLoader2D_3D(FullLoader):
    """
    Dataloader for the full data available in the WDS shards,
    matching 2D renders with 3D pointcloud on-the-fly.
    Adapt and filter to the fields needed for your usage.

    Args:
    ----
        ...:                See FullLoader
        root_url_2D:        The root URL of the 2D data.
        root_dir_3D:        The root directory of the 3D data.
        split:              One of {train, valid, test}
        semantic_level:     One of {fine, coarse}
        num_points:         Number of points to sample.
        n_compositions:     Number of compositions to sample.
        pc_transform:       Transformations to apply to the pointclouds.
        pc_half_precision:  Use half precision floats for pointclouds loading.
        normalize_points:   Normalize pointclouds.
    """

    def __init__(
        self,
        *args,
        root_url_2D,
        root_dir_3D,
        split,
        semantic_level,
        num_points,
        n_compositions,
        pc_transform=None,
        pc_half_precision=False,
        normalize_points=False,
        random=True,
        **kwargs,
    ):
        # Raise error if n_compositions > 10
        if n_compositions > 10:
            raise ValueError("n_compositions must be <= 10.")
        super().__init__(
            *args,
            split=split,
            semantic_level=semantic_level,
            n_compositions=n_compositions,
            root_url=root_url_2D,
            **kwargs,
        )
        self.loader_3D = StylizedShapeLoader_PC(
            root_dir=root_dir_3D,
            split=split,
            semantic_level=semantic_level,
            num_points=num_points,
            transform=pc_transform,
            half_precision=pc_half_precision,
            normalize_points=normalize_points,
            is_rgb=True,
            random=random,
        )

        self.fetch_3D = partial(fetch_3D_PC_batch, self.loader_3D, 6, False)

    def make_loader(self, batch_size, num_workers):
        # Instantiating dataset
        # compose() applies the composed transformation to the dataset
        dataset = super().make_dataset(batch_size).compose(self.fetch_3D)

        # Instantiating loader
        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
        )
        print(self.dataset_size, batch_size)
        # Defining loader length
        loader.length = self.dataset_size // batch_size

        return loader


class EvalLoader2D_3D(EvalLoader):
    """
    Dataloader for the unnanotated test data.

    Args:
    ----
        ...:                See FullLoader2D_3D
    """

    def __init__(
        self,
        *args,
        root_url_2D,
        root_dir_3D,
        split,
        semantic_level,
        num_points,
        n_compositions,
        pc_transform=None,
        pc_half_precision=False,
        normalize_points=False,
        random=True,
        **kwargs,
    ):
        # Raise error if n_compositions > 10
        if n_compositions > 10:
            raise ValueError("n_compositions must be <= 10.")
        super().__init__(
            *args,
            root_url=root_url_2D,
            split=split,
            semantic_level=semantic_level,
            n_compositions=n_compositions,
            **kwargs,
        )
        self.loader_3D = EvalLoader_PC(
            root_dir=root_dir_3D,
            split=split,
            semantic_level=semantic_level,
            num_points=num_points,
            transform=pc_transform,
            half_precision=pc_half_precision,
            normalize_points=normalize_points,
            random=random,
            is_rgb=True,
        )
        self.fetch_3D = partial(fetch_3D_PC_batch, self.loader_3D, 3, True)

    def make_loader(self, batch_size, num_workers):
        # Instantiating dataset
        dataset = super().make_dataset(batch_size).compose(self.fetch_3D)

        # Instantiating loader
        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
        )

        # Defining loader length
        loader.length = self.dataset_size // batch_size

        return loader
