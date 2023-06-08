"""
Writing an HDF5 submission following the challenge format.
"""
import h5py
import tqdm
import numpy as np
from submission_utils import open_hdf5

# from my_cool_model import MyModel

N_POINTS = 2048
MAX_GROUPS = 30


hdf5_name = "/home/ubuntu/3dcompat/workspace/submission/shape_preds_zeropads.hdf5"

f = open_hdf5(hdf5_name, 'r')
print(f.keys())

print(f['shape_preds'].shape, f['part_labels'].shape, f['mat_labels'].shape, f['part_mat_pairs'].shape, f['point_grouping'].shape)
# train_hdf5['shape_preds'][k]    = shape_preds_np[k]
# train_hdf5['part_labels'][k]    = np.zeros(N_POINTS)
# train_hdf5['mat_labels'][k]     = np.zeros(N_POINTS)
# train_hdf5['part_mat_pairs'][k] = np.zeros(MAX_GROUPS * 2) #part_mat_pairs
# train_hdf5['point_grouping'][k] = np.zeros(N_POINTS) #point_grouping