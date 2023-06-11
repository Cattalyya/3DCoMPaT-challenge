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
submission_file = "/home/ubuntu/3dcompat/workspace/3DCoMPaT-v2/models/2D_3D/submission.hdf5"
submission_file_best = "/home/ubuntu/3dcompat/workspace/3DCoMPaT-v2/models/2D_3D/cls_rand.hdf5"

fbest = open_hdf5(submission_file_best, 'r')
f = open_hdf5(submission_file, 'r')
print(f.keys())

print(f['shape_preds'].shape, f['part_labels'].shape, f['mat_labels'].shape, f['part_mat_pairs'].shape, f['point_grouping'].shape)
display_range = range(6740, 6770)
print("=== Shape ===")
print("Curr unique: ",np.unique(f['shape_preds']))
print("Best unique: ",np.unique(fbest['shape_preds']))
print([f['shape_preds'][i] for i in display_range])
print([fbest['shape_preds'][i] for i in display_range])

print("=== Part ===")
print("Curr unique: ",np.unique(f['part_labels']))
print("Best unique: ", np.unique(fbest['part_labels']))
print(f['part_labels'][0])
print(fbest['part_labels'][0])

print("=== Mat ===")
print("Curr unique: ", np.unique(f['mat_labels']))
print("Best unique: ", np.unique(fbest['mat_labels']))

# print([f['shape_preds'][i] for i in range(30)], np.unique(f['shape_preds']), np.unique(f['part_labels']))
# train_hdf5['shape_preds'][k]    = shape_preds_np[k]
# train_hdf5['part_labels'][k]    = np.zeros(N_POINTS)
# train_hdf5['mat_labels'][k]     = np.zeros(N_POINTS)
# train_hdf5['part_mat_pairs'][k] = np.zeros(MAX_GROUPS * 2) #part_mat_pairs
# train_hdf5['point_grouping'][k] = np.zeros(N_POINTS) #point_grouping