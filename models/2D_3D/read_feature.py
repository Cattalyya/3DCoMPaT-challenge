"""
Writing an HDF5 submission following the challenge format.
"""
import h5py
import tqdm
import numpy as np

# from my_cool_model import MyModel

N_POINTS = 2048
MAX_GROUPS = 30


feature_file = "/home/ubuntu/3dcompat/workspace/3DCoMPaT-v2/models/2D_3D/seg2d_logits.hdf5"
# feature_file = "/home/ubuntu/3dcompat/features/seg2d_logits_valid.hdf5"

f = h5py.File(feature_file, 'r')
print(f.keys())

print(f['partseg_2dlogits'].shape, f['matseg_2dlogits'].shape)
display_range = range(6740, 6770)
print("Part unique: ",np.min(f['partseg_2dlogits']), np.max(f['partseg_2dlogits']))
#print(f['partseg_2dlogits'][4000][1000][3])
print("Mat unique: ",np.min(f['matseg_2dlogits']), np.max(f['matseg_2dlogits']))
#print(f['matseg_2dlogits'][4000][1000][3])
