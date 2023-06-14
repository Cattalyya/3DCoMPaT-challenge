"""
Writing an HDF5 submission following the challenge format.
"""
import h5py
import tqdm
import numpy as np

# from my_cool_model import MyModel

N_POINTS = 2048
MAX_GROUPS = 30
DEFAULT_OUTFILE = "/home/ubuntu/3dcompat/workspace/3DCoMPaT-v2/models/2D_3D/seg2d_logits.hdf5"
NULL = 9999

def open_hdf5(hdf5_file, mode):
    hdf5_f = h5py.File(hdf5_file, mode)
    return hdf5_f
# active_columns = [True, True, False, False, False]
class FeatureFile:
    def __init__(self, n_shapes, n_parts, n_mats, active_columns, outpath=DEFAULT_OUTFILE):
        # Creating the selected split
        self.n_shapes = n_shapes
        self.outpath = outpath
        self.column_names = ["partseg_2dlogits", "matseg_2dlogits"]
        self.active_columns = active_columns
        n_views = 8
        with h5py.File(outpath, 'w') as file:
            file.create_dataset('partseg_2dlogits',
                                        shape=(n_shapes, N_POINTS, n_views+1, n_parts), # add sum up
                                        dtype='float',
                                        fillvalue=NULL)
            file.create_dataset('matseg_2dlogits',
                                        shape=(n_shapes, N_POINTS, n_views+1, n_mats),
                                        dtype='float',
                                        fillvalue=NULL)

        self.visited = {cn: set() for cn in self.column_names}

    def update_batched(self, column_name, batched_preds, test_order_map):
        with h5py.File(self.outpath, 'a') as file:
            for key, val in batched_preds.items():
                index = test_order_map[key]
                file[column_name][index] = val
                assert key not in self.visited[column_name], \
                    "Error saving column={} submission: Key {} already existed in {}".format(column_name, key, self.visited[column_name])
                self.visited[column_name].add(key)

    def sanity_check(self, split):
        EXPECT_SIZE = 12560 if split == "test" else 6770 if "val" else 80760 if "train" else None
        # assert len(self.visited[ACTIVE_PREDICTIONS[0]]) == len(self.visited[ACTIVE_PREDICTIONS[1]])
        with h5py.File(self.outpath, 'r') as file:
            for i, column_name in enumerate(self.column_names):
                if not self.active_columns[i]:
                    continue
                column_data = file[column_name]
                assert ~np.isin(NULL, column_data), \
                    "Column {} contains null ({}) data: {}. An example null-value key is {}. Data={}".format(\
                        column_name, NULL, np.unique(column_data),\
                        np.where(column_data==NULL), [s for s in column_data])

                assert len(self.visited[column_name]) == EXPECT_SIZE, \
                    "Predicted result of column {} {} does NOT equal to expected ds size {}".format( \
                    column_name, len(self.visited[column_names]), EXPECT_SIZE)
        print("[INFO] Pass submission sanity check!")

    def write_file(self, checked_sids=None):
        self.sanity_check(checked_sids)
        self.out_hdf5.close()
