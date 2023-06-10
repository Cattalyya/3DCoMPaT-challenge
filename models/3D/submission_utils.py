"""
Writing an HDF5 submission following the challenge format.
"""
import h5py
import tqdm
import numpy as np

# from my_cool_model import MyModel

N_POINTS = 2048
MAX_GROUPS = 30
DEFAULT_OUTFILE = "/home/ubuntu/3dcompat/workspace/submission/parts_fused_pred_test.hdf5"
        
def open_hdf5(hdf5_file, mode):
    hdf5_f = h5py.File(hdf5_file, mode)
    return hdf5_f

class Submission:
    def __init__(self, n_shapes, outpath=DEFAULT_OUTFILE):
        # Creating the selected split
        self.n_shapes = n_shapes
        self.outpath = outpath
        self.MAX_UINT8 = 255
        self.column_names = ["shape_preds", "part_labels", "mat_labels", "part_mat_pairs", "point_grouping"]
        with h5py.File(outpath, 'w') as file:
            file.create_dataset('shape_preds',
                                        shape=(n_shapes),
                                        dtype='uint8',
                                        fillvalue=self.MAX_UINT8)
            file.create_dataset('part_labels',
                                        shape=(n_shapes, N_POINTS),
                                        dtype='int16',
                                        fillvalue=-1)
            file.create_dataset('mat_labels',
                                        shape=(n_shapes, N_POINTS),
                                        dtype='uint8',
                                        fillvalue=self.MAX_UINT8)
            file.create_dataset('part_mat_pairs',
                                        shape=(n_shapes, MAX_GROUPS, 2),
                                        dtype='int16',
                                        fillvalue=-1)
            file.create_dataset('point_grouping',
                                        shape=(n_shapes, N_POINTS),
                                        dtype='uint8',
                                        fillvalue=self.MAX_UINT8)

        self.visited = {cn: set() for cn in self.column_names}
    # sid = unique shape_id
    def update_cl(self, i, cl, key):
        # print("cl=", int(cl))
        self.out_hdf5['shape_preds'][i]= int(cl)
        assert key not in self.cls_loaded
        self.cls_loaded.add(key)
        assert cl <= 41 and cl >= 0

    def update_parts(self, i, parts, key):
        assert len(parts.shape) == 1
        assert parts.shape[0] == N_POINTS
        assert key not in self.parts_loaded
        self.parts_loaded.add(key)
        # print("parts=", parts)
        self.out_hdf5['part_labels'][i] = parts

    def update_batched(self, column_name, batched_preds, test_order_map):
        with h5py.File(self.outpath, 'a') as file:
            for key, val in batched_preds.items():
                index = test_order_map[key]
                file[column_name][index] = val
                assert key not in self.visited[column_name], \
                    "Error saving column={} submission: Key {} already existed in {}".format(column_name, key, self.visited[column_name])
                self.visited[column_name].add(key)

    def sanity_check(self):
        column_checks = ["shape_preds", "part_labels", "mat_labels"]
        NULLs = [self.MAX_UINT8, -1]
        assert len(self.visited[column_checks[0]]) == len(self.visited[column_checks[1]])
        with h5py.File(self.outpath, 'r') as file:
            for i, column_name in enumerate(column_checks):
                column_data = file[column_name]
                assert ~np.isin(NULLs[i], column_data), \
                    "Column {} contains null ({}) data: {}".format(column_name, NULLs[i], np.unique(column_data))
        print("[INFO] Pass submission sanity check!")

    def write_file(self, checked_sids=None):
        self.sanity_check(checked_sids)
        self.out_hdf5.close()

def write_result(shape_preds):
# # Instantiating the dataloader
# data_loader = ... # You must use pre-extracted 3D pointclouds
#                   # for evaluation on the test or validation sets.
#                   # Samples and predictions must be read and written in sequence,
#                   # without any shuffling.
	n_shapes = shape_preds.shape[0]
	shape_preds_np = shape_preds.cpu().numpy()
	print("Writing result for {} shapes...".format(n_shapes))

	# If the file exists, open it instead
	hdf5_name = "/home/ubuntu/3dcompat/workspace/submission/shape_preds_zeropads.hdf5"
	#get_hdf5_name(hdf5_dir, hdf5_name="my_submission")

	# Creating the selected split
	train_hdf5 = open_hdf5(hdf5_name, mode='w')
	train_hdf5.create_dataset('shape_preds',
	                            shape=(n_shapes),
	                            dtype='uint8')
	train_hdf5.create_dataset('part_labels',
	                            shape=(n_shapes, N_POINTS),
	                            dtype='int16')
	train_hdf5.create_dataset('mat_labels',
	                            shape=(n_shapes, N_POINTS),
	                            dtype='uint8')
	train_hdf5.create_dataset('part_mat_pairs',
	                            shape=(n_shapes, MAX_GROUPS, 2),
	                            dtype='int16')
	train_hdf5.create_dataset('point_grouping',
	                            shape=(n_shapes, N_POINTS),
	                            dtype='uint8')


	# Iterating over the test set
	for k in tqdm.tqdm(range(n_shapes)):
	    # shape_id, style_id, pointcloud = data_loader[k]
	    
	    # Forward through your model
	    # shape_preds, point_part_labels, point_mat_labels, part_mat_pairs, point_grouping = \
	    #     MyModel(shape_id, style_id, pointcloud)

	    # If you don't want to predict point groupings/part material pairs yourself,
	    # you can simply fill both matrices with -1

	    # Write the entries
	    train_hdf5['shape_preds'][k]    = shape_preds_np[k]
	    # train_hdf5['part_labels'][k]    = point_part_labels
	    # train_hdf5['mat_labels'][k]     = point_mat_labels
	    # train_hdf5['part_mat_pairs'][k] = part_mat_pairs
	    # train_hdf5['point_grouping'][k] = point_grouping
	    train_hdf5['part_labels'][k]    = np.ones(N_POINTS) * -1
	    train_hdf5['mat_labels'][k]     = np.ones(N_POINTS) * -1
	    train_hdf5['part_mat_pairs'][k] = np.ones((MAX_GROUPS, 2)) * -1#part_mat_pairs * -1
	    train_hdf5['point_grouping'][k] = np.ones(N_POINTS) * -1 #point_grouping
	    # print(train_hdf5['part_labels'][k])
	# Close the HDF5 file
	train_hdf5.close()


# def write_result(dataloader)
# 	hdf5_dir = "/home/ubuntu/3dcompat/workspace/submission/"
# # # Instantiating the dataloader
# # data_loader = ... # You must use pre-extracted 3D pointclouds
# #                   # for evaluation on the test or validation sets.
# #                   # Samples and predictions must be read and written in sequence,
# #                   # without any shuffling.
# 	n_shapes = len(data_loader)

# 	# If the file exists, open it instead
# 	hdf5_name = "/home/ubuntu/3dcompat/workspace/submission/shape_preds.hdf5"
# 	#get_hdf5_name(hdf5_dir, hdf5_name="my_submission")

# 	# Creating the selected split
# 	train_hdf5 = open_hdf5(hdf5_name, mode='w')
# 	train_hdf5.create_dataset('shape_preds',
# 	                            shape=(n_shapes),
# 	                            dtype='uint8')
# 	train_hdf5.create_dataset('part_labels',
# 	                            shape=(n_shapes, N_POINTS),
# 	                            dtype='int16')
# 	train_hdf5.create_dataset('mat_labels',
# 	                            shape=(n_shapes, N_POINTS),
# 	                            dtype='uint8')
# 	train_hdf5.create_dataset('part_mat_pairs',
# 	                            shape=(n_shapes * MAX_GROUPS, N_POINTS),
# 	                            dtype='int16')
# 	train_hdf5.create_dataset('point_grouping',
# 	                            shape=(n_shapes, N_POINTS),
# 	                            dtype='uint8')


# 	# Iterating over the test set
# 	for k in tqdm.tqdm(range(n_shapes)):
# 	    shape_id, style_id, pointcloud = data_loader[k]
	    
# 	    # Forward through your model
# 	    shape_preds, point_part_labels, point_mat_labels, part_mat_pairs, point_grouping = \
# 	        MyModel(shape_id, style_id, pointcloud)

# 	    # If you don't want to predict point groupings/part material pairs yourself,
# 	    # you can simply fill both matrices with -1

# 	    # Write the entries
# 	    train_hdf5['shape_preds'][k]    = shape_preds
# 	    train_hdf5['part_labels'][k]    = point_part_labels
# 	    train_hdf5['mat_labels'][k]     = point_mat_labels
# 	    train_hdf5['part_mat_pairs'][k] = part_mat_pairs
# 	    train_hdf5['point_grouping'][k] = point_grouping

# 	# Close the HDF5 file
# 	train_hdf5.close()