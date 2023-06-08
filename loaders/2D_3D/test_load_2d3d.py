import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../2D/')))
from compat2D import GCRLoader
from compat2D import wds_identity
from compat2D_3D import FullLoader2D_3D
import argparse
from tqdm import tqdm

def parse_args(argv):
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(
        description="Training the Baseline Models on 2D Part and Material Segmentation"
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=60,
        help="Number of epochs to be used for training",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="fine",
        help="Semantic level to be used for training",
    )
    parser.add_argument(
        "--task", type=str, default="part", help="Type of task to be used for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs to be used for training",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="Name of the model file",
    )
    # parser.add_argument(
    #     "--root_url",
    #     type=str,
    #     required=True,
    #     help="Root URL of the 2D data",
    # )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="segformer",
        help="Name of the model file",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size to be used for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate to be used for training",
    )
    parser.add_argument(
        "--n_comp",
        type=int,
        default=1,
        help="Number of components to be used for training",
    )

    parser.add_argument(
        "--npoint",
        type=int,
        default=2048,
        help="Number of point clouds",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="Optimizer to be used for training",
    )
    args = parser.parse_args(argv)

    # Printing input arguments
    print("Input arguments:")
    print(args)

    return args

def project_to_2D(pointcloud, proj_matrix):
    """
    Project 3D pointcloud to 2D image plane.
    """
    pc = np.concatenate([pointcloud, np.ones([pointcloud.shape[0], 1])], axis=1).T

    # Applying the projection matrix
    pc = np.matmul(proj_matrix, pc)
    pc_final = pc/pc[2]

    return pc_final.T

def main(argv=None):
	args = parse_args(argv)
	root_url_2D = "/home/ubuntu/3dcompat/workspace/3DCoMPaT-v2/download/3DCoMPaT_2D/"
	root_dir_3D = os.path.join(os.getcwd(),"../../models/3D/" "data/" + args.data_type + "_grained/")
	num_workers = 6

	valid_loader_2D_3D = (
	    FullLoader2D_3D(root_url_2D=root_url_2D,
	                    root_dir_3D=root_dir_3D,
	                    num_points=2048,
	                    split="valid",
	                    semantic_level=args.data_type,
	                    n_compositions=args.n_comp)
	    .make_loader(batch_size=args.batch_size, num_workers=num_workers)
	)

	# Unpacking the huge data tuple
	for data_tuple in valid_loader_2D_3D:
	    shape_id, image, target, \
	    part_mask, mat_mask, depth, \
	    style_id, view_id, view_type, \
	    cam_parameters,\
	    points, points_part_labels, points_mat_labels = data_tuple
	    break

	for loaded_data in tqdm(loader):
		# print(loaded_data)
		break

if __name__ == "__main__":
    main()
