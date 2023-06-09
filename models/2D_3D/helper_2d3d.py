import torch
import numpy as np


# Modified from https://colab.research.google.com/drive/1wr0CUhict2xn8umvCh7da3iX0ku058h4#scrollTo=FCrax0nI67xQ
def project_to_2D(pointcloud, cam_parameters):
    """
    Project 3D pointcloud to 2D image plane.
    """
    K, M = cam_parameters.chunk(2, axis=0)
    proj_matrix = K @ M
    pc = np.concatenate([pointcloud, np.ones([pointcloud.shape[0], 1])], axis=1).T

    # Applying the projection matrix
    pc = np.matmul(proj_matrix, pc)
    pc_final = pc/pc[2]

    return pc_final.T

def valid_pixel(x, y, imsize):
    return x >= 0 and y >= 0 and x < imsize and y < imsize 

def get_logits_from2d(batch_points, logits_2d, cam_parameters):
    logits_3d = None
    OOB_LOGIT = torch.zeros(logits_2d.shape[1]).cpu()
    imsize = logits_2d.shape[-1]
    for i, points in enumerate(batch_points):
        pixels = np.rint(project_to_2D(points, cam_parameters[i])[:, :-2])
        logit = torch.stack([ logits_2d[i,:,int(x), int(y)] if valid_pixel(x, y, imsize) else OOB_LOGIT for x, y in pixels])
        if logits_3d == None:
            logits_3d = logit.unsqueeze(0)
        else:
            logits_3d = torch.cat((logits_3d, logit.unsqueeze(0)), dim=0)
    return logits_3d


def logits_to_prediction(logits):
    return torch.argmax(logits, -1)

def update_part_logits(saved_results, shape_ids, style_ids, fused_logits):
    style_ids = style_ids.cpu().data.numpy()
    for i, shape_id in enumerate(shape_ids):
        key = (shape_id, style_ids[i])
        accu_logits = saved_results.get(key, torch.zeros((fused_logits.shape)).cpu())
        saved_results[key] = accu_logits + fused_logits[i]
    return saved_results

def update_predictions(shape_ids, style_ids, predicted_cls, predicted_parts, saved_cls_predictions, saved_part_predictions):
    style_ids = style_ids.cpu().data.numpy()
    for i, shape_id in enumerate(shape_ids):
        if shape_id in saved_cls_predictions:
            continue
        key = (shape_id, style_ids[i])
        saved_cls_predictions[key] = predicted_cls[i]
        saved_part_predictions[key] = predicted_parts[i]
    return saved_cls_predictions, saved_part_predictions

def get_fused_prediction(fused_logits, predicted3d):
    fused_prediction = logits_to_prediction(fused_logits)
    fused_prediction -= torch.ones(fused_prediction.shape, dtype=int)
    fused_prediction_np = fused_prediction.numpy()
    fail2d_mask = fused_prediction_np == -1
    fused_prediction_np[fail2d_mask] = predicted3d[fail2d_mask]
    return fused_prediction_np