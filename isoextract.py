## 
import os
import numpy as np
import argparse
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as data_utils
from dataset import Polygon3DSample
from network import ImplicitNet, LossFunction
from common_tools.utils import read_json, draw_colored_points_to_obj, AverageMeter, SimpleLogger, catch_nan
from common_tools.geometry import read_obj_file, write_obj_file

from torch.autograd import grad
import time

# from pytorch3d.ops.knn import _KNN, knn_points

# def _create_tree(points_padded: torch.Tensor, num_points_per_cloud=None):
#     """
#     create a data structure, per-point cache knn-neighbor
#     Args:
#         points_padded (N,P,D)
#         num_points_per_cloud list
#     """
#     knn_k = 8
#     assert (points_padded.ndim == 3)
#     if num_points_per_cloud is None:
#         num_points_per_cloud = torch.tensor([points_padded.shape[1]] * points_padded.shape[0],
#                                             device=points_padded.device, dtype=torch.long)
#     knn_result = knn_points(
#         points_padded, points_padded, num_points_per_cloud, num_points_per_cloud,
#         K=knn_k + 1, return_nn=True, return_sorted=True)
#     return knn_result


# ## g_sdf_resample
# def repulsive_update(init_points, knn_results, num_points, normals):
#     normals = F.normalize(normals, dim=-1)
#     knn_nn = knn_results.knn[..., 1:, :]
#     diag = (init_points.view(-1, 3).max(dim=0).values -
#             init_points.view(-1, 3).min(0).values).norm().item()
#     number = num_points.item()
#     difference = init_points[:, :, None, :] - knn_nn
#     inv_sigma_spatial = number / diag
#     distance = torch.sum(difference ** 2, dim=-1)
#     spatial = torch.exp(-distance * inv_sigma_spatial)
#     difference_proj = difference - (difference * normals[:, :, None, :]).sum(dim=-1, keepdim=True) * normals[:, :, None, :]
#     move = torch.sum(spatial[..., None] * difference_proj, dim=-2)
#     points = init_points + move
#     return points

def repulsive_update(points, gradients, K=8):
    ## TODO: need to change to knn
    np_points = points.detach().cpu().numpy()
    sqdist_matrix = np.sum((np_points[:,None,:] - np_points[None,:,:])**2, axis=-1)
    knn_sqdist, knn_idx = torch.topk(torch.tensor(sqdist_matrix, device=points.device), dim=-1, k=K+1, largest=False)
    del sqdist_matrix

    knn_sqdist = knn_sqdist[:, 1:]
    knn_idx = knn_idx[:, 1:] ## exclude self
    knn_points = points[knn_idx,:] ## [N, K, 3]
    diag = (points.max(dim=0).values - points.min(dim=0).values).norm() ## sample points aabb's diagonal length
    num_points = points.shape[0]
    inv_sigma_spatial = (num_points / diag)*1.5 ## magic number 2.0
    spatial = torch.exp(-knn_sqdist*inv_sigma_spatial) ## some kind of weights
    difference = points[:,None,:] - knn_points
    difference_proj = difference - (difference * gradients[:,None,:]).sum(dim=-1, keepdim=True) * gradients[:,None,:]
    moves = torch.sum(spatial[..., None] * difference_proj, dim=-2)
    return moves


def project_to_surface(points, model, max_iter=10):
    for iter in range(max_iter):
        sdfs = model(points)
        grad = model.grad_compute(points, outputs=sdfs, detach=True)
        points = points - (grad * sdfs)
        if abs(sdfs).max() < 0.0001:
            print(f"converged at {iter}")
            return points
    print("max dist to surface", abs(sdfs).max())
    return points


## compute gradient
def grad_compute(x, outputs, model):
    outputs = model(x)
    d_points = torch.ones_like(outputs, requires_grad=False, device=x.device)
    ori_grad = grad(
        outputs=outputs,
        inputs=x,
        grad_outputs=d_points,  
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )  
    points_grad = ori_grad[0]
    return points_grad



def parse_args():
    parser = argparse.ArgumentParser(description="Modeling multi-body motion with neural implicits")
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    args = parser.parse_args()
    return args



if __name__ == "__main__":

    ## get config
    args = parse_args()
    config_path = args.cfg
    if os.path.isfile(config_path):
        config = read_json(config_path)
        print("==== config ====")
        for k, v in config.items():
            print(k, v)
        print("==== config ====")
    else:
        print("no config file")
        exit()

    ## use cuda or not?
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")    
    print("device: ", device)

    ## directories for monitoring
    saved_dir = config['trainer']['save_dir'] ## save checkpoint
    log_dir = config['trainer']['log_dir'] ## save logger
    res_dir = config['trainer']['res_dir'] ## add timestamp
    
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    shutil.copyfile(config_path, os.path.join(res_dir, 'config.json'))

    ## load network
    net = ImplicitNet(dim=3, **config["network"]["implicit_args"])
    net.load_state_dict(torch.load("saved/checkpoint_epoch500.pth"))
    net = net.to(device)

    ## initialize the samples
    N = int(5e3)
    points_n = torch.randn((N, 3))*0.3 ## normal 
    points_u = (torch.rand((N, 3)) - 0.5)*0.8 ## uniform
    points = torch.cat([points_n, points_u], dim=0)
    # dataset = Polygon3DSample(**config["data"])
    # points = torch.tensor(dataset.vertices, dtype=torch.float32)
    points.requires_grad = True ## requires grad (pytorch 1.8)
    points = points.to(device)
    points = project_to_surface(points, net, 20)

    ## iterative solving
    time1 = time.time()
    for i in range(201):
        print(i)

        sdfs = net(points)
        gradients = net.grad_compute(points, outputs=sdfs, detach=True)     
        moves = repulsive_update(points, gradients, K=20)
        
        if i%25 == 0 and True:
            write_obj_file(f"saved_pc{i}_after.obj", V=points)
        
        points += moves
        points = project_to_surface(points, net, 20)

        if i%100 == 0 and True:
            write_obj_file(f"saved_pc{i}_before.obj", V=points)
        
        
    time2 = time.time()
    print((time2 - time1)/100)
