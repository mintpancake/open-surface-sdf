import open3d as o3d
import numpy as np
from skimage import measure
import trimesh
import os

from tensorboardX import SummaryWriter ## this is for lei's conda env
import torch
# from chamferdist import ChamferDistance
import argparse
import shutil
from network import Net
from common_tools.utils import read_json
import time


def marching_cube_np(filename, sdf_func, resolution, extra_levels:list = None):
    
    levels=[0]
    if not extra_levels is None:
        assert isinstance(extra_levels, list)
        levels = levels + extra_levels

    low, high = -5, 5
    grid = np.linspace(low,high, resolution)

    xg, yg, zg = np.meshgrid(grid, grid, grid, indexing='ij')
    xg = xg.reshape(-1)
    yg = yg.reshape(-1)
    zg = zg.reshape(-1)

    net_input = np.stack((zg, yg, xg), axis=-1)
    net_input = net_input.reshape((resolution*resolution*resolution,3))
    
    max_len=net_input.shape[0]
    batchsize=100000

    with torch.no_grad():
        output_list = []
        i = 0
        status = True
        while status:
            start_id = i*batchsize
            end_id = (i+1)*batchsize
            if max_len < end_id:
                end_id = max_len
                status = False

            cur_input=net_input[start_id:end_id]
            cur_output=sdf_func(cur_input)
            output_list.append(cur_output)
            i = i+1

    net_output = np.concatenate(output_list, axis=0)
    net_output=net_output.reshape(resolution,resolution,resolution)
    net_output=np.transpose(net_output,(2,1,0))

    one_axis_points=net_output.shape[0]-1
    meshexport_list = []
    for lvl in levels:
        print(lvl)
        verts, faces, normals, values = measure.marching_cubes_lewiner(volume=net_output,level=lvl,spacing=(1/one_axis_points,1/one_axis_points,1/one_axis_points))
        vertices = verts + np.array([-0.5, -0.5, -0.5])
        meshexport = trimesh.Trimesh(vertices, faces, vertex_normals=normals, process=False)
        filename_save = os.path.splitext(filename)[0] + f"_{lvl}.obj"
        meshexport.export(filename_save)
        print("saved")
        meshexport_list.append(meshexport)


def marching_cube(filename, model, resolution, extra_levels:list = None):
    
    levels=[0]
    if not extra_levels is None:
        assert isinstance(extra_levels, list)
        levels = levels + extra_levels

    low, high = -0.5, 0.5
    grid = np.linspace(low,high, resolution)

    #####################################
    """
    use this to replace the time-consuming for loop
    """
    xg, yg, zg = np.meshgrid(grid, grid, grid, indexing='ij')
    xg = xg.reshape(-1)
    yg = yg.reshape(-1)
    zg = zg.reshape(-1)

    net_input = np.stack((zg, yg, xg), axis=-1)
    net_input = torch.FloatTensor(net_input)
    net_input = net_input.reshape((resolution*resolution*resolution,3))
    
    max_len=net_input.shape[0]
    batchsize=100000

    with torch.no_grad():
        output_list = []
        i = 0
        status = True
        while status:
            start_id = i*batchsize
            end_id = (i+1)*batchsize
            if max_len < end_id:
                end_id = max_len
                status = False

            cur_input=net_input[start_id:end_id].cuda()
            cur_output=model(cur_input).cpu()
            output_list.append(cur_output)
            i = i+1

    net_output = torch.cat(output_list, dim=0)
    net_output=net_output.reshape(resolution,resolution,resolution)
    net_output=net_output.numpy()
    net_output=np.transpose(net_output,(2,1,0))

    one_axis_points=net_output.shape[0]-1

    meshexport_list = []
    for lvl in levels:
        print("marching level:", lvl)
        filename_save = os.path.splitext(filename)[0] + f"_{lvl}.obj"

        verts, faces, normals, values = measure.marching_cubes_lewiner(volume=net_output,level=lvl,spacing=(1/one_axis_points,1/one_axis_points,1/one_axis_points))
        vertices = verts + np.array([-0.5, -0.5, -0.5])
        meshexport = trimesh.Trimesh(vertices, faces, vertex_normals=normals, process=False)
        meshexport.export(filename_save)
        print("saved")
        meshexport_list.append(meshexport)


def parse_args():
    parser = argparse.ArgumentParser(description="get geometry of an implicit function represented as a neural network")
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--resume',
                        help='path to a trained model',
                        required=True,
                        type=str)
    parser.add_argument('--resolution',
                        default=256,
                        help='marching cube resolution',
                        type=int
                        )
    args = parser.parse_args()
    return args



if __name__ == "__main__":

    ## get config
    args = parse_args()

    resume_pth = args.resume
    assert os.path.isfile(resume_pth)

    config_path = args.cfg
    exp_type=config_path.split(".")[0]
    if os.path.isfile(config_path):
        config = read_json(config_path)
        #print("==== config ====")
        #for k, v in config.items():
        #    print(k, v)
        #print("==== config ====")
    else:
        print("no config file")
        exit()

    ## use cuda or not?
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")    
    print("device: ", device)

    time_info=time.strftime("%m%d_%Hh%Mmin", time.localtime())
    res_dir = os.path.join(config['trainer']['save_dir']+"_test", time_info)

    ## build a folder to save everthing of this experiment
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    shutil.copyfile(config_path, os.path.join(res_dir, 'config.json')) ## copy the config file

    # ## load test data for computing chamfer distance
    # test_data=load_test_data(config['data']['shape_name'],config['trainer']['test_dir'])
    # print("load test data:", test_data.shape)

    net=Net(**config['network'])
    net = net.to(device)

    net.load_state_dict(torch.load(resume_pth, map_location=device))

    time1 = time.time()
    filename = os.path.join(res_dir, "mesh_res.ply")
    resolution = args.resolution
    marching_cube(filename, net, resolution)
    time2 = time.time()
    print(time2-time1)
