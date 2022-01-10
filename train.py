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
from dataset import SDFData
from network import ImplicitNet, LossFunction
from common_tools.utils import read_json, draw_colored_points_to_obj, AverageMeter, SimpleLogger
from output_geometry import marching_cube

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
    gpuid = config["gpu_id"]
    if gpuid is not None:
        device = torch.device(f"cuda:{gpuid}" if use_cuda else "cpu")
    else:
        device = torch.device(f"cuda" if use_cuda else "cpu")

    print("device: ", device)

    ## directories for monitoring
    saved_dir = config['trainer']['save_dir'] ## save checkpoint
    log_dir = config['trainer']['log_dir'] ## save logger
    res_dir = config['trainer']['res_dir'] ## add timestamp
    
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir) 
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    shutil.copyfile(config_path, os.path.join(res_dir, 'config.json'))


    ## set random seed manually
    seed = config["seed"]
    torch.manual_seed(seed)


    # dataset = SDFData("test_data_fandisk_post.txt")
    dataset = SDFData(config["data"]["filepath"])
    dataloader = data_utils.DataLoader(
        dataset,
        batch_size=int(config['batchsize']),
        shuffle=True,
        drop_last=False,
    )


    ## this is to instantiate a network defined
    net = ImplicitNet(**config['network'])
    net = net.to(device)

    ## this is to set an pytorch optimizer
    opt = optim.Adam(net.parameters(), **config['optimizer']['args'])
    loss_fn = LossFunction()

    ## main training process
    total_loss_avg = AverageMeter()
    log = SimpleLogger(res_dir=res_dir, kwarg_list=['epoch', 'loss'], make_file=True)


    ## main training process
    epochs = config['epoch']
    for epoch in range(epochs):
        net.train() # set network to the train mode
        total_loss_avg.reset()
        for out in dataloader:
            opt.zero_grad()

            points_b = out["samples"].to(device)
            sdfs_b = out["sdfs"].to(device)
            
            pred = net(points_b)
            # reshape the pred; you need to check this torch function -- torch.squeeze() -- out
            # compute loss for this batch
            """
            attention: this loss is different from eq.9 in DeepSDF
            """
            loss = loss_fn(pred, sdfs_b)
            loss.backward()
            opt.step()

            total_loss_avg.update(loss.item())
        
        ## log
        log.update(total_loss_avg.val, 'loss')
        log.update(epoch, 'epoch')
        log.flush_out()

        ## evaluation
        with torch.no_grad():
            if epoch%config["trainer"]["save_period"]==0 and epoch!=0:
                try:
                    print("marching cube")
                    filename = f"output3dshape_{epoch}.ply"
                    filename = os.path.join(res_dir, filename)
                    marching_cube(filename, net, config['geometry']['resolution'])
                    print("geometry of epoch ",epoch," is generated.")
                except ValueError:
                    print("fail to catch zero level set.")
                
                pth_path = os.path.join(res_dir, f"checkpoint_epoch{str(epoch)}.pth")
                torch.save(net.state_dict(), pth_path)
                print("model of epoch ",epoch," has been saved.")

        