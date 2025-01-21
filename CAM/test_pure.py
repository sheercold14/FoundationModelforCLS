import os
import sys
import json
import wandb
import argparse

import torch
import numpy as np
from defaults import *
from utils.system_def import *
from utils.launch import dist, launch, synchronize
from utils.helpfuns import check_dir
from PIL import Image 


def parse_arguments():
    parser = argparse.ArgumentParser(description='The main takes as \
                             argument the parameters dictionary from a json file')
    parser.add_argument('--params_path', type=str, required=False, 
                        default="./params.json",
                        help='Give the path of the json file which contains the training parameters')
    parser.add_argument('--checkpoint', type=str, required=False, help='Give a valid checkpoint name')
   
    return parser.parse_args()
"""My Model"""
args = parse_arguments()
parameters = edict(load_params(args))
wrapper = DINOWrapper(parameters)
wrapper.instantiate()
checkpoint = torch.load('/data/lishichao/project/Foundation-Medical/results/checkpoints/TUMOR_test')
model = wrapper.model
model.load_state_dict(checkpoint['state_dict'])
model.to(torch.cuda.current_device())
image = Image.open('/data/lishichao/project/Foundation-Medical/results/im0.png')
img_arr = np.array(image) * 255
img_tensor = torch.FloatTensor(img_arr).cuda()
outputs, features = model(img_tensor.permute(2,0,1).unsqueeze(0), return_embedding = True)
print(outputs)