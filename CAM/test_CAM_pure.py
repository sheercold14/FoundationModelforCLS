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

"""GradCAM"""
import warnings
warnings.filterwarnings('ignore')
from torchvision import transforms
from pytorch_grad_cam import run_dff_on_image, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import cv2
import torch
from typing import List, Callable, Optional


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
checkpoint = torch.load('/data/lishichao/project/Foundation-Medical/results/v4/checkpoints/All_sheet18_withorgan_lr5e4/0/roc_auc')
model = wrapper.model
model.load_state_dict(checkpoint['state_dict'])
model.to(torch.cuda.current_device())
model.eval()
image = Image.open('/data/lishichao/project/Foundation-Medical/results/im0.png')
img_arr = np.array(image)
img_tensor = torch.FloatTensor(img_arr).cuda()
outputs, features = model(img_tensor.permute(2,0,1).unsqueeze(0), return_embedding = True)

target_layer = [model.main_model.backbone.blocks[-1].norm1]
targets_for_gradcam = [ClassifierOutputTarget(0),ClassifierOutputTarget(1)]

def reshape_transform(tensor, height=16, width=16):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
    height, width, tensor.size(2))
    result = result.permute(0,3,1,2)
    return result
cam = GradCAM(model=model,target_layers=target_layer,reshape_transform=reshape_transform)
grayscale_cam = cam(input_tensor=img_tensor.permute(2,0,1).unsqueeze(0), targets=[targets_for_gradcam[0]])

grayscale_cam = grayscale_cam[0, :]

visualization = show_cam_on_image(img_arr / 255, grayscale_cam, use_rgb=True)
img = Image.fromarray(visualization)
img.save('/data/lishichao/project/Foundation-Medical/CAM/cam_results_1_1.png')