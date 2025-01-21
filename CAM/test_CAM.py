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
# image = Image.open('/data/lishichao/project/LLaVA-Med/data/train_images/1473897B_N-W1.jpg')
# img_tensor = transforms.ToTensor()(image)

""" Model wrapper to return a tensor"""
class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

""" Translate the category name to the category index.
    Some models aren't trained on Imagenet but on even larger datasets,
    so we can't just assume that 761 will always be remote-control.

"""
def category_name_to_index(model, category_name):
    name_to_index = dict((v, k) for k, v in model.config.id2label.items())
    return name_to_index[category_name]
    
""" Helper function to run GradCAM on an image and create a visualization.
    (note to myself: this is probably useful enough to move into the package)
    If several targets are passed in targets_for_gradcam,
    e.g different categories,
    a visualization for each of them will be created.
    
"""
def run_grad_cam_on_image(model: torch.nn.Module,
                          target_layer: torch.nn.Module,
                          targets_for_gradcam: List[Callable],
                          reshape_transform: Optional[Callable],
                          input_tensor: torch.nn.Module,
                          input_image: Image,
                          method: Callable=GradCAM):
    with method(model=model,
                 target_layers=[target_layer],
                 reshape_transform=reshape_transform) as cam:

        # Replicate the tensor for each of the categories we want to create Grad-CAM for:
        repeated_tensor = input_tensor[None, :].repeat(len(targets_for_gradcam), 1, 1, 1)

        batch_results = cam(input_tensor=repeated_tensor,
                            targets=targets_for_gradcam)
        results = []
        for grayscale_cam in batch_results:
            visualization = show_cam_on_image(np.float32(input_image)/255,
                                              grayscale_cam,
                                              use_rgb=True)
            # Make it weight less in the notebook:
            visualization = cv2.resize(visualization,
                                       (visualization.shape[1]//2, visualization.shape[0]//2))
            results.append(visualization)
        return np.hstack(results)
    
    
def print_top_categories(model, img_tensor, top_k=5):
    logits = model(img_tensor.unsqueeze(0))
    indices = logits.cpu()[0, :].detach().numpy().argsort()[-top_k :][::-1]
    for i in indices:
        print(f"Predicted class {i}: {model.config.id2label[i]}")

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
model.eval()
image = Image.open('/data/lishichao/project/Foundation-Medical/results/im1.png')
# img_arr = np.array(image) * 255
img_arr = np.array(image)
img_tensor = torch.FloatTensor(img_arr).cuda()
outputs, features = model(img_tensor.permute(2,0,1).unsqueeze(0), return_embedding = True)
print(outputs)

target_layer = [model.main_model.backbone.blocks[-1].norm1]
#target_layer = [model.main_model.backbone.norm]
targets_for_gradcam = [ClassifierOutputTarget(0),ClassifierOutputTarget(1)]
# from functools import partial
# reshape_transform = partial(swinT_reshape_transform_huggingface,
#                             width=img_tensor.shape[2]//32,
#                             height=img_tensor.shape[1]//32)
# img = Image.fromarray(run_dff_on_image(model=model,
#                           target_layer=model.main_model.backbone.fc,
#                           classifier=model.main_model.fc,
#                           img_pil=image,
#                           img_tensor=img_tensor.permute(2,0,1),
#                           n_components=4,
#                           top_k=2))


# list_of_images = []
# for i in model.main_model.backbone.fc:
#     for j in i.layers:
#         target_layer = j
#         list_of_images.append(Image.fromarray(run_grad_cam_on_image(model=model,
#                       target_layer=target_layer,
#                       targets_for_gradcam=targets_for_gradcam,
#                       reshape_transform=None)))

# print(list_of_images)
# run_grad_cam_on_image()


# img = Image.fromarray(run_grad_cam_on_image(model=model,
#                       target_layer=model.main_model.backbone.fc,
#                       targets_for_gradcam=targets_for_gradcam,
#                       input_tensor=img_tensor.permute(2,0,1),
#                       input_image=image,
#                       reshape_transform=None))


# img.save('cam_results.png')
def reshape_transform(tensor, height=16, width=16):
    # 去掉cls token
    result = tensor[:, 1:, :].reshape(tensor.size(0),
    height, width, tensor.size(2))
    # 将通道维度放到第一个位置
    result = result.permute(0,3,1,2)
    return result
cam = GradCAM(model=model,target_layers=target_layer,reshape_transform=reshape_transform)
# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=img_tensor.permute(2,0,1).unsqueeze(0), targets=[targets_for_gradcam[0]])
# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(img_arr / 255, grayscale_cam, use_rgb=True)
img = Image.fromarray(visualization)
img.save('cam_results_1_0.png')