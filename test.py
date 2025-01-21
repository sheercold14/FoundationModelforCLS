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

global debug


def parse_arguments():
    parser = argparse.ArgumentParser(description='The main takes as \
                             argument the parameters dictionary from a json file')
    parser.add_argument('--params_path', type=str, required=False, 
                        default="./params.json",
                        help='Give the path of the json file which contains the training parameters')
    parser.add_argument('--checkpoint', type=str, required=False, help='Give a valid checkpoint name')
   
    return parser.parse_args()


def save_config(params_file, save_dir, model_name):
    path = os.path.join(save_dir, "configs", model_name) 
    check_dir(path)
    shutil.copy(params_file,path)



def main(parameters, args):
    
    # define system
    define_system_params(parameters.system_params)

    wrapper = DINOWrapper(parameters)
    wrapper.instantiate()
    
 
    trainer = Trainer(wrapper)
        
    trainer.test()     
    
if __name__ == '__main__':
    args = parse_arguments()
    parameters = edict(load_params(args))
    try:
        launch(main, (parameters, args))
    except Exception as e:       
        if dist.is_initialized():
            dist.destroy_process_group()            
        raise e
    finally:
        if dist.is_initialized():
            synchronize()         