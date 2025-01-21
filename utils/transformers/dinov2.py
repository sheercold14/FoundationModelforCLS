import torch
import os
def dinov2_vits14():
    return torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
def download_dinov2_model():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    return model
def dinov2_vitb14():
    model_cache_path = '/home/lishichao/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth'
    # 检查模型是否已经缓存
    if os.path.exists(model_cache_path):
        # 创建模型架构
        #model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', source='github', pretrained=False)
        model = torch.hub.load('/data/lishichao/project/dinov2', 'dinov2_vitb14', source='local', pretrained=False)

        # 从本地缓存加载权重      
        model.load_state_dict(torch.load(model_cache_path))
        print("Model loaded from local cache.")
    else:
        # 如果缓存不存在，下载并保存到缓存
        print("Model cache not found, downloading...")
        model = download_dinov2_model()
    return model
    #return torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    #return torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg',weights='/data/lishichao/project/Foundation-Medical/dinov2_vitb14_reg4_pretrain.pth')
    # return torch.hub.load('/data/lishichao/project/Foundation-Medical/dinov2_vitb14_reg4_pretrain.pth','dinov2_vitb14',source='local')

def dinov2_vitl14():
    return torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

def dinov2_vitg14():
    return torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')