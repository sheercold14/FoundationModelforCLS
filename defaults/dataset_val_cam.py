from utils import *
from .bases import BaseSet
from scipy.io import mmread
from torchvision.transforms import ToTensor, ToPILImage


DATA_INFO = {
              "DDSM": {"dataset_location": "DDSM"},
              "CheXpert": {"dataset_location": "CheXpert"},
              "ISIC2019": {"dataset_location": "ISIC2019"},
              "APTOS2019": {"dataset_location": "APTOS2019"},
              "Camelyon": {"dataset_location": "Camelyon"},    
              "Tumor": {"dataset_location": "Tumor"}
}
class TUMOR_All_organ_save_val(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    mean = (0.716, 0.650, 0.623)
    std = (0.204, 0.273, 0.290)
    int_to_labels = {
        0: 'benign',
        1: 'malignant',
    }
    int_to_organ_labels = {
        0: 'Thyroid',
        1: 'Lung',
        2: 'Breast'
    }
    target_metric = 'roc_auc'
    knn_nhood = 5
    n_classes = len(int_to_labels)
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    organs_label_to_int = {val: key for key, val in int_to_organ_labels.items()}
    
    def __init__(self, dataset_params, mode='train',folder=None):
        self.attr_from_dict(dataset_params)
        self.mode = mode
        self.val_folder = folder
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
    def add_organ_info(self,q_dict,organ_name):
        q_dict['organ_label'] = organ_name
        return q_dict
    def get_data_as_list(self):
        data_list = []
        datainfo = [self.add_organ_info(json.loads(q),'Thyroid') for q in open(os.path.expanduser('/data/lishichao/project/Foundation-Medical/data/v2/medical_json_label.jsonl'), "r")]
        Breast_info = [self.add_organ_info(json.loads(q),'Breast') for q in open(os.path.expanduser('/data/lishichao/project/Foundation-Medical/data/v2/Breast_json_label_folder.jsonl'))] 
        Lung_info = [self.add_organ_info(json.loads(q),'Lung') for q in open(os.path.expanduser('/data/lishichao/project/Foundation-Medical/data/v2/Lung_json_label_folder.jsonl'))] 
        Thyroid_info = [self.add_organ_info(json.loads(q),'Thyroid') for q in open(os.path.expanduser('/data/lishichao/project/Foundation-Medical/data/v2/Thyroid_json_label_folder.jsonl'))] 
        datainfo = datainfo + Breast_info + Lung_info + Thyroid_info
        data = [{'img_path': datainfo[i]['image'], 'label': self.labels_to_int[datainfo[i]['label']], 'organ_label':self.organs_label_to_int[datainfo[i]['organ_label']],'dataset': self.name} for i in range(len(datainfo))]
        folder_list = ['0','1','2','3','4']
        train_list = [x for x in folder_list if x != self.val_folder]
    
        train_idxs = [i for i in range(len(data)) if datainfo[i]["split"] in train_list]
        val_idxs = [i for i in range(len(data)) if datainfo[i]["split"] in [str(self.val_folder)]]    
        test_idxs = [i for i in range(len(data)) if datainfo[i]["split"] in [str(self.val_folder)]] 
           
        if self.mode == 'train':
            data = [data[i] for i in train_idxs]
        elif self.mode in ['val', 'eval']:
            data = [data[i] for i in val_idxs]
        else:
            data = [data[i] for i in test_idxs]
        
        return data 
    def __getitem__(self, idx): 
        
        img_path = self.data[idx]['img_path']
        label = torch.as_tensor(self.data[idx]['label'])
        organ_label = torch.as_tensor(self.data[idx]['organ_label'])
        
        png_path = '.'.join(img_path.split('.')[:-1]) + '.png'
        name = img_path.split('/')[-1].split('.')[-2]
        if os.path.exists(png_path):
            img = self.get_x(png_path)
            img_path = png_path
        else:
            img = self.get_x(img_path)

        if self.resizing is not None:
            img = self.resizing(img)

        if self.transform is not None:
            if isinstance(self.transform, list):
                img = [tr(img) for tr in self.transform]
            else:
                if self.is_multi_crop:
                    img = self.multi_crop_aug(img, self.transform)
                else:
                    img = [self.transform(img) for _ in range(self.num_augmentations)]
            img = img[0] if len(img) == 1 and isinstance(img, list) else img      
        if self.mode == 'train':
            return img, label, organ_label
        else:
            return img, label, organ_label, name
class TUMOR_Thyroid_3_save_val(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    mean = (0.716, 0.650, 0.623)
    std = (0.204, 0.273, 0.290)
    int_to_labels = {
        0: 'benign',
        1: 'malignant',
    }
    target_metric = 'roc_auc'
    knn_nhood = 5
    n_classes = len(int_to_labels)
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    
    def __init__(self, dataset_params, mode='train',folder=None):
        self.attr_from_dict(dataset_params)
        self.mode = mode
        self.val_folder = folder
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
        
    def get_data_as_list(self):
        data_list = []
        Thyroid_info = [json.loads(q) for q in open(os.path.expanduser('/data/lishichao/project/Foundation-Medical/data/v2/medical_json_label.jsonl'))] 
        datainfo = Thyroid_info
        data = [{'img_path': datainfo[i]['image'], 'label': self.labels_to_int[datainfo[i]['label']], 'dataset': self.name} for i in range(len(datainfo))]
        folder_list = ['0','1','2','3','4']
        train_list =  [x for x in folder_list if x != self.val_folder]
        train_idxs = [i for i in range(len(data)) if datainfo[i]["split"] in train_list]
        val_idxs = [i for i in range(len(data)) if datainfo[i]["split"] in [str(self.val_folder)]]    
        test_idxs = [i for i in range(len(data)) if datainfo[i]["split"] in [str(self.val_folder)]]
           
        if self.mode == 'train':
            data = [data[i] for i in train_idxs]
        elif self.mode in ['val', 'eval']:
            data = [data[i] for i in val_idxs]
        else:
            data = [data[i] for i in test_idxs]
    
        return data 
    def __getitem__(self, idx): 
        
        img_path = self.data[idx]['img_path']
        label = torch.as_tensor(self.data[idx]['label'])
        
        png_path = '.'.join(img_path.split('.')[:-1]) + '.png'
        name = img_path.split('/')[-1].split('.')[-2]
        if os.path.exists(png_path):
            img = self.get_x(png_path)
            img_path = png_path
        else:
            img = self.get_x(img_path)

        if self.resizing is not None:
            img = self.resizing(img)

        if self.transform is not None:
            if isinstance(self.transform, list):
                img = [tr(img) for tr in self.transform]
            else:
                if self.is_multi_crop:
                    img = self.multi_crop_aug(img, self.transform)
                else:
                    img = [self.transform(img) for _ in range(self.num_augmentations)]
            img = img[0] if len(img) == 1 and isinstance(img, list) else img      
        if self.mode == 'train':
            return img, 'thyroid',label
        else:
            return img, label,'thyroid',name
class TUMOR_Thyroid_clean_save_val(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    mean = (0.716, 0.650, 0.623)
    std = (0.204, 0.273, 0.290)
    int_to_labels = {
        0: 'benign',
        1: 'malignant',
    }
    target_metric = 'roc_auc'
    knn_nhood = 5
    n_classes = len(int_to_labels)
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    
    def __init__(self, dataset_params, mode='train',folder=None):
        self.attr_from_dict(dataset_params)
        self.mode = mode
        self.val_folder = folder
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
        
    def get_data_as_list(self):
        data_list = []
        Thyroid_info = [json.loads(q) for q in open(os.path.expanduser('/data/lishichao/project/Foundation-Medical/data/clean/thyroid_clean.jsonl'))] 
        datainfo = Thyroid_info
        data = [{'img_path': datainfo[i]['image'], 'label': self.labels_to_int[datainfo[i]['label']], 'dataset': self.name} for i in range(len(datainfo))]
        folder_list = ['0','1','2','3','4']
        train_list =  [x for x in folder_list if x != self.val_folder]
        train_idxs = [i for i in range(len(data)) if datainfo[i]["split"] in train_list]
        val_idxs = [i for i in range(len(data)) if datainfo[i]["split"] in [str(self.val_folder)]]    
        test_idxs = [i for i in range(len(data)) if datainfo[i]["split"] in [str(self.val_folder)]]
           
        if self.mode == 'train':
            data = [data[i] for i in train_idxs]
        elif self.mode in ['val', 'eval']:
            data = [data[i] for i in val_idxs]
        else:
            data = [data[i] for i in test_idxs]
    
        return data 
    def __getitem__(self, idx): 
        
        img_path = self.data[idx]['img_path']
        label = torch.as_tensor(self.data[idx]['label'])
        
        png_path = '.'.join(img_path.split('.')[:-1]) + '.png'
        name = img_path.split('/')[-1].split('.')[-2]
        if os.path.exists(png_path):
            img = self.get_x(png_path)
            img_path = png_path
        else:
            img = self.get_x(img_path)

        if self.resizing is not None:
            img = self.resizing(img)

        if self.transform is not None:
            if isinstance(self.transform, list):
                img = [tr(img) for tr in self.transform]
            else:
                if self.is_multi_crop:
                    img = self.multi_crop_aug(img, self.transform)
                else:
                    img = [self.transform(img) for _ in range(self.num_augmentations)]
            img = img[0] if len(img) == 1 and isinstance(img, list) else img      
        if self.mode == 'train':
            return img, 'thyroid',label
        else:
            return img, label,'thyroid',name