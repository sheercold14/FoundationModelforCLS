from utils import *
from .bases import BaseSet
from scipy.io import mmread
from torchvision.transforms import ToTensor, ToPILImage
import ast

DATA_INFO = {
              "DDSM": {"dataset_location": "DDSM"},
              "CheXpert": {"dataset_location": "CheXpert"},
              "ISIC2019": {"dataset_location": "ISIC2019"},
              "APTOS2019": {"dataset_location": "APTOS2019"},
              "Camelyon": {"dataset_location": "Camelyon"},    
              "Tumor": {"dataset_location": "Tumor"}
}



class TUMOR_Thyroid_3_front(BaseSet):
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
        Thyroid_info = [json.loads(q) for q in open(os.path.expanduser('/data/lishichao/project/Foundation-Medical/data/v2/thyroid_3_label.jsonl'))] 
        datainfo = Thyroid_info
        root_path = '/data/lishichao/data/hospital_organ/hospital_3/front'
        data = [{'img_path': os.path.join(root_path,datainfo[i]['image']), 'label': self.labels_to_int[datainfo[i]['label']], 'dataset': self.name} for i in range(len(datainfo))]
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
    
class TUMOR_Thyroid_Clean(BaseSet):
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
# 使用 145 张数据   
# class TUMOR_Thyroid_3_Text(BaseSet):
#     img_channels = 3
#     is_multiclass = True
#     task = 'classification'
#     mean = (0.716, 0.650, 0.623)
#     std = (0.204, 0.273, 0.290)
#     int_to_labels = {
#         0: 'benign',
#         1: 'malignant',
#     }
#     target_metric = 'roc_auc'
#     knn_nhood = 5
#     n_classes = len(int_to_labels)
#     labels_to_int = {val: key for key, val in int_to_labels.items()}
    
#     def __init__(self, dataset_params, mode='train',folder=None):
#         self.attr_from_dict(dataset_params)
#         self.mode = mode
#         self.val_folder = folder
#         self.data = self.get_data_as_list()
#         self.transform, self.resizing = self.get_transforms()
        
#     def get_data_as_list(self):
#         data_list = []
#         Thyroid_info = [q for q in json.load(open(os.path.expanduser('/data/lishichao/project/LLaVA-Med/qwen/result/results_hospital_3.json')))] 
#         datainfo = Thyroid_info
#         data = [{'img_path': datainfo[i]['image'], 'label': self.labels_to_int[datainfo[i]['label']], 'text':datainfo[i]['answers'], 'dataset': self.name} for i in range(len(datainfo))]
        
#         folder_list = ['0','1','2','3','4']
#         train_list =  [x for x in folder_list if x != self.val_folder]
#         train_idxs = [i for i in range(len(data)) if datainfo[i]["split"] in train_list]
#         val_idxs = [i for i in range(len(data)) if datainfo[i]["split"] in [str(self.val_folder)]]    
#         test_idxs = [i for i in range(len(data)) if datainfo[i]["split"] in [str(self.val_folder)]]
           
#         if self.mode == 'train':
#             data = [data[i] for i in train_idxs]
#         elif self.mode in ['val', 'eval']:
#             data = [data[i] for i in val_idxs]
#         else:
#             data = [data[i] for i in test_idxs]
    
#         return data 
    
#     def __getitem__(self, idx): 
#         root_path = '/data/lishichao/project/LLaVA-Med/data/train_images'
#         img_path = os.path.join(root_path,self.data[idx]['img_path'])
#         label = torch.as_tensor(self.data[idx]['label'])    
#         img = self.get_x(img_path)
#         text = self.data[idx]['text']

#         if self.resizing is not None:
#             img = self.resizing(img)

#         if self.transform is not None:
#             if isinstance(self.transform, list):
#                 img = [tr(img) for tr in self.transform]
#             else:
#                 if self.is_multi_crop:
#                     img = self.multi_crop_aug(img, self.transform)
#                 else:
#                     img = [self.transform(img) for _ in range(self.num_augmentations)]
#             img = img[0] if len(img) == 1 and isinstance(img, list) else img      

#         return img, label, text

# 合并 202411月甲状腺数据
class TUMOR_Thyroid_3_Text(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    int_to_labels = {
        0: 'benign',
        1: 'malignant',
    }
    target_metric = 'roc_auc'
    knn_nhood = 5
    n_classes = len(int_to_labels)
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225] 
    
    def __init__(self, dataset_params, mode='train',folder=None):
        self.attr_from_dict(dataset_params)
        self.mode = mode
        self.val_folder = folder
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()

        
    def get_data_as_list(self):
        data_list = []
        Thyroid_info = [q for q in json.load(open(os.path.expanduser('/data/lishichao/project/LLaVA-Med/qwen/result/results_hospital_3.json')))] 
        Thyroid_info_11 = [q for q in json.load(open(os.path.expanduser('/data/lishichao/project/LLaVA-Med/qwen/result/Thyroid_hospital_3_202411.json')))] 
        datainfo = Thyroid_info
        rootpath_1 = '/data/lishichao/project/LLaVA-Med/data/train_images'
        rootpath_2 = '/data/lishichao/data/hospital_organ/organ_202411/Thyroid'
        data = [{'img_path': os.path.join(rootpath_1,datainfo[i]['image']), 'label': self.labels_to_int[datainfo[i]['label']], 'text':datainfo[i]['answers'], 'dataset': self.name, 'split':datainfo[i]['split']} for i in range(len(datainfo))]
        # datainfo = Thyroid_info_11
        # data += [{'img_path': os.path.join(rootpath_2,datainfo[i]['image']), 'label': self.labels_to_int[datainfo[i]['label']], 'text':datainfo[i]['answers'], 'dataset': self.name, 'split':datainfo[i]['split']} for i in range(len(Thyroid_info_11))]
        
        folder_list = ['0','1','2','3','4']
        
        if '[' not in self.val_folder:
            train_list =  [x for x in folder_list if x != self.val_folder]
            train_idxs = [i for i in range(len(data)) if data[i]["split"] in train_list]
            val_idxs = [i for i in range(len(data)) if data[i]["split"] in [str(self.val_folder)]]    
            test_idxs = [i for i in range(len(data)) if data[i]["split"] in [str(self.val_folder)]]
        else:
            self.val_folder = ast.literal_eval(self.val_folder)
            train_list =  [int(x) for x in folder_list if x not in self.val_folder]
            train_idxs = [i for i in range(len(data)) if int(data[i]["split"]) in train_list]
            val_idxs = [i for i in range(len(data)) if int(data[i]["split"]) in self.val_folder]   
            test_idxs = [i for i in range(len(data)) if int(data[i]["split"]) in self.val_folder]
           
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
        img = self.get_x(img_path)
        text = self.data[idx]['text']

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
        if self.text_branch:
            return img, label, text
        else:
            return img, label
class TUMOR_Thyroid_3_202411(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    int_to_labels = {
        0: 'benign',
        1: 'malignant',
    }
    target_metric = 'roc_auc'
    knn_nhood = 5
    n_classes = len(int_to_labels)
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225] 
    
    def __init__(self, dataset_params, mode='train',folder=None):
        self.attr_from_dict(dataset_params)
        self.mode = mode
        self.val_folder = folder
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
        
    def get_data_as_list(self):
        data_list = []
        Thyroid_info = [q for q in json.load(open(os.path.expanduser('/data/lishichao/project/LLaVA-Med/qwen/result/results_hospital_3.json')))] 
        Thyroid_info_11 = [q for q in json.load(open(os.path.expanduser('/data/lishichao/project/LLaVA-Med/qwen/result/Thyroid_hospital_3_202411.json')))] 
        datainfo = Thyroid_info_11
        rootpath = '/data/lishichao/data/hospital_organ/organ_202411/Thyroid'
        data = [{'img_path': os.path.join(rootpath,datainfo[i]['image']), 'label': self.labels_to_int[datainfo[i]['label']], 'text':datainfo[i]['answers'], 'dataset': self.name, 'split':datainfo[i]['split']} for i in range(len(datainfo))]
        
        folder_list = ['0','1','2','3','4']
        
        if '[' not in self.val_folder:
            train_list =  [x for x in folder_list if x != self.val_folder]
            train_idxs = [i for i in range(len(data)) if data[i]["split"] in train_list]
            val_idxs = [i for i in range(len(data)) if data[i]["split"] in [str(self.val_folder)]]    
            test_idxs = [i for i in range(len(data)) if data[i]["split"] in [str(self.val_folder)]]
        else:
            self.val_folder = ast.literal_eval(self.val_folder)
            train_list =  [int(x) for x in folder_list if x not in self.val_folder]
            train_idxs = [i for i in range(len(data)) if int(data[i]["split"]) in train_list]
            val_idxs = [i for i in range(len(data)) if int(data[i]["split"]) in self.val_folder]   
            test_idxs = [i for i in range(len(data)) if int(data[i]["split"]) in self.val_folder]
           
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
        img = self.get_x(img_path)
        text = self.data[idx]['text']

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
        if self.text_branch:
            return img, label, text
        else:
            return img, label
import pickle        
class TUMOR_Thyroid_3_Text_quality(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    int_to_labels = {
        0: 'benign',
        1: 'malignant',
    }
    target_metric = 'roc_auc'
    knn_nhood = 5
    n_classes = len(int_to_labels)
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225] 
    
    def __init__(self, dataset_params, mode='train',folder=None):
        self.attr_from_dict(dataset_params)
        self.mode = mode
        self.val_folder = folder
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
        
    def get_data_as_list(self):
        data_list = []
        # """要删除的代码"""
        # pickle_file_path = '/data/lishichao/project/Foundation-Medical/data/v2/clean.pkl'
        # with open(pickle_file_path, 'rb') as f:
        #     clean_list = pickle.load(f)
            
        Thyroid_info = [q for q in json.load(open(os.path.expanduser('/data/lishichao/project/LLaVA-Med/qwen/result/thyroid_3_quality_1.json')))] 
    
        datainfo = Thyroid_info
        rootpath = '/data/lishichao/data/hospital_organ/hospital_3/quality/Thyroid_set1'

        data = [{'img_path': os.path.join(rootpath,datainfo[i]['image']), 'label': self.labels_to_int[datainfo[i]['label']], 'text':datainfo[i]['answers'], 'dataset': self.name, 'split':datainfo[i]['split']} for i in range(len(datainfo))]
       
        folder_list = ['0','1','2','3','4']
        # train_list =  [x for x in folder_list if x != self.val_folder]
        # train_idxs = [i for i in range(len(data)) if data[i]["split"] in train_list]
        # val_idxs = [i for i in range(len(data)) if data[i]["split"] in [str(self.val_folder)]]    
        # test_idxs = [i for i in range(len(data)) if data[i]["split"] in [str(self.val_folder)]]
        
        if '[' not in self.val_folder:
            train_list =  [x for x in folder_list if x != self.val_folder]
            train_idxs = [i for i in range(len(data)) if data[i]["split"] in train_list]
            val_idxs = [i for i in range(len(data)) if data[i]["split"] in [str(self.val_folder)]]    
            test_idxs = [i for i in range(len(data)) if data[i]["split"] in [str(self.val_folder)]]
        else:
            self.val_folder = ast.literal_eval(self.val_folder)
            train_list =  [int(x) for x in folder_list if x not in self.val_folder]
            train_idxs = [i for i in range(len(data)) if int(data[i]["split"]) in train_list]
            val_idxs = [i for i in range(len(data)) if int(data[i]["split"]) in self.val_folder]   
            test_idxs = [i for i in range(len(data)) if int(data[i]["split"]) in self.val_folder]
           
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
        img = self.get_x(img_path)
        text = self.data[idx]['text']

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
        if self.text_branch:
            return img, label, text
        else:
            return img, label

class Hospital_Organ_Multilabel(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225] 
    int_to_labels = {
        0: 'benign',
        1: 'malignant',
    }
    target_metric = 'roc_auc'
    knn_nhood = 5
    n_classes = len(int_to_labels)
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    
    def __init__(self, dataset_params, mode='train', folder=None):
        self.attr_from_dict(dataset_params)
        self.mode = mode
        self.val_folder = folder
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
        
    def get_data_as_list(self):
        data_list = []
        datainfo = [q for q in json.load(open(os.path.expanduser('/data/lishichao/project/LLaVA-Med/qwen/result/hospital3_all_organ_doctor.json')))] 
        data = [{'img_path': datainfo[i]['image'], 'label': self.labels_to_int[datainfo[i]['label']],'split':datainfo[i]['split'],'text':datainfo[i]['answers']} for i in range(len(datainfo))]

        folder_list = ['0','1','2','3','4']
        
        if '[' not in self.val_folder:
            train_list =  [x for x in folder_list if x != self.val_folder]
            train_idxs = [i for i in range(len(data)) if data[i]["split"] in train_list]
            val_idxs = [i for i in range(len(data)) if data[i]["split"] in [str(self.val_folder)]]    
            test_idxs = [i for i in range(len(data)) if data[i]["split"] in [str(self.val_folder)]]
        else:
            self.val_folder = ast.literal_eval(self.val_folder)
            train_list =  [int(x) for x in folder_list if x not in self.val_folder]
            train_idxs = [i for i in range(len(data)) if int(data[i]["split"]) in train_list]
            val_idxs = [i for i in range(len(data)) if int(data[i]["split"]) in self.val_folder]   
            test_idxs = [i for i in range(len(data)) if int(data[i]["split"]) in self.val_folder]
           
        if self.mode == 'train':
            data = [data[i] for i in train_idxs]
        elif self.mode in ['val', 'eval']:
            data = [data[i] for i in val_idxs]
        else:
            data = [data[i] for i in test_idxs]
    
        return data 
    def __getitem__(self, idx): 
        root_path = '/data/lishichao/data/hospital_organ/hospital3_organ_merged'
        img_path = os.path.join(root_path,self.data[idx]['img_path'])
        label = torch.as_tensor(self.data[idx]['label'])    
        img = self.get_x(img_path)
        text = self.data[idx]['text']

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
        if self.text_branch:
            return img, label, text
        else:
            return img, label

class Hospital_Thyroid(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225] 
    int_to_labels = {
        0: 'benign',
        1: 'malignant',
    }
    target_metric = 'roc_auc'
    knn_nhood = 5
    n_classes = len(int_to_labels)
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    
    def __init__(self, dataset_params, mode='train', folder=None):
        self.attr_from_dict(dataset_params)
        self.mode = mode
        self.val_folder = folder
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
        
    def get_data_as_list(self):
        data_list = []
        datainfo = [q for q in json.load(open(os.path.expanduser('/data/lishichao/project/LLaVA-Med/qwen/result/hospital3_all_thyroid.json')))] 
        data = [{'img_path': datainfo[i]['image'], 'label': self.labels_to_int[datainfo[i]['label']],'split':datainfo[i]['split'],'text':datainfo[i]['answers']} for i in range(len(datainfo))]

        folder_list = ['0','1','2','3','4']
        
        if '[' not in self.val_folder:
            train_list =  [x for x in folder_list if x != self.val_folder]
            train_idxs = [i for i in range(len(data)) if data[i]["split"] in train_list]
            val_idxs = [i for i in range(len(data)) if data[i]["split"] in [str(self.val_folder)]]    
            test_idxs = [i for i in range(len(data)) if data[i]["split"] in [str(self.val_folder)]]
        else:
            self.val_folder = ast.literal_eval(self.val_folder)
            train_list =  [int(x) for x in folder_list if x not in self.val_folder]
            train_idxs = [i for i in range(len(data)) if int(data[i]["split"]) in train_list]
            val_idxs = [i for i in range(len(data)) if int(data[i]["split"]) in self.val_folder]   
            test_idxs = [i for i in range(len(data)) if int(data[i]["split"]) in self.val_folder]
           
        if self.mode == 'train':
            data = [data[i] for i in train_idxs]
        elif self.mode in ['val', 'eval']:
            data = [data[i] for i in val_idxs]
        else:
            data = [data[i] for i in test_idxs]
    
        return data 
    def __getitem__(self, idx): 
        root_path = '/data/lishichao/data/hospital_organ/hospital3_thyroid_merged'
        img_path = os.path.join(root_path,self.data[idx]['img_path'])
        label = torch.as_tensor(self.data[idx]['label'])    
        img = self.get_x(img_path)
        text = self.data[idx]['text']

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
        if self.text_branch:
            return img, label, text
        else:
            return img, label
